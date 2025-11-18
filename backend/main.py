import io
import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForAudioClassification, AutoProcessor


MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
KEYWORDS = (
    "fire alarm",
    "smoke alarm",
    "smoke detector",
    "fire detector",
    "carbon monoxide detector",
    "alarm bell",
    "fire alarm bell",
    "siren",
    "evacuation horn",
)

SECONDARY_KEYWORDS = ("bell", "alarm clock", "clang", "buzzer")
SECONDARY_MIN_SCORE = 0.2


class FireAlarmDetector:
    """Wraps a pre-trained AST AudioSet classifier for fire-alarm detection."""

    def __init__(self, threshold: float = 0.4) -> None:
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.sample_rate = int(getattr(self.processor, "sampling_rate", 16000))
        self.id2label = {
            int(idx): label
            for idx, label in self.model.config.id2label.items()
        }
        label_map = {idx: label.lower() for idx, label in self.id2label.items()}
        self.target_indices = [
            idx
            for idx, label in label_map.items()
            if any(keyword in label for keyword in KEYWORDS)
        ]
        if not self.target_indices:
            raise RuntimeError("Unable to find fire-alarm related labels in the model")
        self.threshold = threshold

    def _prepare_waveform(self, audio_bytes: bytes) -> torch.Tensor:
        try:
            audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as exc:  # pragma: no cover - conversion errors bubble up
            logger.warning("Audio decode failed: %s", exc)
            raise HTTPException(status_code=400, detail="Unsupported audio format") from exc

        if audio_array.ndim == 1:
            waveform = torch.from_numpy(audio_array).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_array.T)

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # The model works best for multi-second clips. Zero-pad if clip is too short
        min_length = self.sample_rate * 2
        if waveform.size(1) < min_length:
            pad = min_length - waveform.size(1)
            waveform = F.pad(waveform, (0, pad))

        return waveform[:, : self.sample_rate * 10]

    def _to_processor_inputs(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        array = waveform.squeeze().numpy()
        inputs = self.processor(
            array,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def predict(self, audio_bytes: bytes) -> tuple[bool, float, torch.Tensor, list[tuple[str, float]]]:
        waveform = self._prepare_waveform(audio_bytes)
        inputs = self._to_processor_inputs(waveform)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        all_probs = torch.sigmoid(logits)[0]
        probabilities = all_probs[self.target_indices]
        score = torch.max(probabilities).item()
        top_values, top_indices = torch.topk(all_probs, k=5)
        top_predictions = [
            (self.id2label[idx.item()], top_values[i].item())
            for i, idx in enumerate(top_indices)
        ]
        return score >= self.threshold, score, waveform, top_predictions


class TonalAlarmHeuristic:
    def __init__(
        self,
        min_frequency: int = 600,
        max_frequency: int = 4000,
        ratio_threshold: float = 12.0,
        min_rms: float = 0.02,
    ) -> None:
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.ratio_threshold = ratio_threshold
        self.min_rms = min_rms

    def predict(self, waveform: torch.Tensor, sample_rate: int) -> tuple[bool, float]:
        data = waveform.squeeze().cpu().numpy()
        if data.size < sample_rate:
            return False, 0.0

        max_window = min(data.size, sample_rate * 2)
        window = data[:max_window]
        rms = float(np.sqrt(np.mean(window**2)))
        if rms < self.min_rms:
            return False, 0.0

        window = window * np.hanning(window.size)
        spectrum = np.abs(np.fft.rfft(window))
        freqs = np.fft.rfftfreq(window.size, d=1 / sample_rate)
        mask = (freqs >= self.min_frequency) & (freqs <= self.max_frequency)
        if not np.any(mask):
            return False, 0.0

        focus = spectrum[mask]
        if focus.size == 0:
            return False, 0.0

        peak = float(focus.max())
        floor = float(focus.mean()) + 1e-8
        ratio = peak / floor
        score = min(1.0, ratio / self.ratio_threshold)
        return ratio >= self.ratio_threshold, score


def has_secondary_alarm(predictions: list[tuple[str, float]]) -> bool:
    for label, score in predictions:
        if score < SECONDARY_MIN_SCORE:
            continue
        lower = label.lower()
        if any(keyword in lower for keyword in SECONDARY_KEYWORDS):
            return True
    return False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fire_alarm")

app = FastAPI(title="Fire Alarm Detector", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = FireAlarmDetector()
tonal_detector = TonalAlarmHeuristic()

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"


@app.post("/api/detect")
async def detect_fire_alarm(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        model_detected, model_score, waveform, top_predictions = detector.predict(audio_bytes)
        tonal_detected, tonal_score = tonal_detector.predict(waveform, detector.sample_rate)
        tonal_support = tonal_detected and model_score >= 0.15
        secondary_support = tonal_detected and has_secondary_alarm(top_predictions)
        detected = model_detected or tonal_support or secondary_support
        combined_score = max(model_score, tonal_score if tonal_support or secondary_support else 0)
        logger.info(
            "Model top labels: %s | tonal_score=%.3f | secondary=%s",
            [f"{label}:{score:.2f}" for label, score in top_predictions],
            tonal_score,
            secondary_support,
        )
        return {
            "detected": detected,
            "score": combined_score,
            "model_score": model_score,
            "tonal_score": tonal_score,
            "tonal_support": tonal_support,
            "secondary_support": secondary_support,
            "top_predictions": [
                {"label": label, "score": score}
                for label, score in top_predictions
            ],
        }
    except HTTPException as exc:
        logger.warning("Detection failed with client error: %s", exc.detail)
        raise
    except Exception as exc:  # pragma: no cover - bubbled to client
        logger.exception("Unexpected detection failure")
        raise HTTPException(status_code=500, detail="Fire alarm detection failed") from exc


@app.get("/")
async def serve_index() -> FileResponse:
    index_file = frontend_dir / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_file)


app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
