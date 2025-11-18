const API_URL = "/api/detect";

const toggleBtn = document.getElementById("toggle");
const statusEl = document.getElementById("status");
const alarmStateEl = document.getElementById("alarm-state");
const scoreEl = document.getElementById("score");

const AudioCtx = window.AudioContext || window.webkitAudioContext;
const audioContext = new AudioCtx();
let mediaStream;
let listening = false;
let sourceNode;
let processorNode;
let silentGain;
let chunkSamples = 0;
const bufferQueue = [];
let bufferedSamples = 0;
const CHUNK_SECONDS = 1;

async function ensurePermissions() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Your browser does not support microphone access");
  }
  mediaStream = mediaStream || (await navigator.mediaDevices.getUserMedia({ audio: true }));
  if ("Notification" in window && Notification.permission === "default") {
    await Notification.requestPermission();
  }
}

function updateUi(state, statusText) {
  alarmStateEl.textContent = state;
  alarmStateEl.classList.remove("alert");
  statusEl.textContent = statusText;
  toggleBtn.textContent = listening ? "Stop Listening" : "Start Listening";
  toggleBtn.classList.toggle("listening", listening);
}

function setupProcessor() {
  if (processorNode) {
    return;
  }
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processorNode = audioContext.createScriptProcessor(4096, 1, 1);
  silentGain = audioContext.createGain();
  silentGain.gain.value = 0;
  sourceNode.connect(processorNode);
  processorNode.connect(silentGain);
  silentGain.connect(audioContext.destination);
  chunkSamples = Math.floor(audioContext.sampleRate * CHUNK_SECONDS);
  processorNode.onaudioprocess = handleAudioProcess;
}

function handleAudioProcess(event) {
  if (!listening) {
    return;
  }
  const input = event.inputBuffer.getChannelData(0);
  enqueueSamples(new Float32Array(input));
  while (bufferedSamples >= chunkSamples) {
    const samples = dequeueSamples(chunkSamples);
    processSamples(samples).catch((error) => {
      console.error(error);
      updateUi("Error", "Unable to process audio");
    });
  }
}

async function processSamples(samples) {
  const wavBuffer = encodePcmToWav(samples, audioContext.sampleRate);
  const blob = new Blob([wavBuffer], { type: "audio/wav" });
  await sendToBackend(blob);
}

function enqueueSamples(chunk) {
  bufferQueue.push({ data: chunk, offset: 0 });
  bufferedSamples += chunk.length;
}

function dequeueSamples(targetLength) {
  const result = new Float32Array(targetLength);
  let written = 0;
  while (written < targetLength && bufferQueue.length) {
    const entry = bufferQueue[0];
    const available = entry.data.length - entry.offset;
    const toCopy = Math.min(available, targetLength - written);
    result.set(entry.data.subarray(entry.offset, entry.offset + toCopy), written);
    written += toCopy;
    entry.offset += toCopy;
    if (entry.offset >= entry.data.length) {
      bufferQueue.shift();
    }
  }
  bufferedSamples -= written;
  return result;
}

function encodePcmToWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const clipped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
    offset += 2;
  }

  return buffer;
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i += 1) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

async function sendToBackend(blob) {
  const formData = new FormData();
  formData.append("file", blob, "clip.wav");

  const response = await fetch(API_URL, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Detection request failed");
  }

  const data = await response.json();
  const state = data.detected ? "🚨 Fire alarm" : "No alarm";
  const score = (data.score || 0).toFixed(2);
  scoreEl.textContent = score;
  updateUi(state, listening ? "Listening..." : "Stopped");

  if (data.detected) {
    triggerNotification(score);
  }
}

function triggerNotification(score) {
  const body = `Confidence: ${score}`;
  if ("Notification" in window && Notification.permission === "granted") {
    new Notification("Fire alarm detected", { body });
  }
  alarmStateEl.classList.add("alert");
  setTimeout(() => alarmStateEl.classList.remove("alert"), 1200);
}

async function startListening() {
  try {
    await ensurePermissions();
    await audioContext.resume();
  } catch (error) {
    console.error(error);
    updateUi("Permission needed", error.message);
    return;
  }

  setupProcessor();
  listening = true;
  updateUi("Listening", "Streaming audio to detector");
}

function stopListening() {
  listening = false;
  bufferQueue.length = 0;
  bufferedSamples = 0;
  if (processorNode) {
    processorNode.disconnect();
    processorNode = null;
  }
  if (silentGain) {
    silentGain.disconnect();
    silentGain = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  chunkSamples = 0;
  updateUi("Stopped", "Microphone idle");
}

function setup() {
  toggleBtn.addEventListener("click", async () => {
    if (!listening) {
      await startListening();
    } else {
      stopListening();
    }
  });

  updateUi("Not listening", "Microphone idle");
}

setup();
