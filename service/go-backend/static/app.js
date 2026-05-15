const CHUNK_DURATION_MS = 30000;

const state = {
  token: localStorage.getItem("token") || "",
  user: JSON.parse(localStorage.getItem("user") || "null"),
  stream: null,
  audioContext: null,
  audioSource: null,
  audioProcessor: null,
  session: null,
  currentResult: null,
  chunkIndex: 0,
  stopping: false,
  startedAt: null,
  timerId: null,
  chunkTimerId: null,
  pollId: null,
  uploadChain: Promise.resolve(),
  wavBuffers: [],
  wavSampleRate: 0,
  wavSampleCount: 0,
};

const el = {
  authView: document.getElementById("authView"),
  recorderView: document.getElementById("recorderView"),
  historyView: document.getElementById("historyView"),
  resultView: document.getElementById("resultView"),
  mainNav: document.getElementById("mainNav"),
  recordNavBtn: document.getElementById("recordNavBtn"),
  historyNavBtn: document.getElementById("historyNavBtn"),
  userEmail: document.getElementById("userEmail"),
  logoutBtn: document.getElementById("logoutBtn"),
  authMessage: document.getElementById("authMessage"),
  recordMessage: document.getElementById("recordMessage"),
  historyMessage: document.getElementById("historyMessage"),
  email: document.getElementById("email"),
  password: document.getElementById("password"),
  loginBtn: document.getElementById("loginBtn"),
  registerBtn: document.getElementById("registerBtn"),
  taskNr: document.getElementById("taskNr"),
  taskAsr: document.getElementById("taskAsr"),
  taskDiar: document.getElementById("taskDiar"),
  language: document.getElementById("language"),
  diarizationMode: document.getElementById("diarizationMode"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  sessionStatus: document.getElementById("sessionStatus"),
  sessionId: document.getElementById("sessionId"),
  chunkCount: document.getElementById("chunkCount"),
  recordTimer: document.getElementById("recordTimer"),
  processingBlock: document.getElementById("processingBlock"),
  resultBlock: document.getElementById("resultBlock"),
  transcript: document.getElementById("transcript"),
  segments: document.getElementById("segments"),
  newSessionBtn: document.getElementById("newSessionBtn"),
  refreshHistoryBtn: document.getElementById("refreshHistoryBtn"),
  sessionList: document.getElementById("sessionList"),
  downloadFullAudioBtn: document.getElementById("downloadFullAudioBtn"),
  downloadTextBtn: document.getElementById("downloadTextBtn"),
  speakerLabelsBlock: document.getElementById("speakerLabelsBlock"),
  speakerLabels: document.getElementById("speakerLabels"),
  saveSpeakerLabelsBtn: document.getElementById("saveSpeakerLabelsBtn"),
};

function authHeaders() {
  return { Authorization: `Bearer ${state.token}` };
}

function showMessage(node, text, type = "ok") {
  node.textContent = text;
  node.className = text ? `message ${type}` : "message hidden";
}

function showAuthenticated() {
  el.authView.classList.add("hidden");
  el.mainNav.classList.remove("hidden");
  el.userEmail.textContent = state.user ? ` ${state.user.email}` : "";
  showRecorder();
}

function showAuth() {
  el.authView.classList.remove("hidden");
  el.recorderView.classList.add("hidden");
  el.historyView.classList.add("hidden");
  el.resultView.classList.add("hidden");
  el.mainNav.classList.add("hidden");
  el.userEmail.textContent = "";
}

function showRecorder() {
  el.authView.classList.add("hidden");
  el.historyView.classList.add("hidden");
  el.resultView.classList.add("hidden");
  el.recorderView.classList.remove("hidden");
  setActiveNav("record");
}

function showHistory() {
  stopPolling();
  el.authView.classList.add("hidden");
  el.recorderView.classList.add("hidden");
  el.resultView.classList.add("hidden");
  el.historyView.classList.remove("hidden");
  setActiveNav("history");
  loadHistory();
}

function setActiveNav(active) {
  el.recordNavBtn.classList.toggle("active", active === "record");
  el.historyNavBtn.classList.toggle("active", active === "history");
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    throw new Error(data?.error || response.statusText);
  }
  return data;
}

async function login() {
  await submitAuth("/api/v1/auth/login", true);
}

async function register() {
  await submitAuth("/api/v1/auth/register", false);
}

async function submitAuth(url, shouldStoreToken) {
  const email = el.email.value.trim();
  const password = el.password.value;
  if (!email || !password) {
    showMessage(el.authMessage, "Email and password are required.", "error");
    return;
  }

  try {
    const data = await requestJson(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (!shouldStoreToken) {
      showMessage(el.authMessage, "Account created. You can log in now.");
      return;
    }

    state.token = data.token;
    state.user = data.user;
    localStorage.setItem("token", state.token);
    localStorage.setItem("user", JSON.stringify(state.user));
    showMessage(el.authMessage, "");
    showAuthenticated();
  } catch (error) {
    showMessage(el.authMessage, error.message, "error");
  }
}

function logout() {
  stopPolling();
  stopTimer();
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
  }
  closeAudioGraph();
  localStorage.clear();
  state.token = "";
  state.user = null;
  state.session = null;
  showAuth();
}

async function createSession() {
  const tasks = {
    nr: el.taskNr.checked,
    asr: el.taskAsr.checked,
    diar: el.taskDiar.checked,
  };
  if (!tasks.nr && !tasks.asr && !tasks.diar) {
    throw new Error("Enable at least one ML task.");
  }

  return requestJson("/api/v1/sessions", {
    method: "POST",
    headers: {
      ...authHeaders(),
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      tasks,
      language: el.language.value,
      diarization_mode: el.diarizationMode.value,
      chunk_duration_sec: 30,
    }),
  });
}

async function startRecording() {
  try {
    showMessage(el.recordMessage, "");
    resetResult();

    state.session = await createSession();
    state.chunkIndex = 0;
    state.stopping = false;
    state.uploadChain = Promise.resolve();
    resetWavBuffer();
    state.startedAt = Date.now();

    state.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    startAudioGraph(state.stream);

    el.sessionId.textContent = state.session.id;
    el.chunkCount.textContent = "0";
    setStatus("recording");
    el.startBtn.disabled = true;
    el.stopBtn.disabled = false;
    startTimer();
    state.chunkTimerId = window.setInterval(() => {
      flushCurrentWavChunk(false);
    }, CHUNK_DURATION_MS);
  } catch (error) {
    showMessage(el.recordMessage, error.message, "error");
    cleanupRecorder();
  }
}

function enqueueChunkUpload(blob, isFinal) {
  const currentIndex = state.chunkIndex;
  state.chunkIndex += 1;
  el.chunkCount.textContent = String(state.chunkIndex);
  state.uploadChain = state.uploadChain.then(() => uploadChunk(blob, currentIndex, isFinal));
}

async function uploadChunk(blob, chunkIndex, isFinal) {
  const form = new FormData();
  form.append("audio", blob, `chunk_${chunkIndex}${fileExtensionForBlob(blob)}`);
  form.append("chunk_index", String(chunkIndex));
  form.append("is_final", String(isFinal));

  try {
    await requestJson(`/api/v1/sessions/${state.session.id}/chunks`, {
      method: "POST",
      headers: authHeaders(),
      body: form,
    });
  } catch (error) {
    throw new Error(`failed to add chunk ${chunkIndex}: ${error.message}`);
  }
}

function fileExtensionForBlob(blob) {
  if (blob.type.includes("wav")) {
    return ".wav";
  }
  if (blob.type.includes("ogg")) {
    return ".ogg";
  }
  if (blob.type.includes("mp4") || blob.type.includes("mpeg")) {
    return ".m4a";
  }
  return ".webm";
}

async function stopRecording() {
  if (!state.stream || state.stopping) {
    return;
  }
  state.stopping = true;
  setStatus("stopping");
  el.stopBtn.disabled = true;
  if (state.chunkTimerId) {
    window.clearInterval(state.chunkTimerId);
    state.chunkTimerId = null;
  }
  flushCurrentWavChunk(true);
  finishRecording();
}

async function finishRecording() {
  stopTimer();
  cleanupRecorder();

  try {
    await state.uploadChain;
    setStatus("processing");
    el.recorderView.classList.add("hidden");
    el.resultView.classList.remove("hidden");
    el.processingBlock.classList.remove("hidden");
    startPolling();
  } catch (error) {
    setStatus("failed");
    showMessage(el.recordMessage, error.message, "error");
    el.recorderView.classList.remove("hidden");
    el.resultView.classList.add("hidden");
    el.startBtn.disabled = false;
  }
}

function cleanupRecorder() {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
  }
  if (state.chunkTimerId) {
    window.clearInterval(state.chunkTimerId);
    state.chunkTimerId = null;
  }
  closeAudioGraph();
  state.stream = null;
  el.startBtn.disabled = false;
  el.stopBtn.disabled = true;
}

function startAudioGraph(stream) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  state.audioContext = new AudioContextClass();
  state.wavSampleRate = state.audioContext.sampleRate;
  state.audioSource = state.audioContext.createMediaStreamSource(stream);
  state.audioProcessor = state.audioContext.createScriptProcessor(4096, 1, 1);

  state.audioProcessor.onaudioprocess = (event) => {
    if (state.stopping) {
      return;
    }
    const input = event.inputBuffer.getChannelData(0);
    state.wavBuffers.push(new Float32Array(input));
    state.wavSampleCount += input.length;
  };

  state.audioSource.connect(state.audioProcessor);
  state.audioProcessor.connect(state.audioContext.destination);
}

function closeAudioGraph() {
  if (state.audioProcessor) {
    state.audioProcessor.disconnect();
    state.audioProcessor.onaudioprocess = null;
  }
  if (state.audioSource) {
    state.audioSource.disconnect();
  }
  if (state.audioContext && state.audioContext.state !== "closed") {
    state.audioContext.close();
  }
  state.audioProcessor = null;
  state.audioSource = null;
  state.audioContext = null;
}

function resetWavBuffer() {
  state.wavBuffers = [];
  state.wavSampleCount = 0;
}

function flushCurrentWavChunk(isFinal) {
  if (!state.session || state.wavSampleCount === 0) {
    return;
  }
  const samples = mergeFloat32Buffers(state.wavBuffers, state.wavSampleCount);
  const blob = encodeWav(samples, state.wavSampleRate);
  resetWavBuffer();
  enqueueChunkUpload(blob, isFinal);
}

function mergeFloat32Buffers(buffers, totalLength) {
  const output = new Float32Array(totalLength);
  let offset = 0;
  for (const buffer of buffers) {
    output.set(buffer, offset);
    offset += buffer.length;
  }
  return output;
}

function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (const sample of samples) {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

function startPolling() {
  stopPolling();
  state.pollId = window.setInterval(loadResult, 3000);
  loadResult();
}

function stopPolling() {
  if (state.pollId) {
    window.clearInterval(state.pollId);
    state.pollId = null;
  }
}

async function loadResult() {
  if (!state.session) {
    return;
  }

  try {
    const data = await requestJson(`/api/v1/sessions/${state.session.id}/result`, {
      headers: authHeaders(),
    });
    const session = data.session;
    setStatus(session.status);

    if (session.status === "failed") {
      stopPolling();
      el.processingBlock.classList.add("hidden");
      showMessage(el.recordMessage, session.error || "Processing failed.", "error");
      el.recorderView.classList.remove("hidden");
      el.resultView.classList.add("hidden");
      return;
    }

    if (session.status === "done") {
      stopPolling();
      el.processingBlock.classList.add("hidden");
      renderFinalResult(parseJsonField(session.final_result), session.id);
    }
  } catch (error) {
    stopPolling();
    el.processingBlock.classList.add("hidden");
    el.resultView.classList.add("hidden");
    el.recorderView.classList.remove("hidden");
    showMessage(el.recordMessage, error.message, "error");
  }
}

function parseJsonField(value) {
  if (!value) {
    return null;
  }
  if (typeof value === "object") {
    return value;
  }
  try {
    return JSON.parse(value);
  } catch (_) {
    try {
      const bytes = Uint8Array.from(atob(value), (char) => char.charCodeAt(0));
      return JSON.parse(new TextDecoder("utf-8").decode(bytes));
    } catch (error) {
      return null;
    }
  }
}

function renderFinalResult(result, sessionId = state.session?.id) {
  state.currentResult = result;
  el.resultBlock.classList.remove("hidden");
  el.transcript.textContent = result?.transcript || "No transcript was produced.";
  el.downloadTextBtn.classList.toggle("hidden", !result);
  if (sessionId && result?.audio_key) {
    el.downloadFullAudioBtn.href = "#";
    el.downloadFullAudioBtn.dataset.sessionId = sessionId;
    el.downloadFullAudioBtn.classList.remove("hidden");
  } else {
    el.downloadFullAudioBtn.dataset.sessionId = "";
    el.downloadFullAudioBtn.classList.add("hidden");
  }

  const segments = result?.segments || [];
  renderSpeakerLabels(result);
  el.segments.innerHTML = segments.length
    ? segments.map(renderSegment).join("")
    : '<p class="muted">No timestamped segments were produced.</p>';
}

function renderSpeakerLabels(result) {
  const speakers = uniqueSpeakers(result);
  if (!speakers.length) {
    el.speakerLabelsBlock.classList.add("hidden");
    el.speakerLabels.innerHTML = "";
    return;
  }

  const labels = result?.speaker_labels || {};
  el.speakerLabelsBlock.classList.remove("hidden");
  el.speakerLabels.innerHTML = speakers.map((speaker) => `
    <label class="speaker-label-row">
      <span>${escapeHtml(speaker)}</span>
      <input type="text" data-speaker="${escapeHtml(speaker)}" value="${escapeHtml(labels[speaker] || "")}" placeholder="${escapeHtml(speaker)}">
    </label>
  `).join("");
}

function uniqueSpeakers(result) {
  const speakers = new Set();
  for (const item of result?.segments || []) {
    if (item.speaker) {
      speakers.add(item.speaker);
    }
  }
  for (const item of result?.speaker_turns || []) {
    if (item.speaker) {
      speakers.add(item.speaker);
    }
  }
  return Array.from(speakers).sort();
}

async function loadHistory() {
  try {
    showMessage(el.historyMessage, "");
    el.sessionList.innerHTML = '<p class="muted">Loading recordings...</p>';
    const sessions = await requestJson("/api/v1/sessions", {
      headers: authHeaders(),
    });

    if (!sessions || sessions.length === 0) {
      el.sessionList.innerHTML = '<p class="muted">No recordings yet.</p>';
      return;
    }

    el.sessionList.innerHTML = sessions.map(renderSessionItem).join("");
    el.sessionList.querySelectorAll("[data-session-id]").forEach((button) => {
      button.addEventListener("click", () => openHistorySession(button.dataset.sessionId));
    });
  } catch (error) {
    el.sessionList.innerHTML = "";
    showMessage(el.historyMessage, error.message, "error");
  }
}

function renderSessionItem(session) {
  const created = formatDate(session.created_at);
  const duration = session.total_duration_sec ? `${Math.round(session.total_duration_sec)} sec` : "-";
  const title = session.final_result?.transcript
    ? truncate(session.final_result.transcript, 88)
    : `Session ${session.id.slice(0, 8)}`;
  return `
    <button class="session-item" data-session-id="${session.id}">
      <div>
        <div class="session-title">${escapeHtml(title)}</div>
        <div class="session-meta">
          ${created} · status: ${escapeHtml(session.status)} · duration: ${duration}<br>
          ASR: ${session.asr ? "on" : "off"} · diarization: ${session.diar ? session.diarization_mode : "off"} · NR: ${session.nr ? "on" : "off"}
        </div>
      </div>
      <span class="status">${escapeHtml(session.status)}</span>
    </button>
  `;
}

async function openHistorySession(sessionId) {
  try {
    const data = await requestJson(`/api/v1/sessions/${sessionId}/result`, {
      headers: authHeaders(),
    });
    state.session = data.session;
    el.historyView.classList.add("hidden");
    el.recorderView.classList.add("hidden");
    el.resultView.classList.remove("hidden");
    el.processingBlock.classList.add("hidden");
    renderFinalResult(parseJsonField(data.session.final_result), data.session.id);
  } catch (error) {
    showMessage(el.historyMessage, error.message, "error");
  }
}

async function downloadFullAudio(event) {
  event.preventDefault();
  const sessionId = el.downloadFullAudioBtn.dataset.sessionId;
  if (!sessionId) {
    return;
  }

  const response = await fetch(`/api/v1/sessions/${sessionId}/download?type=full_audio`, {
    headers: authHeaders(),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    showMessage(el.recordMessage, data?.error || "Failed to download audio.", "error");
    return;
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `session_${sessionId}_full.wav`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function saveSpeakerLabels() {
  if (!state.session || !state.currentResult) {
    return;
  }

  const labels = {};
  el.speakerLabels.querySelectorAll("[data-speaker]").forEach((input) => {
    const value = input.value.trim();
    if (value) {
      labels[input.dataset.speaker] = value;
    }
  });

  try {
    const data = await requestJson(`/api/v1/sessions/${state.session.id}/speaker-labels`, {
      method: "PATCH",
      headers: {
        ...authHeaders(),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ speaker_labels: labels }),
    });
    state.currentResult = parseJsonField(data.final_result);
    renderFinalResult(state.currentResult, state.session.id);
  } catch (error) {
    showMessage(el.recordMessage, error.message, "error");
  }
}

function downloadTextResult() {
  if (!state.session || !state.currentResult) {
    return;
  }

  const text = buildTextResult(state.currentResult);
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `session_${state.session.id}_result.txt`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function buildTextResult(result) {
  const lines = [
    "Transcript",
    "",
    result.transcript || "",
    "",
    "Segments",
    "",
  ];

  for (const segment of result.segments || []) {
    const speaker = displaySpeaker(segment);
    lines.push(`[${formatSeconds(segment.start)} - ${formatSeconds(segment.end)}] ${speaker}: ${segment.text || ""}`);
  }

  if (result.speaker_turns?.length) {
    lines.push("", "Speaker turns", "");
    for (const turn of result.speaker_turns) {
      const speaker = displaySpeaker(turn);
      lines.push(`[${formatSeconds(turn.start)} - ${formatSeconds(turn.end)}] ${speaker}`);
    }
  }

  return `${lines.join("\n")}\n`;
}

function renderSegment(segment) {
  const speaker = displaySpeaker(segment);
  const start = formatSeconds(segment.start);
  const end = formatSeconds(segment.end);
  const text = escapeHtml(segment.text || "");
  return `
    <div class="segment">
      <div class="segment-meta">${speaker}<br>${start} - ${end}</div>
      <div>${text}</div>
    </div>
  `;
}

function displaySpeaker(item) {
  const speaker = item.speaker || "UNKNOWN";
  const labels = state.currentResult?.speaker_labels || {};
  return item.speaker_label || labels[speaker] || speaker;
}

function formatSeconds(value) {
  const seconds = Number(value || 0);
  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60);
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function startTimer() {
  stopTimer();
  state.timerId = window.setInterval(() => {
    const elapsed = Math.floor((Date.now() - state.startedAt) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    el.recordTimer.textContent = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }, 500);
}

function stopTimer() {
  if (state.timerId) {
    window.clearInterval(state.timerId);
    state.timerId = null;
  }
}

function setStatus(status) {
  el.sessionStatus.textContent = status;
}

function resetResult() {
  state.currentResult = null;
  el.resultView.classList.add("hidden");
  el.resultBlock.classList.add("hidden");
  el.processingBlock.classList.add("hidden");
  el.downloadFullAudioBtn.classList.add("hidden");
  el.downloadTextBtn.classList.add("hidden");
  el.speakerLabelsBlock.classList.add("hidden");
  el.transcript.textContent = "";
  el.segments.innerHTML = "";
  el.speakerLabels.innerHTML = "";
}

function newSession() {
  stopPolling();
  state.session = null;
  state.chunkIndex = 0;
  state.stopping = false;
  el.sessionId.textContent = "-";
  el.chunkCount.textContent = "0";
  el.recordTimer.textContent = "00:00";
  setStatus("idle");
  resetResult();
  showRecorder();
}

function formatDate(value) {
  if (!value) {
    return "-";
  }
  return new Date(value).toLocaleString();
}

function truncate(value, maxLength) {
  if (!value || value.length <= maxLength) {
    return value || "";
  }
  return `${value.slice(0, maxLength - 1)}...`;
}

el.loginBtn.addEventListener("click", login);
el.registerBtn.addEventListener("click", register);
el.recordNavBtn.addEventListener("click", showRecorder);
el.historyNavBtn.addEventListener("click", showHistory);
el.logoutBtn.addEventListener("click", logout);
el.startBtn.addEventListener("click", startRecording);
el.stopBtn.addEventListener("click", stopRecording);
el.newSessionBtn.addEventListener("click", newSession);
el.refreshHistoryBtn.addEventListener("click", loadHistory);
el.downloadFullAudioBtn.addEventListener("click", downloadFullAudio);
el.downloadTextBtn.addEventListener("click", downloadTextResult);
el.saveSpeakerLabelsBtn.addEventListener("click", saveSpeakerLabels);

if (state.token) {
  showAuthenticated();
} else {
  showAuth();
}
