const messagesEl = document.querySelector("#messages");
const sourcesEl = document.querySelector("#sources");
const sourceMetaEl = document.querySelector("#sourceMeta");
const statusLineEl = document.querySelector("#statusLine");
const formEl = document.querySelector("#chatForm");
const inputEl = document.querySelector("#questionInput");
const sendButtonEl = document.querySelector("#sendButton");

const starter = "How do I add a YouTube video in OptiSigns?";

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function addMessage(role, text, options = {}) {
  const node = document.createElement("article");
  node.className = `message ${role}${options.error ? " error" : ""}${options.typing ? " typing" : ""}`;
  node.innerHTML = `
    <div class="label">${role === "user" ? "You" : "OptiBot"}</div>
    <div class="bubble">${escapeHtml(text)}</div>
  `;
  messagesEl.appendChild(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return node;
}

function clearSources() {
  sourceMetaEl.textContent = "No citations yet";
  sourcesEl.innerHTML = '<div class="empty-state">Sources will appear here</div>';
}

function renderSources(payload) {
  const sources = Array.isArray(payload.sources) ? payload.sources : [];
  if (!sources.length) {
    clearSources();
    return;
  }

  sourceMetaEl.textContent = `${payload.retrieval || "retrieval"} · ${sources.length} citation${sources.length === 1 ? "" : "s"}`;
  sourcesEl.innerHTML = "";
  sources.slice(0, 8).forEach((source) => {
    const title = source.file_name || source.path || "Source";
    const excerpt = source.source_excerpt || (source.score ? `Score: ${source.score}` : "");
    const item = document.createElement("section");
    item.className = "source-item";
    item.innerHTML = `
      <div class="source-title">${escapeHtml(title)}</div>
      ${excerpt ? `<div class="source-excerpt">${escapeHtml(excerpt)}</div>` : ""}
    `;
    sourcesEl.appendChild(item);
  });
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const status = await response.json();
    const retrieval = status.retrieval === "gemini_file_search" ? "Gemini File Search" : "Local chunks";
    statusLineEl.textContent = `${retrieval} · ${status.model} · ${status.chunk_count} chunks`;
  } catch {
    statusLineEl.textContent = "Status unavailable";
  }
}

async function ask(question) {
  addMessage("user", question);
  const pending = addMessage("assistant", "Thinking...", { typing: true });
  sendButtonEl.disabled = true;
  inputEl.disabled = true;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }

    pending.remove();
    addMessage("assistant", payload.answer || "No answer returned.");
    renderSources(payload);
  } catch (error) {
    pending.remove();
    addMessage("assistant", error.message || "Request failed.", { error: true });
  } finally {
    sendButtonEl.disabled = false;
    inputEl.disabled = false;
    inputEl.focus();
  }
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = inputEl.value.trim();
  if (!question) return;
  inputEl.value = "";
  inputEl.style.height = "auto";
  ask(question);
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 140)}px`;
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

clearSources();
addMessage("assistant", "Hi, I'm OptiBot.");
inputEl.value = starter;
refreshStatus();
