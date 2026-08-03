/* Brain MRI Triage — local page logic.
 *
 * No frameworks, no external scripts. Everything talks to the local server
 * only, at same-origin relative paths. There is no code path here that can
 * reach the internet.
 */

"use strict";

const el = (id) => document.getElementById(id);

let currentResult = null;
let status = null;

/* ------------------------------------------------------------------ setup */

async function loadStatus() {
  const response = await fetch("/api/status");
  if (!response.ok) throw new Error("The app did not start correctly.");
  status = await response.json();

  // The version only. The name is in the heading two inches to the left, and
  // printing it twice is the kind of thing that makes a screen look generated.
  el("app-version").textContent = status.app_version;
  el("disclaimer-text").textContent = status.disclaimer;

  // The short form down here, the full one on the result.
  //
  // Both used to be the full text, which meant the same four lines appeared
  // twice on one screen, a few inches apart. Printing a safety notice twice
  // does not make it twice as heeded. It makes the page look automatic, and it
  // teaches the reader that the text is scenery.
  el("footer-disclaimer").textContent = status.disclaimer_short;
  // Two separate statements on purpose. What the tool reads is not the same
  // claim as what has been measured about it, and merging them reads as a
  // validation claim the project cannot currently support.
  el("footer-scope").textContent = "Built to read: " + status.intended_scope;
  el("footer-validation").textContent = status.validation_statement;

  // "Nothing leaves this laptop" is true of the clinic build and false of a
  // hosted one. A privacy promise that is false is worse than none at all, so
  // the line is rewritten rather than left to reassure someone wrongly.
  if (status.served_publicly) {
    const privacy = el("privacy-line");
    if (privacy) {
      privacy.innerHTML =
        "<strong>This scan is sent over the network.</strong> This copy is " +
        "served from a server, not from this device. The image travels to " +
        "that server to be read. Do not upload a real patient's scan unless " +
        "you know who runs it and what they keep.";
      privacy.classList.add("privacy-warning");
    }
  }

  // The banner has to survive being read fifty times a day. A wall of red text
  // that nobody finishes is not a safety feature, it is decoration, and the
  // thing it is trying to say gets lost. So the top line carries the two facts
  // that change what somebody does: this is a development build, and it is not
  // the only thing you decide on. The full reasoning stays available underneath
  // rather than being removed, because a clinic deciding whether to trust this
  // is entitled to the whole of it.
  if (status.state !== "clinical") {
    const banner = el("mode-banner");
    banner.textContent = "";

    const headline = document.createElement("span");
    headline.className = "banner-headline";
    headline.textContent = "Development build";
    banner.appendChild(headline);

    const lead = document.createElement("span");
    lead.className = "banner-detail";
    lead.textContent = "A second opinion, not a decision. A person makes the call.";
    banner.appendChild(lead);

    const messages = (status.warnings || []).map((w) => w.message).join(" ");
    if (messages) {
      const more = document.createElement("details");
      more.className = "banner-more";
      const summary = document.createElement("summary");
      summary.textContent = "What has and has not been tested";
      more.appendChild(summary);
      const body = document.createElement("p");
      body.textContent = messages;
      more.appendChild(body);
      banner.appendChild(more);
    }

    banner.classList.remove("hidden");
    if (status.config && status.config.is_stub) {
      el("config-badge").textContent = "STUB CONFIG";
    }
  }
}

/* --------------------------------------------------------------- analysis */

const PANELS = ["upload-panel", "working", "result-panel", "error-panel", "batch-panel"];

function show(panelId) {
  // One panel at a time. The dropzone in particular goes away while a scan is
  // being read: leaving it there invites a second file on top of the first,
  // and this app reads one at a time.
  PANELS.forEach((id) => el(id).classList.add("hidden"));
  el(panelId).classList.remove("hidden");
  // The download panel is reference material, not a step in reading a scan.
  // It stays out of the way while a result is on screen.
  const dl = el("download-panel");
  if (dl) dl.classList.toggle("hidden", panelId !== "upload-panel");
}

/* --------------------------------------------------------------- downloads */

function humanBytes(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
  if (n >= 1e6) return Math.round(n / 1e6) + " MB";
  if (n >= 1e3) return Math.round(n / 1e3) + " KB";
  return n + " bytes";
}

async function loadDownloads() {
  const table = el("download-table");
  const summary = el("download-summary");
  if (!table) return;

  let data;
  try {
    const response = await fetch("/api/downloads");
    if (!response.ok) throw new Error("status " + response.status);
    data = await response.json();
  } catch (err) {
    summary.textContent = "The download list could not be loaded.";
    return;
  }

  const cfg = data.app_config || {};
  summary.innerHTML =
    "<strong>" + data.files.length + " files, " + data.total_size_human +
    " in total.</strong> This server runs " +
    (cfg.ensemble ? cfg.seeds.length + " averaged " : "a single ") +
    (cfg.backbone || "") + " model" + (cfg.ensemble ? "s" : "") +
    ", referring at p_tumor &ge; " + cfg.tumor_threshold +
    " and sending a scan to a human when " + (cfg.defer_signal || "uncertainty") +
    " &ge; " + Number(cfg.defer_threshold).toFixed(4) + " " + (cfg.entropy_units || "") + ".";

  const rows = ['<tr><th>File</th><th>Size</th><th>What it is</th><th>SHA-256</th></tr>'];
  data.files.forEach((f) => {
    rows.push(
      "<tr>" +
      '<td><a class="download-link" href="/api/downloads/' + encodeURIComponent(f.key) +
      '" download>' + f.filename + "</a></td>" +
      "<td>" + f.size_human + "</td>" +
      "<td>" + f.description + "</td>" +
      '<td class="hash">' + f.sha256 + "</td>" +
      "</tr>"
    );
  });
  table.innerHTML = rows.join("");

  el("download-warning").textContent = data.not_validated || "";

  const sizeSpan = el("setup-size");
  if (sizeSpan) sizeSpan.textContent = data.total_size_human;

  // The ready-built Windows app, when the server has one to give. Shown first
  // because it is the only route that asks nothing of the person running it.
  const pkg = data.windows_package;
  if (pkg) {
    el("package-link").href = "/api/downloads/" + encodeURIComponent(pkg.key);
    el("package-size").textContent = pkg.size_human;
    el("package-block").classList.remove("hidden");
  }

  // The setup program. Shown above the zip because it is the only route that
  // ends with a working app without the person doing anything but wait.
  const setup = data.windows_installer;
  if (setup) {
    el("installer-link").href = "/api/downloads/" + encodeURIComponent(setup.key);
    el("installer-size").textContent = setup.size_human;
    el("installer-block").classList.remove("hidden");
  }
}

async function readOne(file) {
  const form = new FormData();
  form.append("file", file, file.name);

  const response = await fetch("/api/analyze", { method: "POST", body: form });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || "The scan could not be read.");
  }
  return response.json();
}

async function analyze(file) {
  el("working-text").textContent = "Reading the scan. This takes a few seconds.";
  show("working");

  try {
    currentResult = await readOne(file);
    render(currentResult);
    show("result-panel");
  } catch (error) {
    el("error-text").textContent = error.message;
    show("error-panel");
  }
}

/* ----------------------------------------------------------------- batches */

/* Several slices at once, because that is how a study arrives.
 *
 * One at a time, deliberately. The app holds one model in memory and reads a
 * scan in well under a second; firing twenty requests at once would queue them
 * anyway and lose the progress count, which is the only thing making a
 * twenty-slice wait bearable.
 *
 * Worst first in the list. A clinician scanning a list of twenty rows should
 * not have to reach row nineteen to find the one that says refer. */
const CALL_ORDER = { tumor: 0, uncertain: 1, cannot_read: 2, no_tumor: 3 };

let batchResults = [];

async function analyzeMany(files) {
  batchResults = [];
  show("working");

  for (let i = 0; i < files.length; i++) {
    el("working-text").textContent =
      "Reading slice " + (i + 1) + " of " + files.length + ".";
    try {
      const result = await readOne(files[i]);
      batchResults.push({ name: files[i].name, result: result, error: null });
    } catch (error) {
      batchResults.push({ name: files[i].name, result: null, error: error.message });
    }
  }

  renderBatch();
  show("batch-panel");
}

function renderBatch() {
  const sorted = batchResults.slice().sort((a, b) => {
    const ra = a.result ? CALL_ORDER[a.result.call_key] : 2;
    const rb = b.result ? CALL_ORDER[b.result.call_key] : 2;
    return ra - rb;
  });

  const flagged = batchResults.filter(
    (row) => row.result && row.result.call_key === "tumor"
  ).length;
  const unread = batchResults.filter((row) => !row.result).length;

  // The number that decides what happens next goes first, on its own line.
  //
  // It used to open with two clauses about study-level thresholds and reach
  // "3 flagged for referral" in the fourth line, past a colon and a semicolon.
  // The caveat is still here and still says the same thing. It is just no
  // longer standing in front of the only number anybody acts on.
  const note = el("batch-note");
  note.textContent = "";

  const headline = document.createElement("span");
  headline.className = "batch-headline";
  headline.textContent =
    flagged + " of " + batchResults.length + " flagged for referral" +
    (unread ? ", " + unread + " could not be read" : "") + ".";
  note.appendChild(headline);

  const explain = document.createElement("span");
  explain.className = "batch-explain";
  explain.textContent =
    "Read separately and NOT combined into a single answer for the study: " +
    "there is no measured threshold for a whole study. Open any row for the " +
    "full result.";
  note.appendChild(explain);

  const list = el("batch-list");
  list.textContent = "";

  sorted.forEach((row) => {
    const item = document.createElement("li");
    item.className = "batch-item " + (row.result ? row.result.call_key : "cannot_read");

    const button = document.createElement("button");
    button.type = "button";
    button.className = "batch-button";

    const name = document.createElement("span");
    name.className = "batch-name";
    name.textContent = row.name;

    const call = document.createElement("span");
    call.className = "batch-call";
    call.textContent = row.result ? row.result.call : "Could not be read";

    const detail = document.createElement("span");
    detail.className = "batch-detail";
    detail.textContent = row.result ? row.result.confidence : row.error;

    button.appendChild(name);
    button.appendChild(call);
    button.appendChild(detail);

    if (row.result) {
      button.addEventListener("click", () => {
        currentResult = row.result;
        render(row.result);
        show("result-panel");
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
    } else {
      button.disabled = true;
    }

    item.appendChild(button);
    list.appendChild(item);
  });
}

/* ---------------------------------------------------------------- render */

function render(result) {
  const box = el("call-box");
  box.className = "call-box " + result.call_key;
  el("call-text").textContent = result.call;
  el("confidence-text").textContent = result.confidence;

  const probability = el("probability-text");
  if (result.probability_text) {
    probability.textContent = result.probability_text;
    probability.classList.remove("hidden");
  } else {
    probability.textContent = "";
    probability.classList.add("hidden");
  }

  el("next-step-text").textContent = result.next_step;
  el("reason-text").textContent = result.reason;

  // Notes: stub warnings, failed heatmaps, anything the operator must see.
  const notes = el("notes-list");
  notes.textContent = "";
  if (result.notes && result.notes.length) {
    result.notes.forEach((note) => {
      const item = document.createElement("li");
      item.textContent = note;
      notes.appendChild(item);
    });
    notes.classList.remove("hidden");
  } else {
    notes.classList.add("hidden");
  }

  renderImages(result);
  renderTumorType(result);
  renderTechnical(result);

  el("disclaimer-text").textContent = result.disclaimer;
}

function renderImages(result) {
  const block = el("images-block");
  const warning = el("heatmap-warning");

  if (!result.original_png) {
    block.classList.add("hidden");
    return;
  }
  block.classList.remove("hidden");

  // Session D publishes this wording and asks that it be shown verbatim.
  el("heatmap-caveat").textContent = result.heatmap_caveat || "";

  el("img-original").src = result.original_png;
  el("img-overlay-base").src = result.original_png;

  if (result.overlay_png) {
    el("img-overlay").src = result.overlay_png;
    el("img-overlay").classList.remove("hidden");
    el("overlay-slider").disabled = false;
    el("overlay-toggle").disabled = false;
    setOverlayOpacity(el("overlay-slider").value);
  } else {
    el("img-overlay").removeAttribute("src");
    el("img-overlay").classList.add("hidden");
    el("overlay-slider").disabled = true;
    el("overlay-toggle").disabled = true;
  }

  if (result.explainer_is_stub) {
    warning.textContent =
      "This heatmap is a test pattern, not where the model looked. " +
      "The real heatmap is not installed in this build.";
    warning.classList.remove("hidden");
  } else {
    warning.classList.add("hidden");
  }
}

function renderTumorType(result) {
  const block = el("type-block");
  if (result.tumor_type && result.call_key === "tumor") {
    el("tumor-type-text").textContent = result.tumor_type;
    block.classList.remove("hidden");
  } else {
    block.classList.add("hidden");
  }
}

function renderTechnical(result) {
  const rows = [
    ["File name", result.display_name],
    ["Read at (UTC)", result.read_at_utc || "not recorded"],
    ["Image fingerprint", result.image_sha256],
    ["Model", result.model_backbone + ", seed(s) " + (result.model_seeds || []).join(", ")],
    ["Config version", result.config_version],
    ["Repeated readings", result.mc_passes],
    ["Disagreement between readings",
      result.entropy === null || result.entropy === undefined
        ? "not measured"
        : result.entropy.toFixed(3) + " " + (result.entropy_units || "")],
    ["Time taken", Math.round(result.latency_ms) + " ms"],
    ["Image check", result.validator_method],
    ["Came from", result.source_format === "dicom"
      ? "DICOM, converted by this app"
      : "an image file, as supplied"],
  ];

  const table = el("technical-table");
  table.textContent = "";
  rows.forEach(([label, value]) => {
    const tr = document.createElement("tr");
    const tdLabel = document.createElement("td");
    tdLabel.textContent = label;
    const tdValue = document.createElement("td");
    tdValue.textContent = value === null || value === undefined ? "not available" : String(value);
    tr.appendChild(tdLabel);
    tr.appendChild(tdValue);
    table.appendChild(tr);
  });
}

/* --------------------------------------------------------------- overlay */

function setOverlayOpacity(value) {
  el("img-overlay").style.opacity = String(Number(value) / 100);
}

/* ----------------------------------------------------------------- export */

async function saveReport() {
  if (!currentResult) return;
  const payload = Object.assign({}, currentResult, {
    include_filename: el("include-filename").checked,
    // Operator-typed, export-only. Never added to currentResult, so it cannot
    // travel anywhere else by accident.
    case_reference: el("case-reference").value.trim(),
  });

  const response = await fetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    el("error-text").textContent = "The result could not be saved.";
    show("error-panel");
    return;
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "mri-triage-result.html";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/* ------------------------------------------------------------------ wire */

/* One slice goes straight to the result. Several go to the list. */
function handle(fileList) {
  if (!fileList || !fileList.length) return;
  const files = Array.prototype.slice.call(fileList);
  if (files.length === 1) {
    analyze(files[0]);
  } else {
    analyzeMany(files);
  }
}

function reset() {
  currentResult = null;
  batchResults = [];
  el("file-input").value = "";
  // The reference belongs to one patient's sheet. Carrying it to the next scan
  // would file one person's result under another's number.
  el("case-reference").value = "";
  show("upload-panel");
}

document.addEventListener("DOMContentLoaded", () => {
  const dropzone = el("dropzone");
  const fileInput = el("file-input");

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => {
    handle(fileInput.files);
  });

  // The class name has to match the stylesheet exactly. It said "dragging"
  // here and ".dropzone.dragover" there, so dropping a file lit up nothing at
  // all: the one moment the page most needs to say "yes, I have this" was the
  // one moment it stayed silent.
  ["dragenter", "dragover"].forEach((name) =>
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((name) =>
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (event) => {
    handle(event.dataTransfer && event.dataTransfer.files);
  });

  el("overlay-slider").addEventListener("input", (event) =>
    setOverlayOpacity(event.target.value)
  );

  el("overlay-toggle").addEventListener("click", () => {
    const slider = el("overlay-slider");
    const hidden = Number(slider.value) === 0;
    slider.value = hidden ? 100 : 0;
    setOverlayOpacity(slider.value);
    el("overlay-toggle").textContent = hidden ? "Hide heatmap" : "Show heatmap";
  });

  el("save-report").addEventListener("click", saveReport);
  el("new-scan").addEventListener("click", () => {
    // Back to the list if there is one, rather than all the way to the start.
    // Losing twenty readings because somebody wanted the previous row is the
    // kind of thing that makes people stop using a tool.
    if (batchResults.length > 1) {
      show("batch-panel");
    } else {
      reset();
    }
  });
  el("batch-new").addEventListener("click", reset);
  el("error-retry").addEventListener("click", reset);

  loadStatus().catch((error) => {
    el("error-text").textContent = error.message;
    show("error-panel");
  });

  // Separate from loadStatus on purpose. Hashing five 344 MB files takes a few
  // seconds on the first request, and reading a scan must never wait on the
  // download list.
  loadDownloads();
});
