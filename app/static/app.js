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

  el("app-version").textContent = status.app_name + " " + status.app_version;
  el("disclaimer-text").textContent = status.disclaimer;
  el("footer-disclaimer").textContent = status.disclaimer;
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

  if (status.state !== "clinical") {
    const banner = el("mode-banner");
    const messages = (status.warnings || []).map((w) => w.message).join(" ");
    banner.textContent = "DEVELOPMENT BUILD — NOT FOR CLINICAL USE";
    const detail = document.createElement("span");
    detail.className = "banner-detail";
    detail.textContent = messages;
    banner.appendChild(detail);
    banner.classList.remove("hidden");
    el("config-badge").textContent = "STUB CONFIG";
  }
}

/* --------------------------------------------------------------- analysis */

function show(panelId) {
  ["upload-panel", "working", "result-panel", "error-panel"].forEach((id) => {
    el(id).classList.toggle("hidden", id !== panelId && id !== "upload-panel");
  });
  el("upload-panel").classList.toggle("hidden", panelId !== "upload-panel");
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

async function analyze(file) {
  show("working");
  const form = new FormData();
  form.append("file", file, file.name);

  try {
    const response = await fetch("/api/analyze", { method: "POST", body: form });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || "The scan could not be read.");
    }
    currentResult = await response.json();
    render(currentResult);
    show("result-panel");
  } catch (error) {
    el("error-text").textContent = error.message;
    show("error-panel");
  }
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
    ["DICOM", "Not supported by this version"],
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

function reset() {
  currentResult = null;
  el("file-input").value = "";
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
    if (fileInput.files && fileInput.files[0]) analyze(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach((name) =>
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragging");
    })
  );
  ["dragleave", "drop"].forEach((name) =>
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragging");
    })
  );
  dropzone.addEventListener("drop", (event) => {
    const files = event.dataTransfer && event.dataTransfer.files;
    if (files && files[0]) analyze(files[0]);
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
  el("new-scan").addEventListener("click", reset);
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
