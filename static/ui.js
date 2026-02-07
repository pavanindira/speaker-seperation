function $(id) {
  return document.getElementById(id);
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function setPill(status) {
  const el = $("statusPill");
  if (!el) return;
  el.textContent = status || "unknown";
  el.classList.remove("ok", "bad");
  if (status === "completed" || status === "separated" || status === "diagnosed") {
    el.classList.add("ok");
  }
  if (status === "failed") {
    el.classList.add("bad");
  }
}

function setHidden(id, hidden) {
  const el = $(id);
  if (!el) return;
  el.classList.toggle("hidden", !!hidden);
}

function setAlert(message) {
  const el = $("alert");
  if (!el) return;
  if (!message) {
    el.textContent = "";
    el.classList.add("hidden");
    return;
  }
  el.textContent = message;
  el.classList.remove("hidden");
}

function setStepActive(stepNum) {
  for (let i = 1; i <= 4; i++) {
    const el = $(`step${i}`);
    if (!el) continue;
    el.classList.toggle("active", i === stepNum);
    el.classList.toggle("done", i < stepNum);
  }
}

function setWizardTitle(title, subtitle) {
  const t = $("wizardTitle");
  const s = $("wizardSubtitle");
  if (t) t.textContent = title || "Step";
  if (s) s.textContent = subtitle || "";
}

function setProgress(progress, message) {
  const fill = $("progressFill");
  const text = $("progressText");
  const bar = document.querySelector(".progressBar");
  const p = Number.isFinite(progress) ? Math.max(0, Math.min(100, progress)) : 0;
  if (fill) fill.style.width = `${p}%`;
  if (bar) bar.setAttribute("aria-valuenow", String(p));
  if (text) text.textContent = message || "";
}

function renderDiagnosis(report) {
  if (!report) return;
  const card = $("diagnosisCard");
  if (card) card.classList.remove("hidden");

  if ($("diagSpeakers")) $("diagSpeakers").textContent = report.estimated_speakers ?? "—";
  if ($("diagQuality")) $("diagQuality").textContent = report.audio_quality ?? "—";
  if ($("diagDuration")) $("diagDuration").textContent = report.duration ? `${report.duration.toFixed(1)}s` : "—";
  if ($("diagSr")) $("diagSr").textContent = report.sample_rate ? `${report.sample_rate} Hz` : "—";

  const issues = $("diagIssues");
  const recs = $("diagRecs");
  if (issues) issues.innerHTML = (report.issues || []).length
    ? (report.issues || []).map((x) => `<li>${x}</li>`).join("")
    : "<li>None detected</li>";
  if (recs) recs.innerHTML = (report.recommendations || []).length
    ? (report.recommendations || []).map((x) => `<li>${x}</li>`).join("")
    : "<li>No recommendations</li>";
}

function showLoader(title, subtitle) {
  const overlay = $("loaderOverlay");
  if (!overlay) return;
  const t = $("loaderTitle");
  const s = $("loaderSubtitle");
  if (t) t.textContent = title || "Working…";
  if (s) s.textContent = subtitle || "Please wait.";
  overlay.classList.remove("hidden");
}

function hideLoader() {
  const overlay = $("loaderOverlay");
  if (!overlay) return;
  overlay.classList.add("hidden");
}

function renderDownloads(job) {
  const container = $("downloads");
  if (!container) return;

  const blocks = [];
  const addBlock = (title, files, fileType) => {
    if (!files) return;
    const links = Object.entries(files).map(([key, filename]) => {
      const inlineHref = `/api/v1/download/${job.job_id}/${fileType}/${encodeURIComponent(filename)}?inline=1`;
      const dlHref = `/api/v1/download/${job.job_id}/${fileType}/${encodeURIComponent(filename)}`;
      return `
        <div class="fileRow">
          <div class="fileMeta">
            <div class="fileName"><span class="mono">${key}</span></div>
            <div class="fileSubtle mono">${filename}</div>
          </div>
          <audio class="fileAudio" controls preload="metadata" src="${inlineHref}"></audio>
          <a class="fileDownload" href="${dlHref}">Download</a>
        </div>
      `;
    });
    if (links.length) {
      blocks.push(`<div class="resultBlock"><div class="resultTitle">${title}</div>${links.join("")}</div>`);
    }
  };

  addBlock("Separated", job.separated_files, "separated");
  addBlock("Cleaned", job.cleaned_files, "cleaned");

  container.innerHTML = blocks.length ? blocks.join("") : "No files yet.";
}

function updateWizard(job) {
  // Hide all sections
  setHidden("sectionDiagnose", true);
  setHidden("sectionSeparate", true);
  setHidden("sectionProcessing", true);
  setHidden("sectionClean", true);
  setHidden("sectionDownload", true);
  setHidden("sectionFailed", true);

  // Details toggle button visibility
  if ($("btnToggleDetails")) $("btnToggleDetails").disabled = false;
  if ($("btnToggleDetails2")) $("btnToggleDetails2").disabled = false;

  // Defaults from diagnosis
  const report = job.diagnostic_report;
  if (report && $("numSpeakers") && !window.__speakersAutoSet) {
    $("numSpeakers").value = report.estimated_speakers || 2;
    window.__speakersAutoSet = true;
  }
  if (report && $("speakerHint")) {
    $("speakerHint").textContent = report.estimated_speakers
      ? `Suggested: ${report.estimated_speakers}`
      : "";
  }

  renderDiagnosis(report);

  // Original audio preview (inline)
  if ($("audioPreview") && $("audioPreviewWrap") && !window.__audioPreviewSet) {
    $("audioPreview").src = `/api/v1/jobs/${job.job_id}/original`;
    $("audioPreviewWrap").classList.remove("hidden");
    window.__audioPreviewSet = true;
  }

  // Minimal separation summary
  const chosenSpeakers = parseInt($("numSpeakers")?.value || "2", 10);
  const chosenMethod = $("method")?.value || "gmm";
  const suggested = report?.estimated_speakers;
  if ($("separationSummary")) {
    $("separationSummary").textContent = suggested
      ? `Suggested: ${suggested} speaker(s) • Using: ${chosenSpeakers} • Method: ${chosenMethod}`
      : `Using ${chosenSpeakers} speaker(s) • Method: ${chosenMethod}`;
  }

  // Decide which single next step to show
  const status = job.status;
  setProgress(job.progress ?? 0, job.progress_message || status || "");
  if (status === "failed") {
    hideLoader();
    setStepActive(1);
    setWizardTitle("Failed", "Something went wrong. See error details.");
    setHidden("sectionFailed", false);
    setAlert(job.error || "Unknown error");
    return;
  }

  setAlert("");

  if (status === "uploaded") {
    hideLoader();
    setStepActive(1);
    setWizardTitle("Step 1: Diagnose", "Analyze the audio before separating.");
    setHidden("sectionDiagnose", false);
    // Auto-run diagnose once to reduce clicks (simple guided flow)
    if (!window.__autoDiagnose) {
      window.__autoDiagnose = true;
      window.setTimeout(() => runDiagnose(), 400);
    }
    return;
  }

  if (status === "diagnosed" || status === "awaiting_confirmation") {
    hideLoader();
    setStepActive(2);
    setWizardTitle("Step 2: Separate", "Confirm settings and start separation.");
    setHidden("sectionSeparate", false);
    return;
  }

  if (status === "separating") {
    showLoader("Separating…", "Background processing is running. This can take a bit.");
    setStepActive(2);
    setWizardTitle("Separating…", "This can take a bit. The page will update automatically.");
    setHidden("sectionProcessing", false);
    return;
  }

  if (status === "separated") {
    hideLoader();
    setStepActive(3);
    setWizardTitle("Step 3: Clean (optional)", "You can clean the separated files, or skip to downloads.");
    setHidden("sectionClean", false);
    return;
  }

  if (status === "cleaning") {
    showLoader("Cleaning…", "Finishing touches in progress.");
    setStepActive(3);
    setWizardTitle("Cleaning…", "Almost there. The page will update automatically.");
    setHidden("sectionProcessing", false);
    return;
  }

  if (status === "completed") {
    hideLoader();
    setStepActive(4);
    setWizardTitle("Step 4: Download", "Your output files are ready.");
    setHidden("sectionDownload", false);
    return;
  }

  // Fallback
  hideLoader();
  setStepActive(1);
  setWizardTitle("Job", "Waiting for updates…");
  setHidden("sectionProcessing", false);
}

async function fetchJob() {
  const res = await fetch(`/api/v1/jobs/${jobId}`);
  if (!res.ok) throw new Error(`Failed to fetch job: ${res.status}`);
  return await res.json();
}

async function poll() {
  try {
    const job = await fetchJob();
    setPill(job.status);
    if ($("jobJson")) $("jobJson").textContent = pretty(job);
    renderDownloads(job);
    updateWizard(job);
  } catch (e) {
    if ($("jobJson")) $("jobJson").textContent = `Error: ${e.message}`;
    setAlert(e.message);
  } finally {
    window.setTimeout(poll, 1500);
  }
}

async function runDiagnose() {
  $("btnDiagnose").disabled = true;
  showLoader("Diagnosing…", "Analyzing audio and estimating speakers.");
  try {
    const res = await fetch(`/api/v1/jobs/${jobId}/diagnose`);
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `Diagnosis failed: ${res.status}`);
    }
    const job = await res.json();
    updateWizard(job);
    setAlert("");
    hideLoader();
  } finally {
    $("btnDiagnose").disabled = false;
  }
}

async function runProceed() {
  $("btnProceed").disabled = true;
  showLoader("Starting separation…", "Uploading settings and starting background process.");
  try {
    const numSpeakers = parseInt($("numSpeakers").value || "2", 10);
    const method = $("method").value || "gmm";

    const res = await fetch(`/api/v1/jobs/${jobId}/proceed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "proceed", num_speakers: numSpeakers, method }),
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `Proceed failed: ${res.status}`);
    }
    const job = await res.json();
    updateWizard(job);
    setAlert("");
  } finally {
    $("btnProceed").disabled = false;
  }
}

async function runClean() {
  $("btnClean").disabled = true;
  showLoader("Starting cleaning…", "Applying silence/noise/click removal and normalization.");
  try {
    const res = await fetch(`/api/v1/jobs/${jobId}/clean`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        remove_silence: true,
        reduce_noise: true,
        remove_clicks: true,
        normalize: true,
      }),
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `Clean failed: ${res.status}`);
    }
    const job = await res.json();
    updateWizard(job);
    setAlert("");
  } finally {
    $("btnClean").disabled = false;
  }
}

function toggleDetails() {
  // Legacy (detailsPanel was removed in the minimal UI)
  return;
}

function wire() {
  if ($("btnDiagnose")) $("btnDiagnose").addEventListener("click", runDiagnose);
  if ($("btnProceed")) $("btnProceed").addEventListener("click", runProceed);
  if ($("btnClean")) $("btnClean").addEventListener("click", runClean);
  if ($("btnRefresh")) $("btnRefresh").addEventListener("click", () => poll());
  if ($("btnBackToDiagnose")) $("btnBackToDiagnose").addEventListener("click", () => {
    // Force wizard back to diagnose view (without changing backend status)
    setWizardTitle("Step 1: Diagnose", "Analyze the audio before separating.");
    setStepActive(1);
    setHidden("sectionSeparate", true);
    setHidden("sectionDiagnose", false);
  });
  if ($("btnSkipClean")) $("btnSkipClean").addEventListener("click", () => {
    setWizardTitle("Step 4: Download", "Your output files are ready.");
    setStepActive(4);
    setHidden("sectionClean", true);
    setHidden("sectionDownload", false);
  });
}

wire();
poll();

