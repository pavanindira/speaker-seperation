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
      const href = `/api/v1/download/${job.job_id}/${fileType}/${encodeURIComponent(filename)}`;
      return `<div><span class="mono">${key}</span>: <a href="${href}">${filename}</a></div>`;
    });
    if (links.length) {
      blocks.push(`<div style="margin-bottom:10px;"><strong>${title}</strong>${links.join("")}</div>`);
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

  // Decide which single next step to show
  const status = job.status;
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
  const panel = $("detailsPanel");
  if (!panel) return;
  const nowHidden = panel.classList.toggle("hidden");
  const b1 = $("btnToggleDetails");
  const b2 = $("btnToggleDetails2");
  if (b1) b1.textContent = nowHidden ? "Show technical details" : "Hide technical details";
  if (b2) b2.textContent = nowHidden ? "Show technical details" : "Hide technical details";
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
  if ($("btnToggleDetails")) $("btnToggleDetails").addEventListener("click", toggleDetails);
  if ($("btnToggleDetails2")) $("btnToggleDetails2").addEventListener("click", toggleDetails);
}

wire();
poll();

