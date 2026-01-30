// ======= CONFIG =======
// If the frontend is served by Flask (recommended), use relative API:
const API_BASE = "/api";

// ======= NAV =======
const drawer = document.getElementById("drawer");
document.getElementById("hamburger").addEventListener("click", () => {
  drawer.classList.toggle("open");
});

function showTab(id) {
  document.querySelectorAll(".tab").forEach((s) => s.classList.remove("visible"));
  const el = document.getElementById(id);
  if (el) el.classList.add("visible");
  drawer.classList.remove("open");

  // Auto-load tables when opening tabs
  if (id === "analyzed") loadFiles("processed", document.querySelector("#analyzed .subtab.active"));
  if (id === "global") loadFiles("training", document.querySelector("#global .subtab.active"));
}
showTab("home");

// ======= Loader =======
const loader = document.getElementById("loader");
function showLoader(v = true) {
  loader.style.display = v ? "flex" : "none";
}

// ======= Stats =======
async function refreshStats() {
  try {
    const r = await fetch(`${API_BASE}/stats`);
    const d = await r.json();
    document.getElementById("statTraining").textContent = d.training_rows ?? 0;
    document.getElementById("statProcessed").textContent = d.processed_rows ?? 0;
  } catch (e) {
    // silent
  }
}
refreshStats();

// ======= Rules =======
async function loadRules() {
  try {
    const r = await fetch(`${API_BASE}/rules`);
    const rules = await r.json();

    // Grid
    const grid = document.getElementById("rulesGrid");
    grid.innerHTML = "";
    Object.keys(rules).forEach((k) => {
      const it = rules[k];
      const div = document.createElement("div");
      div.className = "ruleCard";

      const img = document.createElement("img");
      img.src = it.icon; // served by same host
      img.alt = k;

      const title = document.createElement("div");
      title.innerHTML = `<strong>${k}</strong><br>${it.fr} / ${it.en}`;

      div.appendChild(img);
      div.appendChild(title);
      grid.appendChild(div);
    });

    // List
    const list = document.getElementById("rulesList");
    list.innerHTML = "";
    Object.keys(rules).forEach((k) => {
      const it = rules[k];
      const p = document.createElement("p");
      p.innerHTML = `<strong>${k}:</strong> ${it.fr} / ${it.en}`;
      list.appendChild(p);
    });
  } catch (e) {
    console.error("Failed to load rules", e);
  }
}
loadRules();

// ======= Download template =======
function downloadTemplate() {
  window.open(`${API_BASE}/template`, "_blank");
}

// ======= Training upload =======
async function uploadTraining() {
  const f = document.getElementById("fTrain").files[0];
  const msg = document.getElementById("msgTrain");
  msg.textContent = "";

  if (!f) {
    msg.textContent = "Veuillez sélectionner un fichier Excel.";
    return;
  }

  const form = new FormData();
  form.append("file", f);

  showLoader(true);
  try {
    const r = await fetch(`${API_BASE}/upload-training`, { method: "POST", body: form });
    const d = await r.json();
    showLoader(false);

    if (r.ok) {
      msg.textContent = d.message || "Importation réussie.";
      refreshStats();
      // refresh file list if on analyzed/global
    } else {
      msg.textContent = d.error || "Erreur lors de l’import.";
    }
  } catch (e) {
    showLoader(false);
    msg.textContent = "Erreur réseau.";
  }
}

// ======= Process data =======
async function processData() {
  const f = document.getElementById("fProcess").files[0];
  const msg = document.getElementById("msgProcess");
  const preview = document.getElementById("preview");

  msg.textContent = "";
  preview.innerHTML = "";

  if (!f) {
    msg.textContent = "Veuillez sélectionner un fichier Excel.";
    return;
  }

  const form = new FormData();
  form.append("file", f);

  const btn = document.getElementById("btnAnalyze");
  btn.disabled = true;
  showLoader(true);

  try {
    const r = await fetch(`${API_BASE}/process-data`, { method: "POST", body: form });
    const d = await r.json();

    showLoader(false);
    btn.disabled = false;

    if (!r.ok) {
      msg.textContent = d.error || "Erreur de traitement.";
      return;
    }

    msg.textContent = `Analyse OK (${d.rows} lignes).`;

    // Preview table
    const table = document.createElement("table");
    table.className = "table";

    const thead = document.createElement("thead");
    thead.innerHTML =
      "<tr><th>Date</th><th>SACS ID N°</th><th>Description</th><th>RO</th><th>Icône</th></tr>";
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    (d.preview || []).forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row["Date"] || ""}</td>
        <td>${row["SACS ID N°"] || ""}</td>
        <td>${row["Description"] || ""}</td>
        <td>${row["RO_predite"] || ""}</td>
        <td>${row["RO_icon"] ? `<img src="${row["RO_icon"]}" style="width:20px;height:20px;object-fit:contain" />` : ""}</td>
      `;
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    preview.appendChild(table);

    // Export links
    const exp = document.createElement("div");
    exp.style.marginTop = "10px";
    exp.innerHTML = `
      <a class="primary" href="${API_BASE}/export/excel/${d.file_id}" target="_blank">Exporter en Excel</a>
    `;
    preview.appendChild(exp);

    refreshStats();
    // refresh analyzed list
  } catch (e) {
    showLoader(false);
    btn.disabled = false;
    msg.textContent = "Erreur réseau.";
  }
}

// ======= Files list (history) =======
async function loadFiles(kind, btnEl) {
  // set active button UI if provided
  if (btnEl && btnEl.parentElement) {
    btnEl.parentElement.querySelectorAll(".subtab").forEach((b) => b.classList.remove("active"));
    btnEl.classList.add("active");
  }

  try {
    const r = await fetch(`${API_BASE}/files?kind=${encodeURIComponent(kind)}`);
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Failed to load files");

    const files = d.files || [];

    // Two places use tables:
    const filesTbody = document.getElementById("filesTbody");
    const globalTbody = document.getElementById("globalTbody");

    // If we are on analyzed tab
    if (filesTbody) {
      filesTbody.innerHTML = "";
      files.forEach((f) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${f.id}</td>
          <td>${f.kind}</td>
          <td>${f.filename}</td>
          <td>${f.rows}</td>
          <td>${(f.created_at || "").replace("T", " ").replace("Z", "")}</td>
          <td>
            ${f.kind === "processed" ? `<a class="link" href="${API_BASE}/export/excel/${f.id}" target="_blank">Export Excel</a>` : ""}
          </td>
        `;
        filesTbody.appendChild(tr);
      });
    }

    // If we are on global tab
    if (globalTbody) {
      globalTbody.innerHTML = "";
      files.forEach((f) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${f.id}</td>
          <td>${f.kind}</td>
          <td>${f.filename}</td>
          <td>${f.rows}</td>
          <td>${(f.created_at || "").replace("T", " ").replace("Z", "")}</td>
          <td>
            ${f.kind === "processed" ? `<a class="link" href="${API_BASE}/export/excel/${f.id}" target="_blank">Export Excel</a>` : ""}
          </td>
        `;
        globalTbody.appendChild(tr);
      });
    }
  } catch (e) {
    console.error(e);
  }
}
