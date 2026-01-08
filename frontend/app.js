
// ======= CONFIG =======
const API_BASE = 'https://ro-classifier2.onrender.com'; // ex: https://ro-classifier-backend.onrender.com

// ======= NAV =======
const drawer = document.getElementById('drawer');
document.getElementById('hamburger').addEventListener('click', () => {
  drawer.classList.toggle('open');
});
function showTab(id){
  document.querySelectorAll('.tab').forEach(s => s.classList.remove('visible'));
  document.getElementById(id).classList.add('visible');
  drawer.classList.remove('open');
}
showTab('home'); // default

// ======= Loader =======
const loader = document.getElementById('loader');
function showLoader(v=true){ loader.style.display = v ? 'flex' : 'none'; }

// ======= Stats =======
async function refreshStats(){
  try{
    const r = await fetch(`${API_BASE}/stats`);
    const d = await r.json();
    document.getElementById('statTraining').textContent = d.training_rows ?? 0;
    document.getElementById('statProcessed').textContent = d.processed_rows ?? 0;
  }catch(e){}
}
refreshStats();

// ======= Rules =======
async function loadRules(){
  const r = await fetch(`${API_BASE}/rules`);
  const rules = await r.json();
  // Grid
  const grid = document.getElementById('rulesGrid');
  grid.innerHTML = '';
  Object.keys(rules).forEach(k=>{
    const it = rules[k];
    const div = document.createElement('div');
    div.className = 'ruleCard';
    const img = document.createElement('img');
    img.src = `${API_BASE}${it.icon}`;
    img.alt = k;
    const title = document.createElement('div');
    title.innerHTML = `<strong>${k}</strong><br>${it.fr} / ${it.en}`;
    div.appendChild(img); div.appendChild(title);
    grid.appendChild(div);
  });
  // List
  const list = document.getElementById('rulesList');
  list.innerHTML = '';
  Object.keys(rules).forEach(k=>{
    const it = rules[k];
    const p = document.createElement('p');
    p.innerHTML = `<strong>${k}:</strong> ${it.fr} / ${it.en}`;
    list.appendChild(p);
  });
}
loadRules();

// ======= Training upload =======
async function uploadTraining(){
  const f = document.getElementById('fTrain').files[0];
  const msg = document.getElementById('msgTrain');
  msg.textContent = '';
  if(!f){ msg.textContent = 'Veuillez sélectionner un fichier Excel.'; return; }
  const form = new FormData(); form.append('file', f);
  showLoader(true);
  try{
    const r = await fetch(`${API_BASE}/upload-training`, {method:'POST', body: form});
    const d = await r.json();
    showLoader(false);
    if(r.ok){ msg.textContent = d.message || 'Importation successfully'; refreshStats(); }
    else{ msg.textContent = d.error || 'Erreur lors de l’import.'; }
  }catch(e){ showLoader(false); msg.textContent = 'Erreur réseau.'; }
}

// ======= Process data =======
async function processData(){
  const f = document.getElementById('fProcess').files[0];
  const msg = document.getElementById('msgProcess');
  const preview = document.getElementById('preview');
  msg.textContent = ''; preview.innerHTML='';
  if(!f){ msg.textContent = 'Veuillez sélectionner un fichier Excel.'; return; }

  const form = new FormData(); form.append('file', f);
  document.getElementById('btnAnalyze').disabled = true;
  showLoader(true);
  try{
    const r = await fetch(`${API_BASE}/process-data`, {method:'POST', body: form});
    const d = await r.json();
    showLoader(false);
    document.getElementById('btnAnalyze').disabled = false;
    if(!r.ok){ msg.textContent = d.error || 'Erreur de traitement.'; return; }

    msg.textContent = `Analyse OK (${d.rows} lignes).`;
    // Preview table
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    thead.innerHTML = '<tr><th>Date</th><th>SACS ID N°</th><th>Description</th><th>RO</th><th>Icône</th></tr>';
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    d.preview.forEach(row=>{
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${row["Date"]||''}</td>
                      <td>${row["SACS ID N°"]||''}</td>
                      <td>${row["Description"]||''}</td>
                      <td>${row["RO_predite"]||''}</td>
                      <td>${row["RO_icon"]? `<img src="${API_BASE}${row["RO_icon"]}" style="width:20px;height:20px;object-fit:contain"/>` : ''}</td>`;
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    preview.appendChild(table);

    // Offer exports
    const exp = document.createElement('div');
    exp.style.marginTop='10px';
    exp.innerHTML = `
      <a class="primary" href="${API_BASE}/export/excel/${d.job_id}" target="_blank">Exporter en Excel</a>
      <a class="primary" href="${API_BASE}/export/ppt/${d.job_id}" target="_blank">Exporter en PPT</a>`;
    preview.appendChild(exp);

    refreshStats();
  }catch(e){ showLoader(false); document.getElementById('btnAnalyze').disabled = false; msg.textContent = 'Erreur réseau.'; }
}

// ======= Files listing / deletion =======
async function loadFiles(kind='processed', btn=null){
  // set active
  document.querySelectorAll('.subtab').forEach(b=>b.classList.remove('active'));
  if(btn) btn.classList.add('active');

  const target = document.getElementById(document.querySelector('.tab.visible').id==='global' ? 'globalTbody' : 'filesTbody');
  target.innerHTML = '';
  const msg = document.getElementById('msgFiles'); if(msg) msg.textContent='';

  try{
    const r = await fetch(`${API_BASE}/files?type=${kind}`);
    const data = await r.json();
    data.forEach(f=>{
      const tr = document.createElement('tr');
      const date = new Date(f.created_at).toLocaleString();
      tr.innerHTML = `<td>${f.id}</td><td>${f.kind}</td><td>${f.filename}</td><td>${f.rows}</td><td>${date}</td>
        <td>
          ${f.kind==='processed' ? `<a class="primary" href="${API_BASE}/export/excel/${f.id}" target="_blank">Excel</a>
                                     <a class="primary" href="${API_BASE}/export/ppt/${f.id}" target="_blank">PPT</a>` : ''}
          <button onclick="deleteFile(${f.id})">Supprimer</button>
        </td>`;
      target.appendChild(tr);
    });
  }catch(e){
    if(msg) msg.textContent = 'Erreur de chargement.';
  }
}

async function deleteFile(id){
  if(!confirm('Supprimer ce fichier et ses lignes ?')) return;
  try{
    const r = await fetch(`${API_BASE}/files/${id}`, {method:'DELETE'});
    if(r.ok){ loadFiles('processed'); loadFiles('training'); refreshStats(); }
  }catch(e){}
}

// ======= Template =======
function downloadTemplate(){
  window.open(`${API_BASE}/template`, '_blank');
}
