const $ = (id) => document.getElementById(id);
let assets = [];
let activeJob = null;

async function api(url, options={}) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.error || `${res.status} ${res.statusText}`);
  return data;
}

async function refreshHealth(){
  try{
    const h=await api('/api/li/health');
    const ready=h.weaviate_ready && h.collection_ready && h.model.model_exists && h.model.worker_python_exists;
    $('healthPill').textContent=ready?'LI system ready':'Setup required';
    $('healthPill').className=`pill ${ready?'good':'bad'}`;
    $('healthPill').title=JSON.stringify(h,null,2);
  }catch(e){$('healthPill').textContent='Health error';$('healthPill').className='pill bad';}
}

function updateMode(){
  const mode=$('ingestMode').value;
  const images=mode==='images';
  $('pathLabel').firstChild.textContent=images?'Image file or folder path':'Text file or folder path';
  $('uploadLabel').firstChild.textContent=images?'Image files':'Text files';
  $('uploadFiles').accept=images?'image/*,.avif':'.txt,.md,.markdown,.rst,.log,.csv,.tsv,.json,.jsonl,.yaml,.yml,.xml,.html,.htm,.toml,.ini,.cfg,.conf,.sql,.py,.js,.jsx,.ts,.tsx,.java,.c,.cc,.cpp,.h,.hpp,.go,.rs,.sh,.bash,.zsh,text/plain,application/json';
}

function setTab(path){
  $('pathTab').classList.toggle('active',path);$('uploadTab').classList.toggle('active',!path);
  $('pathForm').classList.toggle('active',path);$('uploadForm').classList.toggle('active',!path);
}

async function pollJob(){
  if(!activeJob)return;
  try{
    const j=await api(`/api/li/jobs/${activeJob}`);
    $('percent').textContent=`${Math.round(j.overall_pct||0)}%`;$('bar').style.width=`${j.overall_pct||0}%`;
    $('stage').textContent=j.stage||'';$('detail').textContent=j.detail||'';
    const c=j.counters||{};$('objects').textContent=c.weaviate_objects??'—';$('lateVectors').textContent=(c.late_vectors??'—').toLocaleString?.()??c.late_vectors;
    $('textChunks').textContent=c.text_chunk_done??c.text_chunk_total??'—';$('images').textContent=c.image_done??c.image_total??'—';
    if(j.status==='failed'){$('errorBox').textContent=j.error+'\n\n'+(j.detail||'');$('errorBox').classList.remove('hidden');activeJob=null;}
    else if(j.status==='completed'){$('errorBox').classList.add('hidden');activeJob=null;await refreshAssets();}
    else setTimeout(pollJob,900);
  }catch(e){$('errorBox').textContent=e.message;$('errorBox').classList.remove('hidden');activeJob=null;}
}

$('pathForm').addEventListener('submit',async(e)=>{e.preventDefault();try{
  const mode=$('ingestMode').value; const body=mode==='images'?{image_path:$('sourcePath').value,asset_name:$('assetName').value||null}:{text_path:$('sourcePath').value,asset_name:$('assetName').value||null};
  const r=await api(`/api/li/ingest/${mode}/path`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); activeJob=r.job_id;pollJob();
}catch(err){$('errorBox').textContent=err.message;$('errorBox').classList.remove('hidden');}});

$('uploadForm').addEventListener('submit',async(e)=>{e.preventDefault();try{
  const mode=$('ingestMode').value; const files=[...$('uploadFiles').files]; if(!files.length)throw new Error('Select at least one file');
  const fd=new FormData(); const field=mode==='images'?'images':'texts'; files.forEach(f=>fd.append(field,f));fd.append('asset_name',$('uploadAssetName').value||'');
  const r=await api(`/api/li/ingest/${mode}/upload`,{method:'POST',body:fd});activeJob=r.job_id;pollJob();
}catch(err){$('errorBox').textContent=err.message;$('errorBox').classList.remove('hidden');}});

async function refreshAssets(){
  assets=await api('/api/li/assets'); const s=$('assetSelect'); const current=s.value;s.innerHTML='<option value="">All late-interaction assets</option>';
  assets.forEach(a=>{const o=document.createElement('option');o.value=a.asset_id;o.textContent=`${a.name} · ${a.asset_type}`;s.appendChild(o)}); if([...s.options].some(o=>o.value===current))s.value=current;renderAsset();
}
function renderAsset(){const id=$('assetSelect').value;const a=assets.find(x=>x.asset_id===id);if(!a){$('assetSummary').textContent=`${assets.length} LI assets indexed.`;return;}$('assetSummary').textContent=`${a.name} · ${a.asset_type} · ${a.weaviate_objects} objects · ${Number(a.late_vectors||0).toLocaleString()} late vectors · avg ${Number(a.average_late_vectors||0).toFixed(1)}/object · dense 2048d · late 128d`;}
$('assetSelect').addEventListener('change',renderAsset);$('refreshAssets').onclick=refreshAssets;$('removeAsset').onclick=async()=>{const id=$('assetSelect').value;if(!id)return;await api(`/api/li/assets/${id}`,{method:'DELETE'});await refreshAssets();};

$('queryType').addEventListener('change',()=>{const image=$('queryType').value==='image';$('textQueryWrap').classList.toggle('hidden',image);$('imageQueryWrap').classList.toggle('hidden',!image);});

function renderResults(id,rows,kind){const root=$(id);root.innerHTML='';if(!rows?.length){root.innerHTML='<p class="muted">No results.</p>';return;}rows.forEach((r,i)=>{const el=document.createElement('div');el.className='result';const rank=r.rerank_rank||r.late_rank||r.dense_rank||i+1;let media='';if(r.image_url)media=`<img loading="lazy" src="${r.thumbnail_url||r.image_url}" alt="${r.source_name||'result'}">`;let text=r.text?`<pre>${escapeHtml(r.text)}</pre>`:'';let score='';if(kind==='dense')score=`distance ${fmt(r.dense_distance)}`;if(kind==='late')score=`MaxSim/HNSW distance ${fmt(r.late_distance)} · ${r.late_vector_count||0} doc vectors`;if(kind==='m0')score=`m0 ${fmt(r.rerank_score)} · late rank ${r.late_rank||'—'}`;el.innerHTML=`<div class="result-head"><strong>#${rank} ${escapeHtml(r.source_name||r.chunk_id||'result')}</strong><span class="badge">${r.modality}</span></div>${media}${text}<div class="score">${score}</div>`;root.appendChild(el);});}
function fmt(v){return v===null||v===undefined?'—':Number(v).toFixed(5)}function escapeHtml(s){return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}

$('searchForm').addEventListener('submit',async(e)=>{e.preventDefault();try{
  const type=$('queryType').value;const base={asset_id:$('assetSelect').value||null,modality:$('modality').value,limit:Number($('limit').value),late_candidate_limit:Number($('lateLimit').value),m0_candidate_limit:Number($('m0Limit').value),rerank:$('rerank').checked};let data;
  if(type==='text'){const q=$('textQuery').value.trim();if(!q)throw new Error('Enter a text query');data=await api('/api/li/search/text',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...base,query:q})});}
  else{const file=$('imageQuery').files[0];if(!file)throw new Error('Choose an image query');const fd=new FormData();fd.append('query_image',file);Object.entries(base).forEach(([k,v])=>{if(v!==null)fd.append(k,String(v))});data=await api('/api/li/search/image',{method:'POST',body:fd});}
  renderResults('denseResults',data.dense_results,'dense');renderResults('lateResults',data.late_results,'late');renderResults('m0Results',data.m0_results,'m0');
  const d=data.diagnostics||{};$('diagnostics').textContent=`Dense ${d.dense_ms} ms · Late ${d.late_ms} ms · m0 ${d.m0_ms} ms · total ${d.total_ms} ms · query late vectors ${d.query_late_vectors} · late candidates ${d.late_candidate_count}/${d.late_candidate_limit}`+(d.m0_error?` · m0 error: ${d.m0_error}`:'');
}catch(err){$('diagnostics').textContent=`Search failed: ${err.message}`;}});

$('ingestMode').addEventListener('change',updateMode);$('pathTab').onclick=()=>setTab(true);$('uploadTab').onclick=()=>setTab(false);
updateMode();refreshHealth();refreshAssets();
