const $ = (id) => document.getElementById(id);
let currentJob = null;
let currentAsset = null;
let pollTimer = null;

function fmtTime(sec=0){const s=Math.max(0,Number(sec)||0);const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),r=Math.floor(s%60);return h?`${h}:${String(m).padStart(2,'0')}:${String(r).padStart(2,'0')}`:`${m}:${String(r).padStart(2,'0')}`}
function esc(s=''){return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}

async function checkHealth(){
  const p=$('healthPill');
  try{const r=await fetch('/api/health');const h=await r.json();
    if(h.model_exists&&h.weaviate_ready){p.textContent='MLX + Weaviate ready';p.className='pill ok'}
    else{p.textContent=`Needs attention · ${!h.model_exists?'model ':''}${!h.weaviate_ready?'weaviate':''}`;p.className='pill bad';p.title=h.weaviate_error||h.model_path}
  }catch(e){p.textContent='Backend unavailable';p.className='pill bad'}
}

function selectTab(which){
  const path=which==='path';$('pathTab').classList.toggle('active',path);$('uploadTab').classList.toggle('active',!path);$('pathForm').classList.toggle('active',path);$('uploadForm').classList.toggle('active',!path)
}
$('pathTab').onclick=()=>selectTab('path');$('uploadTab').onclick=()=>selectTab('upload');

async function startPath(e){e.preventDefault();const body={video_path:$('videoPath').value.trim(),transcript_path:$('transcriptPath').value.trim(),asset_name:$('assetName').value.trim()||null};const r=await fetch('/api/ingest/path',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!r.ok)throw new Error(await r.text());watchJob((await r.json()).job_id)}
async function startUpload(e){e.preventDefault();const fd=new FormData();fd.append('video',$('videoFile').files[0]);fd.append('transcript',$('transcriptFile').files[0]);fd.append('asset_name',$('uploadAssetName').value.trim());const r=await fetch('/api/ingest/upload',{method:'POST',body:fd});if(!r.ok)throw new Error(await r.text());watchJob((await r.json()).job_id)}
$('pathForm').addEventListener('submit',e=>startPath(e).catch(showTopError));$('uploadForm').addEventListener('submit',e=>startUpload(e).catch(showTopError));

function showTopError(e){$('errorBox').classList.remove('hidden');$('errorBox').textContent=String(e)}
function watchJob(id){currentJob=id;$('errorBox').classList.add('hidden');$('searchSection').classList.add('disabled');if(pollTimer)clearInterval(pollTimer);pollJob();pollTimer=setInterval(pollJob,700)}
async function pollJob(){if(!currentJob)return;try{const r=await fetch(`/api/jobs/${currentJob}`);const j=await r.json();renderJob(j);if(j.status==='completed'||j.status==='failed'){clearInterval(pollTimer);pollTimer=null;if(j.status==='completed')activateAsset(j.asset_id)}}catch(e){showTopError(e)}}
function renderJob(j){const pct=Number(j.overall_pct||0);$('percent').textContent=`${pct.toFixed(0)}%`;$('bar').style.width=`${pct}%`;$('stage').textContent=j.stage||'';$('detail').textContent=j.detail||'';const c=j.counters||{};$('textCounter').textContent=c.transcript_total!=null?`${c.transcript_done||0}/${c.transcript_total}`:'—';$('videoCounter').textContent=c.video_total!=null?`${c.video_done||0}/${c.video_total}`:'—';$('objectCounter').textContent=c.weaviate_objects!=null?c.weaviate_objects:'—';if(j.status==='failed'){$('errorBox').classList.remove('hidden');$('errorBox').textContent=j.error+'\n\n'+(j.detail||'')}}
async function activateAsset(assetId){currentAsset=assetId;$('searchSection').classList.remove('disabled');$('videoPlayer').src=`/api/assets/${assetId}/media`;$('assetLabel').textContent=assetId;$('results').innerHTML='<div class="empty">Ready. Search across both transcript and 10-second video vectors.</div>';}

$('searchForm').addEventListener('submit',async(e)=>{e.preventDefault();if(!currentAsset)return;const q=$('searchQuery').value.trim();if(!q)return;$('results').innerHTML='<div class="empty">Searching…</div>';try{const r=await fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q,asset_id:currentAsset,modality:$('modalityFilter').value,limit:12})});if(!r.ok)throw new Error(await r.text());renderResults((await r.json()).results)}catch(err){$('results').innerHTML=`<div class="empty">${esc(err)}</div>`}});

function renderResults(rows){if(!rows.length){$('results').innerHTML='<div class="empty">No matching chunks.</div>';return}$('results').innerHTML=rows.map((r,i)=>{const visual=r.thumbnail_url?`<img class="thumb" src="${r.thumbnail_url}" loading="lazy">`:`<div class="text-thumb">TRANSCRIPT</div>`;const preview=r.text||'Visual 10-second video segment';const dist=r.distance==null?'—':Number(r.distance).toFixed(4);return `<div class="result" data-start="${r.start_sec}" data-end="${r.end_sec}" data-i="${i}">${visual}<div><h4><span class="badge">${esc(r.modality)}</span> &nbsp; ${fmtTime(r.start_sec)}–${fmtTime(r.end_sec)}</h4><p>${esc(preview)}</p></div><div class="distance">distance<strong>${dist}</strong></div></div>`}).join('');document.querySelectorAll('.result').forEach(el=>el.addEventListener('click',()=>{const start=Number(el.dataset.start);$('videoPlayer').currentTime=start;$('videoPlayer').play().catch(()=>{});$('seekLabel').textContent=`Jumped to ${fmtTime(start)} · result interval ends ${fmtTime(Number(el.dataset.end))}`}))}

checkHealth();
