const $ = (id) => document.getElementById(id);

let currentJob = null;
let currentAsset = null;
let currentAssetInfo = null;
let assetRows = [];
let pollTimer = null;
let lastResults = [];
let chunkMin = 1;
let chunkMax = 120;
let chunkDefault = 10;

function fmtTime(sec=0){
  const s=Math.max(0,Number(sec)||0),h=Math.floor(s/3600),m=Math.floor((s%3600)/60),r=Math.floor(s%60);
  return h?`${h}:${String(m).padStart(2,'0')}:${String(r).padStart(2,'0')}`:`${m}:${String(r).padStart(2,'0')}`;
}
function esc(s=''){return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
function assetIcon(a){
  if(a.asset_type==='images') return '🖼️';
  if(a.asset_type==='texts') return '📄';
  return '🎬';
}
function fmtSeconds(value){
  const n=Number(value);
  if(!Number.isFinite(n)) return '—';
  return `${Number.isInteger(n)?n:n.toFixed(1).replace(/\.0$/,'')}s`;
}
function clampChunkSeconds(value){
  const n=Number(value);
  if(!Number.isFinite(n)) return chunkDefault;
  return Math.max(chunkMin,Math.min(chunkMax,n));
}
function setChunkControl(prefix,value){
  const v=clampChunkSeconds(value);
  const range=$(`${prefix}ChunkRange`),input=$(`${prefix}ChunkSeconds`),out=$(`${prefix}ChunkValue`);
  if(range){range.min=String(chunkMin);range.max=String(chunkMax);range.value=String(v)}
  if(input){input.min=String(chunkMin);input.max=String(chunkMax);input.value=String(v)}
  if(out)out.textContent=fmtSeconds(v);
}
function bindChunkControl(prefix){
  const range=$(`${prefix}ChunkRange`),input=$(`${prefix}ChunkSeconds`),out=$(`${prefix}ChunkValue`);
  if(!range||!input||!out)return;
  const update=(raw,commit=false)=>{
    let v=Number(raw);
    if(!Number.isFinite(v))v=chunkDefault;
    if(commit)v=clampChunkSeconds(v);
    range.value=String(clampChunkSeconds(v));
    if(commit||document.activeElement!==input)input.value=String(clampChunkSeconds(v));
    out.textContent=fmtSeconds(clampChunkSeconds(v));
  };
  range.addEventListener('input',()=>{input.value=range.value;out.textContent=fmtSeconds(range.value)});
  input.addEventListener('input',()=>update(input.value,false));
  input.addEventListener('change',()=>{
    const v=clampChunkSeconds(input.value);
    input.value=String(v);range.value=String(v);out.textContent=fmtSeconds(v);
  });
}
function selectedChunkSeconds(prefix){
  return clampChunkSeconds($(`${prefix}ChunkSeconds`)?.value ?? chunkDefault);
}

async function checkHealth(){
  const p=$('healthPill');
  try{
    const r=await fetch('/api/health'); const h=await r.json();
    chunkMin=Number(h.video_chunk_seconds_min||1);
    chunkMax=Number(h.video_chunk_seconds_max||120);
    chunkDefault=clampChunkSeconds(h.video_chunk_seconds||10);
    setChunkControl('path',chunkDefault);
    setChunkControl('upload',chunkDefault);
    if(h.model_exists&&h.weaviate_ready){p.textContent='MLX + Weaviate ready';p.className='pill ok'}
    else{p.textContent=`Needs attention · ${!h.model_exists?'model ':''}${!h.weaviate_ready?'weaviate':''}`;p.className='pill bad';p.title=h.weaviate_error||h.model_path}
  }catch(e){p.textContent='Backend unavailable';p.className='pill bad'}
}

function selectTab(which){
  const path=which==='path';
  $('pathTab').classList.toggle('active',path);
  $('uploadTab').classList.toggle('active',!path);
  $('pathForm').classList.toggle('active',path);
  $('uploadForm').classList.toggle('active',!path);
}
$('pathTab').onclick=()=>selectTab('path');
$('uploadTab').onclick=()=>selectTab('upload');

function setIngestMode(){
  const mode=$('ingestMode').value;
  document.querySelectorAll('[data-mode]').forEach(el=>el.classList.toggle('hidden',el.dataset.mode!==mode));
}
$('ingestMode').addEventListener('change',setIngestMode);

async function startPath(e){
  e.preventDefault();
  const mode=$('ingestMode').value;
  let endpoint, body;

  if(mode==='video'){
    if(!$('videoPath').value.trim()||!$('transcriptPath').value.trim()) throw new Error('Video path and ASR JSON path are required.');
    const videoChunkSeconds=selectedChunkSeconds('path');
    endpoint='/api/ingest/path';
    body={
      video_path:$('videoPath').value.trim(),
      transcript_path:$('transcriptPath').value.trim(),
      asset_name:$('assetName').value.trim()||null,
      video_chunk_seconds:videoChunkSeconds
    };
    $('videoCounterLabel').textContent=`Video ${fmtSeconds(videoChunkSeconds)}`;
  }else if(mode==='images'){
    if(!$('imagePath').value.trim()) throw new Error('Image file/folder path is required.');
    endpoint='/api/ingest/images/path';
    body={image_path:$('imagePath').value.trim(),asset_name:$('assetName').value.trim()||null};
  }else{
    if(!$('textPath').value.trim()) throw new Error('Text file/folder path is required.');
    endpoint='/api/ingest/texts/path';
    body={text_path:$('textPath').value.trim(),asset_name:$('assetName').value.trim()||null};
  }

  const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok)throw new Error(await r.text());
  watchJob((await r.json()).job_id);
}

async function startUpload(e){
  e.preventDefault();
  const mode=$('ingestMode').value;
  const fd=new FormData();
  let endpoint;

  if(mode==='video'){
    const v=$('videoFile').files[0],t=$('transcriptFile').files[0];
    if(!v||!t) throw new Error('Choose both a video and ASR JSON.');
    const videoChunkSeconds=selectedChunkSeconds('upload');
    endpoint='/api/ingest/upload';
    fd.append('video',v);fd.append('transcript',t);
    fd.append('video_chunk_seconds',String(videoChunkSeconds));
    $('videoCounterLabel').textContent=`Video ${fmtSeconds(videoChunkSeconds)}`;
  }else if(mode==='images'){
    const imgs=[...$('imageFiles').files];
    if(!imgs.length) throw new Error('Choose at least one image.');
    endpoint='/api/ingest/images/upload';
    imgs.forEach(img=>fd.append('images',img));
  }else{
    const texts=[...$('textFiles').files];
    if(!texts.length) throw new Error('Choose at least one text file.');
    endpoint='/api/ingest/texts/upload';
    texts.forEach(file=>fd.append('texts',file));
  }

  fd.append('asset_name',$('uploadAssetName').value.trim());
  const r=await fetch(endpoint,{method:'POST',body:fd});
  if(!r.ok)throw new Error(await r.text());
  watchJob((await r.json()).job_id);
}
$('pathForm').addEventListener('submit',e=>startPath(e).catch(showTopError));
$('uploadForm').addEventListener('submit',e=>startUpload(e).catch(showTopError));

function showTopError(e){$('errorBox').classList.remove('hidden');$('errorBox').textContent=String(e)}
function watchJob(id){
  currentJob=id;
  $('errorBox').classList.add('hidden');
  if(pollTimer)clearInterval(pollTimer);
  pollJob();
  pollTimer=setInterval(pollJob,700);
}
async function pollJob(){
  if(!currentJob)return;
  try{
    const r=await fetch(`/api/jobs/${currentJob}`),j=await r.json();
    renderJob(j);
    if(j.status==='completed'||j.status==='failed'){
      clearInterval(pollTimer);pollTimer=null;
      if(j.status==='completed')await loadAssets(j.asset_id);
    }
  }catch(e){showTopError(e)}
}
function renderJob(j){
  const pct=Number(j.overall_pct||0);
  $('percent').textContent=`${pct.toFixed(0)}%`;
  $('bar').style.width=`${pct}%`;
  $('stage').textContent=j.stage||'';
  $('detail').textContent=j.detail||'';
  const c=j.counters||{};
  $('textCounter').textContent=c.transcript_total!=null?`${c.transcript_done||0}/${c.transcript_total}`:'—';
  if(c.video_chunk_seconds!=null)$('videoCounterLabel').textContent=`Video ${fmtSeconds(c.video_chunk_seconds)}`;
  $('videoCounter').textContent=c.video_total!=null?`${c.video_done||0}/${c.video_total}`:'—';
  $('imageCounter').textContent=c.image_total!=null?`${c.image_done||0}/${c.image_total}`:'—';
  $('textDocCounter').textContent=c.text_chunk_total!=null?`${c.text_chunk_done||0}/${c.text_chunk_total}`:'—';
  $('objectCounter').textContent=c.weaviate_objects!=null?c.weaviate_objects:'—';
  if(j.status==='failed'){
    $('errorBox').classList.remove('hidden');
    $('errorBox').textContent=(j.error||'Failed')+'\n\n'+(j.detail||'');
  }
}

async function loadAssets(preferId=null){
  const r=await fetch('/api/assets');
  if(!r.ok) throw new Error(await r.text());
  assetRows=await r.json();
  const select=$('assetSelect');
  const previous=preferId||currentAsset;
  select.innerHTML='';

  const all=document.createElement('option');
  all.value='';all.textContent=`All indexed assets (${assetRows.length})`;
  select.appendChild(all);

  assetRows.forEach(a=>{
    const opt=document.createElement('option');
    opt.value=a.asset_id;
    let count;
    if(a.asset_type==='images'){
      count=`${a.image_count} images`;
    }else if(a.asset_type==='texts'){
      count=`${a.text_file_count} files · ${a.text_chunks} chunks`;
    }else{
      count=`${a.video_chunks} × ${fmtSeconds(a.video_chunk_seconds||chunkDefault)} video + ${a.transcript_chunks} transcript`;
    }
    opt.textContent=`${assetIcon(a)} ${a.name} · ${count}`;
    select.appendChild(opt);
  });

  let target='';
  if(preferId&&assetRows.some(a=>a.asset_id===preferId)) target=preferId;
  else if(previous&&assetRows.some(a=>a.asset_id===previous)) target=previous;
  else if(assetRows.length) target=assetRows[0].asset_id;

  select.value=target;
  $('searchSection').classList.toggle('disabled',assetRows.length===0);
  await activateAsset(target);
}
$('refreshAssets').addEventListener('click',()=>loadAssets().catch(showTopError));
$('assetSelect').addEventListener('change',()=>activateAsset($('assetSelect').value).catch(showTopError));

function setPreview(kind){
  $('emptyPreview').classList.toggle('hidden',kind!=='empty');
  $('videoPane').classList.toggle('hidden',kind!=='video');
  $('imagePane').classList.toggle('hidden',kind!=='images');
  $('textPane').classList.toggle('hidden',kind!=='texts');
}
async function activateAsset(assetId){
  currentAsset=assetId||null;
  currentAssetInfo=null;
  if($('assetSelect').value!==String(assetId||'')) $('assetSelect').value=assetId||'';
  $('removeAsset').disabled=!currentAsset;

  if(!currentAsset){
    setPreview('empty');
    $('emptyPreview').textContent=assetRows.length?'Searching can span every indexed asset. Select one asset to preview it.':'No indexed assets yet.';
    $('assetLabel').textContent='All assets';
    $('assetSummary').innerHTML=assetRows.length?`<strong>${assetRows.length}</strong> indexed assets available for global retrieval.`:'No indexed assets yet.';
    $('scopeLabel').textContent='All indexed assets';
    return;
  }

  const r=await fetch(`/api/assets/${currentAsset}`);
  if(!r.ok)throw new Error(await r.text());
  const a=await r.json();
  currentAssetInfo=a;
  $('assetLabel').textContent=`${assetIcon(a)} ${a.name}`;
  $('scopeLabel').textContent=a.name;

  if(a.asset_type==='video'){
    setPreview(a.media_available?'video':'empty');
    if(a.media_available){
      $('videoPlayer').src=a.media_url;
      $('videoPlayer').load();
      $('seekLabel').textContent='Select a video/transcript result to jump to its timestamp.';
    }else{
      $('emptyPreview').textContent='The vectors are still searchable, but the original video file is no longer at its saved path.';
    }
    const chunkSeconds=fmtSeconds(a.video_chunk_seconds||chunkDefault);
    $('assetSummary').innerHTML=`<strong>${esc(a.name)}</strong> · video · ${a.video_chunks} visual chunks at ${chunkSeconds}/chunk · ${a.transcript_chunks} transcript chunks · ${a.weaviate_objects} Weaviate objects${a.media_available?'':' · source file unavailable'}`;
  }else if(a.asset_type==='images'){
    setPreview('images');
    renderGallery(a);
    $('assetSummary').innerHTML=`<strong>${esc(a.name)}</strong> · image collection · ${a.image_count} images · ${a.weaviate_objects} Weaviate objects${a.media_available?'':' · some/all source images unavailable'}`;
  }else{
    setPreview('texts');
    renderTextCollection(a);
    $('assetSummary').innerHTML=`<strong>${esc(a.name)}</strong> · text collection · ${a.text_file_count} files · ${a.text_chunks} recursive chunks · ${a.weaviate_objects} Weaviate objects${a.media_available?'':' · source files unavailable'}`;
  }
}

function renderGallery(a){
  const gallery=$('imageGallery');
  gallery.innerHTML='';
  const urls=a.preview_urls||[];
  if(urls.length){
    $('imagePreview').src=`/api/assets/${a.asset_id}/image/0`;
    urls.forEach((url,index)=>{
      const img=document.createElement('img');
      img.src=url;img.loading='lazy';img.alt=`Image ${index+1}`;
      img.addEventListener('click',()=>{$('imagePreview').src=`/api/assets/${a.asset_id}/image/${index}`});
      gallery.appendChild(img);
    });
  }else{
    $('imagePreview').removeAttribute('src');
  }
  $('imageGalleryNote').textContent=a.image_count>urls.length?`Showing the first ${urls.length} thumbnails of ${a.image_count}. Search can retrieve every indexed image.`:`${a.image_count} indexed images`;
}

function renderTextCollection(a){
  const list=$('textFileList');
  list.innerHTML='';
  const files=a.preview_files||[];
  if(files.length){
    files.forEach(name=>{
      const chip=document.createElement('span');
      chip.className='file-chip';
      chip.textContent=name;
      list.appendChild(chip);
    });
  }else{
    list.innerHTML='<span class="muted">No source filenames are available.</span>';
  }
  $('textResultPreview').textContent='Run a search and click a text result to preview the full retrieved chunk here.';
  $('textPreviewNote').textContent=a.text_file_count>files.length
    ?`Showing ${files.length} of ${a.text_file_count} filenames. Search covers all ${a.text_chunks} indexed text chunks.`
    :`${a.text_file_count} files · ${a.text_chunks} indexed text chunks`;
}

$('removeAsset').addEventListener('click',async()=>{
  if(!currentAsset)return;
  const a=currentAssetInfo;
  if(!confirm(`Remove "${a?.name||currentAsset}" from the Weaviate index? Source files will not be deleted.`))return;
  const r=await fetch(`/api/assets/${currentAsset}`,{method:'DELETE'});
  if(!r.ok) return showTopError(await r.text());
  currentAsset=null;
  await loadAssets();
});

function setQueryType(){
  const image=$('queryType').value==='image';
  $('textQueryPane').classList.toggle('hidden',image);
  $('imageQueryPane').classList.toggle('hidden',!image);
}
$('queryType').addEventListener('change',setQueryType);

$('searchForm').addEventListener('submit',async(e)=>{
  e.preventDefault();
  if(!assetRows.length)return;
  $('results').innerHTML='<div class="empty">Searching…</div>';
  const modality=$('modalityFilter').value;
  const limit=Number($('resultLimit').value||12);

  try{
    let r;
    if($('queryType').value==='text'){
      const q=$('searchQuery').value.trim();
      if(!q)throw new Error('Enter a text query.');
      r=await fetch('/api/search',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({query:q,asset_id:currentAsset,modality,limit})
      });
    }else{
      const img=$('queryImage').files[0];
      if(!img)throw new Error('Choose a query image.');
      const fd=new FormData();
      fd.append('query_image',img);
      fd.append('asset_id',currentAsset||'');
      fd.append('modality',modality);
      fd.append('limit',String(limit));
      r=await fetch('/api/search/image',{method:'POST',body:fd});
    }
    if(!r.ok)throw new Error(await r.text());
    const payload=await r.json();
    lastResults=payload.results||[];
    renderResults(lastResults);
  }catch(err){
    $('results').innerHTML=`<div class="empty">${esc(err)}</div>`;
  }
});

function renderResults(rows){
  if(!rows.length){$('results').innerHTML='<div class="empty">No matching indexed items.</div>';return}

  $('results').innerHTML=rows.map((r,i)=>{
    const label=r.modality==='text'?'TEXT':r.modality==='transcript'?'TRANSCRIPT':'';
    const visual=r.thumbnail_url
      ?`<img class="thumb" src="${r.thumbnail_url}" loading="lazy">`
      :`<div class="text-thumb">${label||esc(r.modality||'ITEM').toUpperCase()}</div>`;
    const preview=r.text||(r.modality==='image'?`Image · ${r.source_name||''}`:`Visual video segment`);
    const dist=r.distance==null?'—':Number(r.distance).toFixed(4);
    const interval=(r.modality==='image'||r.modality==='text')
      ?''
      :` · ${fmtTime(r.start_sec)}–${fmtTime(r.end_sec)}`;

    return `<div class="result" data-i="${i}">
      ${visual}
      <div>
        <h4><span class="badge">${esc(r.modality)}</span> ${esc(r.asset_name||r.asset_id)}${interval}</h4>
        <p>${esc(preview)}</p>
        <span class="source">${esc(r.source_name||'')}</span>
      </div>
      <div class="distance">distance<strong>${dist}</strong></div>
    </div>`;
  }).join('');

  document.querySelectorAll('.result').forEach(el=>el.addEventListener('click',()=>openResult(lastResults[Number(el.dataset.i)])));
}

async function openResult(r){
  if(r.asset_id&&r.asset_id!==currentAsset){
    await activateAsset(r.asset_id);
    $('assetSelect').value=r.asset_id;
  }

  if(r.modality==='image'){
    setPreview('images');
    $('imagePreview').src=r.image_url||r.thumbnail_url;
    $('imageGalleryNote').textContent=`Retrieved ${r.source_name||'image'} · distance ${Number(r.distance||0).toFixed(4)}`;
    return;
  }

  if(r.modality==='text'){
    setPreview('texts');
    $('textResultPreview').textContent=r.text||'';
    $('textPreviewNote').textContent=`Retrieved from ${r.source_name||'text file'} · distance ${Number(r.distance||0).toFixed(4)}`;
    return;
  }

  if(r.media_url&&currentAssetInfo?.asset_type==='video'){
    setPreview('video');
    const start=Number(r.start_sec||0);
    $('videoPlayer').currentTime=start;
    $('videoPlayer').play().catch(()=>{});
    $('seekLabel').textContent=`Jumped to ${fmtTime(start)} · result interval ends ${fmtTime(Number(r.end_sec||start))}`;
  }
}

bindChunkControl('path');
bindChunkControl('upload');
setIngestMode();
setQueryType();
checkHealth();
loadAssets().catch(showTopError);
