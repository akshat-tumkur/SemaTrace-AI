import { apiFetch, waitForBackend } from './network'
import { supabase } from './auth'
import { useEffect,useState } from 'react'
import { Upload,FileSearch,LoaderCircle } from 'lucide-react'
import { readJsonResponse } from './api'
import SavedReports from './SavedReports'
import SourceLibrary from './SourceLibrary'
import ProductReport from './ProductReport'
import type { Analysis,Job,ProductStatus } from './productTypes'
import type { SimilarityOptions } from './similarity'
const API=(import.meta.env.VITE_API_URL ?? '').replace(/\/$/,'')
async function api(path:string,options:RequestInit={}) {return readJsonResponse(await apiFetch(API+path,{...options,signal:options.signal ?? AbortSignal.timeout(30000)}))}
export default function App(){
  const [view,setView]=useState('analyze'),[analysis,setAnalysis]=useState<Analysis|null>(null),[file,setFile]=useState<File|null>(null),[job,setJob]=useState<Job|null>(null)
  const [error,setError]=useState(''),[busy,setBusy]=useState(false),[health,setHealth]=useState('Connecting…'),[savedId,setSavedId]=useState('')
  const [backendReady,setBackendReady]=useState(false)
  const [administrator,setAdministrator]=useState(false)
  const [status,setStatus]=useState<ProductStatus|null>(null),[cloud,setCloud]=useState(false),[web,setWeb]=useState(false),[corpus,setCorpus]=useState<{source_origins:number;unique_passages:number}|null>(null)
  function show(payload:Analysis){setAnalysis(payload);setSavedId(payload.analysis_id);setView('analyze');const url=new URL(location.href);url.search='';url.searchParams.set('analysis',payload.analysis_id);history.replaceState(null,'',url)}
  async function refreshCorpus(){try{setCorpus(await api('/api/corpus'))}catch{}}
  async function openSaved(id:string){if(!id.trim())return;setBusy(true);setError('');try{show(await api('/api/analysis/'+encodeURIComponent(id.trim())))}catch(e){setError(e instanceof Error?e.message:'Saved report unavailable')}finally{setBusy(false)}}
  useEffect(()=>{
    let active=true
    const controller=new AbortController()
    async function bootstrap(){
      await waitForBackend(controller.signal,()=>{if(active)setHealth('Waking the free backend?')})
      if(!active)return
      const s:ProductStatus=await api('/api/product/status')
      if(s.authentication_required&&!supabase)throw new Error('This backend requires sign-in. Configure VITE_SUPABASE_URL and VITE_SUPABASE_PUBLISHABLE_KEY for this frontend deployment.')
      if(!active)return
      setStatus(s);setCloud(s.cloud_review_available&&!s.authentication_required)
      if(s.authentication_required){const account=await api('/api/account');if(active)setAdministrator(account.is_admin===true)}
      setHealth('Backend connected');setBackendReady(true)
      void refreshCorpus()
      const u=new URL(location.href),id=u.searchParams.get('analysis'),pending=u.searchParams.get('job')
      if(id)await openSaved(id)
      if(pending){
        const next:Job=await api('/api/analysis-jobs/'+encodeURIComponent(pending))
        if(!active)return
        if(next.status==='completed'&&next.analysis_id)show(await api('/api/analysis/'+next.analysis_id))
        else if(next.status==='failed')setError(next.error ?? 'Analysis stopped')
        else setJob(next)
      }
    }
    void bootstrap().catch((e:Error)=>{if(active){setHealth('Backend unavailable');setError(e.message)}})
    return()=>{active=false;controller.abort()}
  },[])
  useEffect(()=>{
    if(!job || job.status!=='running')return
    let stopped=false,timer:ReturnType<typeof setTimeout>;let failures=0
    async function poll(){try{const next:Job=await api('/api/analysis-jobs/'+job!.job_id);if(stopped)return;failures=0;setJob(next);if(next.status==='completed'&&next.analysis_id){show(await api('/api/analysis/'+next.analysis_id));setJob(null);void refreshCorpus()}else if(next.status==='failed'){setError(next.error ?? 'Analysis stopped');setJob(null)}else timer=setTimeout(poll,1200)}catch{if(stopped)return;failures++;if(failures<8)timer=setTimeout(poll,2000);else{setError('Connection interrupted. Refresh this job URL to reconnect; a running job is not submitted again.');setJob(null)}}}
    timer=setTimeout(poll,300);return()=>{stopped=true;clearTimeout(timer)}
  },[job?.job_id,job?.status])
  async function run(){if(!file)return;setBusy(true);setError('');try{const form=new FormData();form.append('file',file);form.append('cloud_review',String(cloud));form.append('web_search',String(web));const next:Job=await api('/api/analysis-jobs',{method:'POST',body:form,signal:AbortSignal.timeout(120000)});setAnalysis(null);setJob(next);const u=new URL(location.href);u.search='';u.searchParams.set('job',next.job_id);history.replaceState(null,'',u)}catch(e){setError(e instanceof Error?e.message:'Upload failed')}finally{setBusy(false)}}
  async function recalculate(options:SimilarityOptions){if(!analysis)return;const report=await api(`/api/analysis/${analysis.analysis_id}/similarity`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(options)});setAnalysis(a=>a?{...a,similarity_report:report}:a)}
  async function review(){if(!analysis)return;setBusy(true);try{await api(`/api/analysis/${analysis.analysis_id}/ai-review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cloud_review:true}),signal:AbortSignal.timeout(420000)});show(await api('/api/analysis/'+analysis.analysis_id))}finally{setBusy(false)}}
  async function investigate(candidate:string){if(!analysis)return;setBusy(true);try{await api(`/api/analysis/${analysis.analysis_id}/cloud-investigate`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cloud_review:true,candidate_id:candidate}),signal:AbortSignal.timeout(240000)});show(await api('/api/analysis/'+analysis.analysis_id))}finally{setBusy(false)}}
  async function download(kind:'pdf'|'json'){if(!analysis)return;setError('');try{const response=await apiFetch(API+`/api/analysis/${analysis.analysis_id}/`+(kind==='pdf'?'report.pdf':'report'),{signal:AbortSignal.timeout(60000)});if(!response.ok)await readJsonResponse(response);const blob=await response.blob();if(!blob.size)throw new Error('The report export was empty. Retry or download the other format.');const href=URL.createObjectURL(blob),a=document.createElement('a');a.href=href;a.download=kind==='pdf'?'sematrace-report.pdf':`${analysis.filename.replace(/\.[^.]+$/,'')}-sematrace-report.json`;a.click();setTimeout(()=>URL.revokeObjectURL(href),1000)}catch(e){setError(e instanceof Error?e.message:'Report download failed')}}
  function newAnalysis(){setAnalysis(null);setError('');setView('analyze');history.replaceState(null,'',location.pathname)}
  const working=busy || job?.status==='running'
  return <main className="shell product-shell"><header className="topbar"><button className="brand brand-button" onClick={()=>setView('analyze')}><span className="brand-mark">ST</span><span>SemaTrace <em>AI</em></span></button><nav className="main-nav" aria-label="Main navigation"><button aria-pressed={view==='analyze'} onClick={newAnalysis} disabled={!!job || busy}>New analysis</button>{status?.authentication_required?<><button aria-pressed={view==='global'} onClick={()=>setView('global')}>Global Source Library</button><button aria-pressed={view==='library'} onClick={()=>setView('library')}>My Private Sources</button></>:<button aria-pressed={view==='library'} onClick={()=>setView('library')}>Source library <span>{corpus?.source_origins ?? '-'}</span></button>}<button aria-pressed={view==='saved'} onClick={()=>setView('saved')}>{status?.authentication_required?'My Reports':'Saved reports'}</button></nav><div className="status-dot">{health}</div></header>
    {error&&<div className="global-error" role="alert">{error}</div>}
    {job&&<section className="job-progress" aria-live="polite"><div className="section-heading"><h2><LoaderCircle className="spin" size={22}/> {job.stage}</h2><span>{job.progress}%</span></div><progress max={100} value={job.progress}/><p>{job.filename} · You can refresh this page to reconnect to the same analysis.</p></section>}
    {view==='global'&&<SourceLibrary key="global" base={API} scope="global" canManage={administrator} enabled={administrator&&(status?.source_uploads_enabled ?? false)} allowRemoval={administrator} onChanged={()=>void refreshCorpus()}/>}
    {view==='library'&&<SourceLibrary key="private" shared={status?.authentication_required===true} base={API} enabled={status?.source_uploads_enabled ?? false} allowRemoval={status?.storage_mode==='cloud'} onChanged={()=>void refreshCorpus()}/>}
    {view==='saved'&&<section className="saved-page"><p className="eyebrow">PERSISTED EVIDENCE</p><h2>Your saved reports</h2><form className="saved-report" onSubmit={e=>{e.preventDefault();void openSaved(savedId)}}><label htmlFor="saved-id">Open analysis ID</label><input id="saved-id" value={savedId} onChange={e=>setSavedId(e.target.value)}/><button disabled={working}>Open report</button></form><SavedReports initiallyOpen cloud={status?.authentication_required===true} base={API} current={analysis?.analysis_id} onOpen={id=>void openSaved(id)} onDelete={id=>{if(analysis?.analysis_id===id)setAnalysis(null)}}/></section>}
    {view==='analyze'&&!analysis&&!job&&<><section className="hero product-hero"><div><p className="eyebrow">TEXTUAL & SEMANTIC SOURCE COMPARISON</p><h1>Find the source.<br/><span>Inspect the evidence.</span></h1><p className="lede">Compare a document with your reference library and optional public-web sources. Review exact wording, modified passages and AI-identified meaning correspondences.</p><div className="library-summary"><strong>{corpus?.source_origins ?? '—'}</strong> indexed sources · <strong>{corpus?.unique_passages?.toLocaleString() ?? '—'}</strong> passages <button onClick={()=>setView(status?.authentication_required?'global':'library')}>Browse or add references →</button></div></div>
      <div className="upload-panel"><div className="panel-heading"><div><p className="eyebrow">START A COMPARISON</p><h2>Upload your document</h2></div><FileSearch size={25}/></div><label className="dropzone"><input type="file" accept=".txt,.pdf,.docx" aria-label="Submission document" disabled={working} onChange={e=>setFile(e.target.files?.[0] ?? null)}/><Upload size={27}/><strong>{file?.name ?? 'Choose TXT, PDF or DOCX'}</strong><span>{file?`${(file.size/1024).toFixed(1)} KB ready`:'Text-based documents · up to10MB'}</span></label>
        <div className="analysis-options"><label><input type="checkbox" checked={cloud} disabled={!status?.cloud_review_available} onChange={e=>setCloud(e.target.checked)}/><span><strong>AI meaning and paraphrase review</strong><small>Selected excerpts go to OpenAI. Up to48 eligible units per review; no plagiarism verdict.</small></span></label><label><input type="checkbox" checked={web} disabled={!status?.web_search_available} onChange={e=>setWeb(e.target.checked)}/><span><strong>Discover additional public-web sources</strong><small>Two representative queries go to Tavily; up to3 public pages are captured. Keep off for confidential submissions.</small></span></label></div>
        <button className="primary-button" disabled={!file || working || !backendReady} onClick={()=>void run()}>{working?'Starting…':'Run analysis'} <span>↗</span></button><p className="upload-limits">Up to50,000 extracted characters and1,000 text units. Scanned PDFs require OCR elsewhere. {status?.storage_mode==='cloud'?'Cloud storage remains active when both optional services are off.':'Local-only mode works with both external options off.'}</p></div></section><section className="workflow-strip"><div><span>01</span><strong>Retrieve</strong><p>Find candidate passages in indexed sources.</p></div><div><span>02</span><strong>Compare</strong><p>Align wording and review specific meanings.</p></div><div><span>03</span><strong>Verify</strong><p>Inspect actual quotes, attribution and differences.</p></div></section></>}
    {view==='analyze'&&analysis&&<ProductReport analysis={analysis} cloudAvailable={status?.cloud_review_available ?? false} onAIReview={review} onInvestigate={investigate} onRefresh={async()=>show(await api('/api/analysis/'+analysis.analysis_id))} onRecalculate={recalculate} onPdf={()=>void download('pdf')} onJson={()=>void download('json')}/>}
    <footer>SEMATRACE V3 <span>Source-linked evidence · human interpretation</span></footer>
  </main>
}
