import { apiFetch } from './network'
import { useEffect, useState } from 'react'
import { readJsonResponse } from './api'
type Entry = {analysis_id: string; filename: string; created_at: string | null; status: string}
export default function SavedReports({base, current, onOpen, onDelete, initiallyOpen=false,cloud=false}: {base: string; current?: string; onOpen: (id: string) => void; onDelete: (id: string) => void; initiallyOpen?: boolean;cloud?:boolean}) {
  const [rows,setRows]=useState<Entry[]>([])
  const [error,setError]=useState('')
  const [confirm,setConfirm]=useState('')
  const [pending,setPending]=useState(false)
  const [more,setMore]=useState(false)
  async function refresh(append=false) {
    try {
      const response=await apiFetch(base+'/api/analyses?limit=20&offset='+(append?rows.length:0),{signal:AbortSignal.timeout(30000)})
      const page:Entry[]=await readJsonResponse(response)
      setRows(old=>append?[...old,...page]:page);setMore(page.length===20);setError('')
    } catch(e) {setError(e instanceof Error?e.message:'Saved reports unavailable.')}
  }
  useEffect(()=>{void refresh()},[base,current])
  async function remove(id: string) {
    setPending(true)
    try {
      const response=await apiFetch(base+'/api/analysis/'+encodeURIComponent(id),{method:'DELETE',signal:AbortSignal.timeout(30000)})
      if(!response.ok)await readJsonResponse(response)
      setConfirm('');onDelete(id);await refresh()
    } catch(e) {setError(e instanceof Error?e.message:'Report deletion failed.')}
    finally {setPending(false)}
  }
  return <details className="report saved-reports" open={initiallyOpen}><summary>{cloud?'My saved reports':'Saved reports on this device'}</summary>
    <p>{cloud?'Submissions and evidence are stored privately in your cloud account.':'Extracted submissions and evidence remain in local storage until deleted.'} Deletion removes the live report and its evidence; existing exports and backups remain separate.</p>
    <button type="button" disabled={pending} onClick={()=>void refresh()}>Refresh reports</button>
    {rows.map(row=><div className="saved-report-row" data-analysis-id={row.analysis_id} key={row.analysis_id}><button type="button" disabled={pending} onClick={()=>onOpen(row.analysis_id)}>{row.filename}</button><span>{row.created_at?new Date(row.created_at).toLocaleString():'Creation time not recorded'}</span>
      {confirm===row.analysis_id?<span>Delete this stored report? <button type="button" disabled={pending} onClick={()=>void remove(row.analysis_id)}>Confirm deletion</button> <button type="button" disabled={pending} onClick={()=>setConfirm('')}>Cancel</button></span>:<button type="button" disabled={pending} onClick={()=>setConfirm(row.analysis_id)}>Delete</button>}
    </div>)}
    {more&&<button type="button" disabled={pending} onClick={()=>void refresh(true)}>Load more reports</button>}
    {!rows.length&&<p>No saved reports.</p>}{error&&<p role="alert">{error}</p>}
  </details>
}
