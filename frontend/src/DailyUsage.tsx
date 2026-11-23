import {useEffect,useRef,useState} from 'react'
import {apiFetch} from './network'
import {readJsonResponse} from './api'
import {createUsageLoader,readUsage,resetDelay,USAGE_EVENT,usageOperations} from './usage'
import type {DailyUsageSnapshot} from './usage'

const labels={analysis:'Analyses',reference:'Reference uploads',ai:'AI reviews'}
export default function DailyUsage({base}:{base:string}){
  const [data,setData]=useState<DailyUsageSnapshot|null>(null),[error,setError]=useState(false)
  const refresh=useRef<()=>void>(()=>{})
  useEffect(()=>{
    const controller=new AbortController()
    let pending=0,resetTimer:ReturnType<typeof setTimeout>|undefined
    const loader=createUsageLoader(async()=>readUsage(await readJsonResponse(await apiFetch(base+'/api/usage',{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(15000)])}))),next=>{
      setData(next);setError(false)
      clearTimeout(resetTimer);resetTimer=setTimeout(()=>void loader.refresh(),resetDelay(next))
    },()=>setError(true))
    refresh.current=()=>{void loader.refresh()}
    function changed(event:Event){
      const phase=(event as CustomEvent<string>).detail
      if(phase==='start')pending++
      if(phase==='settled')pending=Math.max(0,pending-1)
      void loader.refresh()
    }
    function visible(){if(document.visibilityState==='visible')void loader.refresh()}
    // Long-running uploads/reviews reserve before they return. Poll while pending.
    const activePoll=setInterval(()=>{if(pending>0 && document.visibilityState==='visible')void loader.refresh()},5000)
    const idlePoll=setInterval(visible,60000)
    window.addEventListener(USAGE_EVENT,changed);window.addEventListener('focus',visible);document.addEventListener('visibilitychange',visible)
    void loader.refresh()
    return()=>{loader.stop();controller.abort();clearTimeout(resetTimer);clearInterval(activePoll);clearInterval(idlePoll);window.removeEventListener(USAGE_EVENT,changed);window.removeEventListener('focus',visible);document.removeEventListener('visibilitychange',visible);refresh.current=()=>{}}
  },[base])
  return <section className="usage-panel" aria-labelledby="daily-usage-title">
    <div className="usage-heading"><h2 id="daily-usage-title">Today's Usage</h2><button type="button" onClick={()=>refresh.current()}>Refresh usage</button></div>
    {error&&<p role="status" className="usage-warning">Usage is temporarily unavailable.{data?' Showing last known counts.':''} Retry shortly; server limits still apply.</p>}
    {!data&&!error&&<p role="status">Loading today's usage…</p>}
    {data&&<><div className="usage-grid">{usageOperations.map(operation=>{
      const row=data.usage[operation]
      return <article key={operation} data-usage={operation}><div className="usage-label"><h3 id={'usage-'+operation}>{labels[operation]}</h3><strong>{row.enabled?`${row.available} available`:'Disabled'}</strong></div>
        <progress aria-labelledby={'usage-'+operation} max={Math.max(1,row.limit)} value={Math.min(row.used,row.limit)} />
        <p>{row.used} / {row.limit} reserved · {row.remaining} personal remaining</p>
        {row.enabled&&<p>{row.global_remaining} shared remaining</p>}
        {row.message&&<p className="usage-warning">{row.message}</p>}
      </article>
    })}</div><p className="usage-note">Resets {new Date(data.reset_at).toLocaleString()} (your local time). Limits follow the {data.timezone} database day. Reserved requests count even when processing fails.</p></>}
  </section>
}
