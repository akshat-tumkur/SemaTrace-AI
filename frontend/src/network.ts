import { supabase } from './auth'
import { createApiTransport } from './apiTransport'

const base=(import.meta.env?.VITE_API_URL ?? '').replace(/\/$/,'')

/** Send session credentials only to this deployment's configured API origin. */
export async function apiFetch(input:string,options:RequestInit={}) {
  return createApiTransport(base,location.origin,async()=>{
    if(!supabase)return null
    const {data,error}=await supabase.auth.getSession()
    if(error)throw new Error('Your sign-in session could not be refreshed. Sign in again.')
    return data.session?.access_token ?? null
  })(input,options)
}

/** A cold start may return HTML or time out. Retry health reads, never uploads. */
export async function waitForBackend(signal:AbortSignal,onWaiting:()=>void){
  const deadline=Date.now()+180000
  let attempts=0
  while(!signal.aborted && Date.now()<deadline){
    try{
      const response=await apiFetch(base+'/api/health',{signal:AbortSignal.any([signal,AbortSignal.timeout(12000)])})
      if(response.ok && (await response.json()).status==='ok')return
    }catch{if(signal.aborted)return}
    attempts++;onWaiting()
    await new Promise<void>(resolve=>{
      function finish(){clearTimeout(timer);signal.removeEventListener('abort',finish);resolve()}
      const timer=setTimeout(finish,Math.min(1000*attempts,5000))
      signal.addEventListener('abort',finish,{once:true})
    })
  }
  if(!signal.aborted)throw new Error('The backend is still unavailable. Free hosting may need more time to wake; retry shortly.')
}
