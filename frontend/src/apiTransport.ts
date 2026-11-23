/** A token is attached only after the destination passes the API-origin check. */
export function createApiTransport(base:string,origin:string,getToken:()=>Promise<string|null>,fetcher:typeof fetch=fetch){
  const expected=new URL(base || origin,origin)
  return async (input:string,options:RequestInit={})=>{
    const target=new URL(input,origin)
    if(target.origin!==expected.origin || !target.pathname.startsWith('/api/'))throw new Error('Unexpected API destination')
    const headers=new Headers(options.headers),token=await getToken()
    if(token)headers.set('Authorization','Bearer '+token)
    const signal=options.signal ?? AbortSignal.timeout(30000)
    for(let attempt=0;;attempt++){
      const response=await fetcher(target.href,{...options,headers,signal})
      if((options.method ?? 'GET').toUpperCase()!=='GET' || response.status!==429 || attempt>=3 || !response.headers.has('Retry-After'))return response
      await response.body?.cancel()
      const seconds=Number(response.headers.get('Retry-After'))
      const delay=Number.isFinite(seconds)?Math.max(0,Math.min(3000,seconds*1000)):3000
      await new Promise<void>((resolve,reject)=>{
        function aborted(){clearTimeout(timer);signal.removeEventListener('abort',aborted);reject(signal.reason)}
        const timer=setTimeout(()=>{signal.removeEventListener('abort',aborted);resolve()},delay)
        signal.addEventListener('abort',aborted,{once:true})
        if(signal.aborted)aborted()
      })
    }
  }
}
