export const USAGE_EVENT = 'sematrace:daily-usage'
export const usageOperations = ['analysis','reference','ai'] as const
export type UsageOperation = typeof usageOperations[number]
export type UsageBudget = {used:number;limit:number;remaining:number;global_remaining:number;available:number;enabled:boolean;status:'available'|'disabled'|'user_exhausted'|'global_exhausted';message:string}
export type DailyUsageSnapshot = {day:string;server_time:string;reset_at:string;timezone:string;usage:Record<UsageOperation,UsageBudget>}

export function readUsage(value:unknown):DailyUsageSnapshot {
  const data=value as DailyUsageSnapshot
  if(!data || typeof data.day!=='string' || typeof data.timezone!=='string' || !Number.isFinite(Date.parse(data.server_time)) || !Number.isFinite(Date.parse(data.reset_at)))throw new Error('Usage unavailable')
  for(const operation of usageOperations){
    const row=data.usage?.[operation]
    if(!row || ['used','limit','remaining','global_remaining','available'].some(key=>!Number.isSafeInteger(row[key as keyof UsageBudget]) || Number(row[key as keyof UsageBudget])<0) || typeof row.enabled!=='boolean' || typeof row.message!=='string' || !['available','disabled','user_exhausted','global_exhausted'].includes(row.status))throw new Error('Usage unavailable')
  }
  return data
}

export function resetDelay(data:DailyUsageSnapshot) {
  // Use the server's clock, not the browser's midnight, clock skew or a fixed 24h.
  return Math.max(250,Date.parse(data.reset_at)-Date.parse(data.server_time)+250)
}

export function usesDailyQuota(input:string,method:string,base:string,origin:string) {
  const url=new URL(input,origin),api=new URL(base || origin,origin)
  return url.origin===api.origin && method.toUpperCase()==='POST' && (
    /^\/api\/(analysis-jobs|sources)\/?$/.test(url.pathname) ||
    /^\/api\/analysis\/[^/]+\/(ai-review|cloud-investigate)$/.test(url.pathname)
  )
}

export async function withUsageRefresh<T>(send:()=>Promise<T>,notify:(phase:'start'|'settled')=>void):Promise<T> {
  notify('start')
  try{return await send()}finally{notify('settled')}
}

export function createUsageLoader(fetchUsage:()=>Promise<DailyUsageSnapshot>,onData:(data:DailyUsageSnapshot)=>void,onError:()=>void) {
  let active=true,running=false,queued=false
  return {
    async refresh(){
      if(!active)return
      if(running){queued=true;return}
      running=true
      do{
        queued=false
        try{const data=await fetchUsage();if(active)onData(data)}catch{if(active)onError()}
      }while(active && queued)
      running=false
    },
    stop(){active=false;queued=false},
  }
}
