import test from 'node:test'
import assert from 'node:assert/strict'
import {createUsageLoader,readUsage,resetDelay,usesDailyQuota,withUsageRefresh} from '../src/usage.ts'

function snapshot(used=0,globalRemaining=30){
  const budget={used,limit:10,remaining:Math.max(0,10-used),global_remaining:globalRemaining,available:Math.max(0,Math.min(10-used,globalRemaining)),enabled:true,status:globalRemaining===0?'global_exhausted':'available',message:''}
  return {day:'2026-11-23',server_time:'2026-11-23T23:59:59Z',reset_at:'2026-11-24T00:00:00Z',timezone:'UTC',usage:{analysis:{...budget},reference:{...budget},ai:{...budget,enabled:false,available:0,status:'disabled',message:'Disabled by this deployment.'}}}
}

test('empty and globally exhausted quotas retain personal allowance and show actual available capacity',()=>{
  const empty=readUsage(snapshot())
  assert.equal(empty.usage.analysis.used,0)
  assert.equal(empty.usage.analysis.available,10)
  const exhausted=readUsage(snapshot(2,0))
  assert.equal(exhausted.usage.analysis.remaining,8)
  assert.equal(exhausted.usage.analysis.available,0)
  assert.equal(exhausted.usage.analysis.status,'global_exhausted')
  assert.equal(exhausted.usage.ai.status,'disabled')
})

test('rollover scheduling uses database time rather than the browser clock or fixed 24h',()=>{
  assert.equal(resetDelay(snapshot()),1250)
  const dst=snapshot();dst.server_time='2026-03-08T00:00:00-05:00';dst.reset_at='2026-03-09T00:00:00-04:00'
  assert.equal(resetDelay(dst),23*3600000+250)
  dst.server_time='2026-11-01T00:00:00-04:00';dst.reset_at='2026-11-02T00:00:00-05:00'
  assert.equal(resetDelay(dst),25*3600000+250)
})

test('invalid or unavailable responses are errors, not invented zero counters',()=>{
  for(const data of [{},null,{...snapshot(),reset_at:'invalid'}])assert.throws(()=>readUsage(data),/Usage unavailable/)
  const invalid=snapshot();invalid.usage.analysis.used=-1
  assert.throws(()=>readUsage(invalid),/Usage unavailable/)
})

test('only quota-consuming POSTs on the configured API trigger refreshes',()=>{
  for(const path of ['/api/analysis-jobs','/api/sources?scope=global','/api/analysis/a/ai-review','/api/analysis/a/cloud-investigate'])assert.equal(usesDailyQuota('https://api.example'+path,'POST','https://api.example','https://ui.example'),true)
  for(const [url,method] of [['https://api.example/api/usage','GET'],['https://api.example/api/analysis-jobs/job','GET'],['https://api.example/api/analysis/a/similarity','POST'],['https://other.example/api/sources','POST']])assert.equal(usesDailyQuota(url,method,'https://api.example','https://ui.example'),false)
})

test('accepted, quota-rejected and failed reservations all notify on start and settlement',async()=>{
  for(const status of [202,403,429,503]){
    const phases=[]
    const result=await withUsageRefresh(async()=>new Response('{}',{status}),phase=>phases.push(phase))
    assert.equal(result.status,status);assert.deepEqual(phases,['start','settled'])
  }
  const phases=[]
  await assert.rejects(withUsageRefresh(async()=>{throw new Error('Connection interrupted')},phase=>phases.push(phase)))
  assert.deepEqual(phases,['start','settled'])
})

test('a reservation settling during an older read queues a fresh server snapshot without optimistic increments',async()=>{
  let release,calls=0
  const results=[],loader=createUsageLoader(async()=>{
    calls++
    if(calls===1)return new Promise(resolve=>{release=resolve})
    return snapshot(1)
  },data=>results.push(data),()=>assert.fail('unexpected read failure'))
  const first=loader.refresh()
  await loader.refresh() // A POST settled while the first read was still pending.
  release(snapshot(0));await first
  assert.equal(calls,2)
  assert.equal(results.at(-1).usage.analysis.used,1)
  loader.stop()
})

test('read failure retains last known state and later success recovers',async()=>{
  let fail=false,errors=0
  const results=[],loader=createUsageLoader(async()=>{if(fail)throw new Error('unavailable');return snapshot(2)},data=>results.push(data),()=>errors++)
  await loader.refresh();fail=true;await loader.refresh()
  assert.equal(results.at(-1).usage.analysis.used,2);assert.equal(errors,1)
  fail=false;await loader.refresh();assert.equal(results.length,2)
  loader.stop()
})

test('account unmount discards pending results and suppresses subsequent reads',async()=>{
  let release,calls=0
  const loader=createUsageLoader(()=>new Promise(resolve=>{calls++;release=resolve}),()=>assert.fail('old-account result leaked'),()=>assert.fail('old-account error leaked'))
  const pending=loader.refresh();loader.stop();release(snapshot(5));await pending;await loader.refresh()
  assert.equal(calls,1)
})
