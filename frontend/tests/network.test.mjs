import test from 'node:test'
import assert from 'node:assert/strict'
import {createApiTransport} from '../src/apiTransport.ts'

test('cloud credentials are sent only to configured API routes',async()=>{
  const calls=[]
  const api=createApiTransport('https://api.example.com','https://ui.example.com',async()=>'fixture-token',async(...args)=>{calls.push(args);return new Response('{}')})
  await api('https://api.example.com/api/analyses')
  assert.equal(calls[0][1].headers.get('Authorization'),'Bearer fixture-token')
  await assert.rejects(api('https://other.example.com/api/analyses'),/Unexpected API/)
  await assert.rejects(api('https://api.example.com/unrelated'),/Unexpected API/)
  assert.equal(calls.length,1)
})

test('a rejected destination never obtains a session token',async()=>{
  let requested=0
  const api=createApiTransport('https://api.example.com','https://ui.example.com',async()=>{requested++;return 'token'})
  await assert.rejects(api('https://example.net/api/private'),/Unexpected API/)
  assert.equal(requested,0)
})

test('native local operation needs no cloud session',async()=>{
  let options
  const api=createApiTransport('','http://localhost:8010',async()=>null,async(_url,o)=>{options=o;return new Response('{}')})
  await api('/api/health')
  assert.equal(options.headers.has('Authorization'),false)
})

test('upload bodies, timeouts and content headers are preserved',async()=>{
  const body=new FormData();body.append('file',new Blob(['sample']),'sample.txt')
  const controller=new AbortController();let options
  const api=createApiTransport('','http://localhost:8010',async()=>'own',async(_url,o)=>{options=o;return new Response('{}')})
  await api('/api/analysis-jobs',{method:'POST',body,signal:controller.signal})
  assert.equal(options.body,body);assert.equal(options.signal,controller.signal)
  assert.equal(options.headers.has('Content-Type'),false)
})

test('session failures do not submit an unauthenticated upload',async()=>{
  let called=false
  const api=createApiTransport('','http://localhost:8010',async()=>{throw new Error('Session expired')},async()=>{called=true;return new Response('{}')})
  await assert.rejects(api('/api/analysis-jobs',{method:'POST'}),/Session expired/)
  assert.equal(called,false)
})

test('a busy report read retries safely without repeating uploads',async()=>{
  let calls=0
  const fetcher=async()=>++calls===1?new Response('{}',{status:429,headers:{'Retry-After':'0'}}):new Response('{}')
  const api=createApiTransport('','http://localhost:8010',async()=>null,fetcher)
  assert.equal((await api('/api/analysis/report')).status,200)
  assert.equal(calls,2)
  calls=0
  assert.equal((await api('/api/analysis-jobs',{method:'POST'})).status,429)
  assert.equal(calls,1)
})
