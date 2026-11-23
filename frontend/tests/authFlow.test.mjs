import test from 'node:test'
import assert from 'node:assert/strict'
import {authCallback,restoreAuthSession,startGoogleSignIn} from '../src/authFlow.ts'

test('Google uses the existing auth client and a same-origin root return URL without privilege metadata',async()=>{
  let request
  await startGoogleSignIn({signInWithOAuth:async input=>{request=input;return {data:{url:'https://auth.example/authorize'},error:null}}},'https://sematrace.example')
  assert.deepEqual(request,{provider:'google',options:{redirectTo:'https://sematrace.example/'}})
})

test('Google launch failures are safe, actionable and retryable',async()=>{
  for(const response of [{data:{url:null},error:null},{data:{url:null},error:new Error('provider details')}]){
    await assert.rejects(startGoogleSignIn({signInWithOAuth:async()=>response},'https://ui.example'),/Google sign-in could not start/)
  }
  await assert.rejects(startGoogleSignIn({signInWithOAuth:async()=>{throw new Error('network failure')}},'https://ui.example'),/try again/)
})

test('cancelled OAuth removes callback credentials and never reflects provider text',()=>{
  const result=authCallback('https://ui.example/?analysis=a_saved&error=access_denied&error_description=untrusted#provider_token=fixture&refresh_token=fixture')
  assert.equal(result.present,true)
  assert.match(result.message,/cancelled or denied/)
  assert.equal(result.message.includes('untrusted'),false)
  assert.equal(result.cleanPath,'/?analysis=a_saved')
})

test('success and failure callbacks clean codes but preserve application links and anchors',()=>{
  assert.equal(authCallback('https://ui.example/?code=fixture&sb_flow_id=fixture-flow&job=job_saved#evidence').cleanPath,'/?job=job_saved#evidence')
  assert.equal(authCallback('https://ui.example/?analysis=a_saved#evidence').present,false)
  assert.equal(authCallback('https://ui.example/#error=server_error&error_description=private').cleanPath,'/')
})

test('session restoration waits for SDK PKCE initialization rather than performing a second exchange',async()=>{
  const calls=[],session={user:{id:'ordinary-user'}}
  const auth={initialize:async()=>{calls.push('initialize');return {error:null}},getSession:async()=>{calls.push('session');return {data:{session},error:null}}}
  assert.equal(await restoreAuthSession(auth,'https://ui.example/?code=fixture'),session)
  assert.deepEqual(calls,['initialize','session'])
})

test('expired PKCE codes and missing verifiers produce a sign-in error rather than an endless restoring state',async()=>{
  const failed={initialize:async()=>({error:new Error('invalid grant')}),getSession:async()=>{throw new Error('must not be called')}}
  await assert.rejects(restoreAuthSession(failed,'https://ui.example/?code=expired'),/Try again in this browser/)
  const missing={initialize:async()=>({error:null}),getSession:async()=>({data:{session:null},error:null})}
  await assert.rejects(restoreAuthSession(missing,'https://ui.example/?code=unrecognised'),/Google sign-in could not be completed/)
  assert.equal(await restoreAuthSession(missing,'https://ui.example/'),null)
})

test('restoring a stored session does not initiate Google login',async()=>{
  const session={user:{id:'existing-password-user'}}
  assert.equal(await restoreAuthSession({initialize:async()=>({error:null}),getSession:async()=>({data:{session},error:null})},'https://ui.example/'),session)
  await assert.rejects(restoreAuthSession({initialize:async()=>({error:null}),getSession:async()=>({data:{session:null},error:new Error('fetch failed')})},'https://ui.example/'),/could not be restored/)
})
