import test from 'node:test'
import assert from 'node:assert/strict'
import {createHash} from 'node:crypto'
import {createClient} from '@supabase/supabase-js'
import {restoreAuthSession,startGoogleSignIn} from '../src/authFlow.ts'

test('installed Supabase SDK exchanges PKCE once, restores persisted Google sessions and signs out',async()=>{
  const previous={window:globalThis.window,document:globalThis.document,BroadcastChannel:globalThis.BroadcastChannel}
  const location={href:'https://ui.example/',assign(value){this.href=value}}
  globalThis.window={location,history:{state:null,replaceState(_state,_title,value){location.href=new URL(value,location.href).href}},addEventListener(){},removeEventListener(){}}
  globalThis.document={visibilityState:'hidden'}
  globalThis.BroadcastChannel=undefined
  const clients=[]
  try{
    const values=new Map(),storage={getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)}
    let exchanges=0,challenge
    const user={id:'e482ca78-9913-4b5c-86d2-73484111aa00',email:'google-fixture@example.invalid',aud:'authenticated',role:'authenticated',app_metadata:{provider:'google'},user_metadata:{name:'Google fixture'}}
    const token=Buffer.from(JSON.stringify({alg:'HS256',typ:'JWT'})).toString('base64url')+'.'+Buffer.from(JSON.stringify({sub:user.id,exp:Math.floor(Date.now()/1000)+3600,role:'authenticated'})).toString('base64url')+'.fixture-signature'
    const fetcher=async(input,options={})=>{
      const url=new URL(String(input))
      if(url.pathname.endsWith('/token')){
        assert.equal(url.searchParams.get('grant_type'),'pkce')
        const body=JSON.parse(options.body)
        assert.equal(body.auth_code,'fixture-code')
        assert.equal(createHash('sha256').update(body.code_verifier).digest('base64url'),challenge)
        exchanges++
        return new Response(JSON.stringify({access_token:token,refresh_token:'fixture-refresh',token_type:'bearer',expires_in:3600,user}),{headers:{'Content-Type':'application/json'}})
      }
      if(url.pathname.endsWith('/user'))return new Response(JSON.stringify(user),{headers:{'Content-Type':'application/json'}})
      if(url.pathname.endsWith('/logout'))return new Response(null,{status:204})
      throw new Error('Unexpected auth request: '+url.pathname)
    }
    function client(){
      const c=createClient('https://fixture-project.supabase.co','fixture-public-key',{auth:{persistSession:true,autoRefreshToken:false,detectSessionInUrl:true,flowType:'pkce',storageKey:'sematrace-google-fixture',storage},global:{fetch:fetcher}})
      clients.push(c);return c
    }
    const first=client()
    assert.equal(await restoreAuthSession(first.auth,location.href),null)
    await startGoogleSignIn(first.auth,'https://ui.example')
    const authorize=new URL(location.href)
    assert.equal(authorize.searchParams.get('provider'),'google')
    assert.equal(authorize.searchParams.get('code_challenge_method'),'s256')
    challenge=authorize.searchParams.get('code_challenge')
    assert.ok(challenge)
    const callback=new URL(authorize.searchParams.get('redirect_to'))
    assert.equal(callback.origin,'https://ui.example')
    callback.searchParams.set('code','fixture-code');location.href=callback.href
    const returning=client(),href=location.href
    // React StrictMode can start two restoration calls; both share initialization.
    const [a,b]=await Promise.all([restoreAuthSession(returning.auth,href),restoreAuthSession(returning.auth,href)])
    assert.equal(a.user.id,user.id);assert.equal(b.user.id,user.id)
    assert.equal(exchanges,1)
    assert.equal(new URL(location.href).searchParams.has('code'),false)
    const reloaded=client()
    assert.equal((await restoreAuthSession(reloaded.auth,location.href)).user.id,user.id)
    assert.equal(exchanges,1)
    assert.equal((await reloaded.auth.signOut()).error,null)
    assert.equal((await reloaded.auth.getSession()).data.session,null)
    assert.equal(await restoreAuthSession(client().auth,location.href),null)
  }finally{
    for(const c of clients)await c.auth.stopAutoRefresh()
    for(const [key,value] of Object.entries(previous)){if(value===undefined)delete globalThis[key];else globalThis[key]=value}
  }
})
