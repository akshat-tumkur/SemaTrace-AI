import { useEffect,useRef,useState } from 'react'
import type { Session } from '@supabase/supabase-js'
import { supabase } from './auth'
import { authCallback,restoreAuthSession,startGoogleSignIn } from './authFlow'
import App from './App'

export default function AuthGate(){
  const [session,setSession]=useState<Session|null>(null),[loaded,setLoaded]=useState(!supabase)
  const [email,setEmail]=useState(''),[password,setPassword]=useState(''),[error,setError]=useState('')
  const [busy,setBusy]=useState<'password'|'google'|'signout'|null>(null)
  const pending=useRef(false)
  useEffect(()=>{
    if(!supabase)return
    let active=true
    const href=location.href,callback=authCallback(href)
    const {data}=supabase.auth.onAuthStateChange((_event,next)=>{if(active)setSession(next)})
    restoreAuthSession(supabase.auth,href).then(next=>{if(active)setSession(next)}).catch(()=>{
      if(active)setError(callback.present?callback.message:'Your sign-in session could not be restored. Please sign in again.')
    }).finally(()=>{
      if(active){
        if(callback.present)history.replaceState(history.state,'',authCallback(location.href).cleanPath)
        setLoaded(true)
      }
    })
    function returned(){setBusy(current=>{if(current==='google'){pending.current=false;return null}return current})}
    window.addEventListener('pageshow',returned)
    return()=>{active=false;data.subscription.unsubscribe();window.removeEventListener('pageshow',returned)}
  },[])
  if(!supabase)return <App/>
  const client=supabase

  function begin(action:'password'|'google'|'signout'){
    if(pending.current)return false
    pending.current=true;setBusy(action);setError('');return true
  }
  function finish(){pending.current=false;setBusy(null)}
  async function signIn(){
    if(!begin('password'))return
    try{
      const {error}=await client.auth.signInWithPassword({email,password})
      if(error)throw error
      setPassword('')
    }catch(e){setError(e instanceof Error?e.message:'Sign-in failed')}
    finally{finish()}
  }
  async function google(){
    if(!begin('google'))return
    try{await startGoogleSignIn(client.auth,location.origin)}
    catch(e){setError(e instanceof Error?e.message:'Google sign-in failed');finish()}
  }
  async function signOut(){
    if(!begin('signout'))return
    try{
      const {error}=await client.auth.signOut()
      if(error)throw error
      setSession(null);setPassword('')
      const url=new URL(location.href);url.searchParams.delete('analysis');url.searchParams.delete('job')
      history.replaceState(history.state,'',url.pathname+url.search+url.hash)
    }catch{setError('Sign-out failed. Please try again.')}
    finally{finish()}
  }
  if(!loaded)return <main className="auth-panel" role="status">Restoring your sign-in session…</main>
  if(session)return <><div className="account-bar"><span>Signed in: {session.user.email}</span><button disabled={!!busy} onClick={()=>void signOut()}>{busy==='signout'?'Signing out…':'Sign out'}</button>{error&&<span role="alert" className="error">{error}</span>}</div><App key={session.user.id}/></>
  return <main className="auth-panel"><p className="eyebrow">SEMATRACE AI · PRIVATE WORKSPACE</p><h1>Sign in to compare documents.</h1><p>Search the shared global corpus and your private references. Your submissions and reports remain private to your account. Cloud storage uses Supabase and vector retrieval uses Pinecone. OpenAI and web discovery require separate opt-in.</p>
    <button type="button" className="google-sign-in" disabled={!!busy} onClick={()=>void google()}>{busy==='google'?'Connecting to Google…':'Continue with Google'}</button>
    <p className="auth-divider">or sign in with email</p>
    <form onSubmit={e=>{e.preventDefault();void signIn()}}><label>Email<input type="email" autoComplete="username" required disabled={!!busy} value={email} onChange={e=>setEmail(e.target.value)}/></label><label>Password<input type="password" autoComplete="current-password" required disabled={!!busy} value={password} onChange={e=>setPassword(e.target.value)}/></label><button className="primary-button" disabled={!!busy}>{busy==='password'?'Signing in…':'Sign in'}</button></form>
    {error&&<p role="alert" className="error">{error}</p>}<p className="muted">New Google accounts receive ordinary-user access. Contact the deployment’s operator if you need access or a password reset.</p>
  </main>
}
