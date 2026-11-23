import { useEffect,useState } from 'react'
import type { Session } from '@supabase/supabase-js'
import { supabase } from './auth'
import App from './App'

export default function AuthGate(){
  const [session,setSession]=useState<Session|null>(null),[loaded,setLoaded]=useState(!supabase)
  const [email,setEmail]=useState(''),[password,setPassword]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false)
  useEffect(()=>{
    if(!supabase)return
    let active=true
    supabase.auth.getSession().then(({data,error})=>{if(active){setSession(data.session);setLoaded(true);if(error)setError('Sign-in service unavailable. Retry shortly.')}})
    const {data}=supabase.auth.onAuthStateChange((_event,next)=>{if(active){setSession(next);setLoaded(true)}})
    return()=>{active=false;data.subscription.unsubscribe()}
  },[])
  if(!supabase)return <App/>
  if(!loaded)return <main className="auth-panel" role="status">Restoring your sign-in session…</main>
  if(session)return <><div className="account-bar"><span>Signed in: {session.user.email}</span><button onClick={()=>void supabase!.auth.signOut()}>Sign out</button></div><App key={session.user.id}/></>
  async function signIn(){
    setBusy(true);setError('')
    try{
      const {error}=await supabase!.auth.signInWithPassword({email,password})
      if(error)throw error
      setPassword('')
    }catch(e){setError(e instanceof Error?e.message:'Sign-in failed')}
    finally{setBusy(false)}
  }
  return <main className="auth-panel"><p className="eyebrow">SEMATRACE AI · PRIVATE WORKSPACE</p><h1>Sign in to compare documents.</h1><p>Search the shared global corpus and your private references. Your submissions and reports remain private to your account. Cloud storage uses Supabase and vector retrieval uses Pinecone. OpenAI and web discovery require separate opt-in.</p><form onSubmit={e=>{e.preventDefault();void signIn()}}><label>Email<input type="email" autoComplete="username" required value={email} onChange={e=>setEmail(e.target.value)}/></label><label>Password<input type="password" autoComplete="current-password" required value={password} onChange={e=>setPassword(e.target.value)}/></label><button className="primary-button" disabled={busy}>{busy?'Signing in…':'Sign in'}</button></form>{error&&<p role="alert" className="error">{error}</p>}<p className="muted">Accounts are provisioned by this deployment’s operator. Contact them if you need access or a password reset.</p></main>
}
