import type { SupabaseClient } from '@supabase/supabase-js'

type Auth = SupabaseClient['auth']
const callbackKeys = ['code','sb_flow_id','error','error_code','error_description','access_token','refresh_token','provider_token','provider_refresh_token','expires_in','expires_at','token_type']
const googleFailure = 'Google sign-in could not be completed. Try again in this browser or use email and password.'

export function authCallback(href:string) {
  const url = new URL(href), fragment = new URLSearchParams(url.hash.slice(1))
  const denied = [url.searchParams,fragment].some(p=>p.get('error')==='access_denied')
  const fragmentCallback = callbackKeys.some(key=>fragment.has(key))
  const present = fragmentCallback || callbackKeys.some(key=>url.searchParams.has(key))
  const message = denied ? 'Google sign-in was cancelled or denied. Try again or use email and password.' : googleFailure
  for(const key of callbackKeys){url.searchParams.delete(key);fragment.delete(key)}
  if(fragmentCallback)url.hash=fragment.toString()
  return {present,message,cleanPath:url.pathname+url.search+url.hash}
}

export async function restoreAuthSession(auth:Pick<Auth,'initialize'|'getSession'>,href:string) {
  const callback = authCallback(href)
  // The SDK owns PKCE exchange and shares its initialization promise, including
  // across React StrictMode effects. Never exchange the callback code twice.
  const initialized = await auth.initialize()
  if(initialized.error)throw new Error(callback.present ? callback.message : 'Your sign-in session could not be restored. Please sign in again.')
  const {data,error} = await auth.getSession()
  if(error)throw new Error('Your sign-in session could not be restored. Please sign in again.')
  if(callback.present && !data.session)throw new Error(callback.message)
  return data.session
}

export async function startGoogleSignIn(auth:Pick<Auth,'signInWithOAuth'>,origin:string) {
  const redirectTo = new URL('/',origin).href
  if(!['https:','http:'].includes(new URL(redirectTo).protocol))throw new Error('Invalid sign-in return address.')
  try {
    const {data,error} = await auth.signInWithOAuth({provider:'google',options:{redirectTo}})
    if(error || !data.url)throw new Error('OAuth unavailable')
  } catch {
    throw new Error('Google sign-in could not start. Please try again or use email and password.')
  }
}
