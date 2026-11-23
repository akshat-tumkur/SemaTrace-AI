import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import { env as nodeEnv } from 'node:process'
import { Buffer } from 'node:buffer'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', 'SEMATRACE_API_PROXY_TARGET')
  const publicEnv={...loadEnv(mode,'.','VITE_'),...nodeEnv}
  const key=publicEnv.VITE_SUPABASE_PUBLISHABLE_KEY || publicEnv.VITE_SUPABASE_ANON_KEY || ''
  if(key.startsWith('sb_secret_'))throw new Error('A Supabase server secret must never be used as a VITE_ variable')
  if(key.split('.').length===3){
    let role=''
    try{role=JSON.parse(Buffer.from(key.split('.')[1],'base64url').toString()).role}catch{throw new Error('Invalid Supabase public key')}
    if(role!=='anon')throw new Error('Only the Supabase anon/public key belongs in the frontend')
  }
  if(!!publicEnv.VITE_SUPABASE_URL!==!!key)throw new Error('Set both VITE_SUPABASE_URL and VITE_SUPABASE_PUBLISHABLE_KEY')
  for(const name of Object.keys(publicEnv)){
    if(name.startsWith('VITE_') && /OPENAI|TAVILY|SERVICE_ROLE|DATABASE|PINECONE|SECRET/i.test(name))throw new Error('Server credentials must not use the VITE_ prefix')
  }
  return {
    plugins: [react()],
    // The native launcher serves the built frontend and API on one origin.
    server: { proxy: { '/api': env.SEMATRACE_API_PROXY_TARGET || 'http://127.0.0.1:8010' } },
  }
})
