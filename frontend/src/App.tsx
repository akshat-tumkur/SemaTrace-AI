import { useState } from 'react'
import { ArrowUpRight, Download, FileSearch, LoaderCircle, ShieldCheck, Upload } from 'lucide-react'

type Analysis = {
  analysis_id: string
  filename: string
  status: string
  risk_level: string
  coverage_percent: number
  strong_matches: number
  possible_paraphrases: number
  false_positives: number
  candidate_count: number
  risk_score: number
  mode: string
  audit_log: Array<{ agent: string; event: string; details: string }>
  summary: string
  recommendations: string[]
  matches: Array<{ unit_id: string; submission_text: string; source_text: string; source_title: string; source_url: string; lexical_score?: number; semantic_score?: number; retrieval_score: number; verification_score: number; match_type?: string }>
  message?: string
}

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [sourceFile, setSourceFile] = useState<File | null>(null)
  const [sourceTitle, setSourceTitle] = useState('')
  const [sourceUrl, setSourceUrl] = useState('')
  const [sourceMessage, setSourceMessage] = useState('')
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function runAnalysis() {
    if (!file) return
    setBusy(true)
    setError('')
    setAnalysis(null)
    const body = new FormData()
    body.append('file', file)
    try {
      const response = await fetch(`${API_URL}/api/analyze`, { method: 'POST', body })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail ?? 'Analysis failed')
      setAnalysis(payload)
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Analysis failed')
    } finally {
      setBusy(false)
    }
  }

  async function downloadReport() {
    if (!analysis) return
    const response = await fetch(`${API_URL}/api/analysis/${analysis.analysis_id}/report`)
    if (!response.ok) return
    const blob = await response.blob()
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${analysis.filename.replace(/\.[^.]+$/, '')}-sematrace-report.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  async function indexSource() {
    if (!sourceFile || !sourceTitle || !sourceUrl) return
    const body = new FormData()
    body.append('file', sourceFile)
    body.append('title', sourceTitle)
    body.append('url', sourceUrl)
    const response = await fetch(`${API_URL}/api/sources`, { method: 'POST', body })
    const payload = await response.json()
    setSourceMessage(response.ok ? `Indexed ${payload.units_indexed} source units` : payload.detail ?? 'Source indexing failed')
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark">ST</span><span>SemaTrace <em>AI</em></span></div>
        <span className="status-dot">Agent workflow online</span>
      </header>
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Evidence-backed similarity analysis</p>
          <h1>Trace the thinking,<br /><span>not just the words.</span></h1>
          <p className="lede">SemaTrace decomposes a document into claims, finds meaningful overlap, and leaves a trail of evidence a human can inspect.</p>
        </div>
        <div className="upload-panel">
          <div className="panel-heading"><div><p className="eyebrow">Start an investigation</p><h2>Upload a document</h2></div><FileSearch size={22} /></div>
          <label className="dropzone">
            <input type="file" accept=".txt,.pdf,.docx" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
            <Upload size={28} />
            <strong>{file ? file.name : 'Choose a UTF-8 text file'}</strong>
            <span>{file ? `${(file.size / 1024).toFixed(1)} KB ready` : 'TXT, PDF, and DOCX supported'}</span>
          </label>
          <button className="primary-button" disabled={!file || busy} onClick={runAnalysis}>
            {busy ? <><LoaderCircle className="spin" size={18} /> Agents investigating</> : <>Run analysis <ArrowUpRight size={18} /></>}
          </button>
          {error && <p className="error">{error}</p>}
          <div className="source-index"><p className="eyebrow">Known source, optional</p><strong>Index a source before checking a copy</strong><input type="file" accept=".txt,.pdf,.docx" onChange={(event) => setSourceFile(event.target.files?.[0] ?? null)} /><input placeholder="Source title" value={sourceTitle} onChange={(event) => setSourceTitle(event.target.value)} /><input placeholder="Source URL" value={sourceUrl} onChange={(event) => setSourceUrl(event.target.value)} /><button className="secondary-button" disabled={!sourceFile || !sourceTitle || !sourceUrl} onClick={indexSource}>Index source</button>{sourceMessage && <span>{sourceMessage}</span>}</div>
        </div>
      </section>
      {analysis ? <section className="results">
        <div className="results-heading"><div><p className="eyebrow">Investigation complete</p><h2>{analysis.filename}</h2></div><div className="result-actions"><span className="mode-tag">{analysis.mode}</span><button className="icon-button" title="Download forensic report" onClick={downloadReport}><Download size={17} /></button></div></div>
        <div className="metrics">
          <Metric label="Risk level" value={analysis.risk_level} accent />
          <Metric label="Affected text" value={`${analysis.coverage_percent}%`} />
          <Metric label="Strong matches" value={analysis.strong_matches.toString()} />
          <Metric label="Possible paraphrases" value={analysis.possible_paraphrases.toString()} />
        </div>
        <div className="activity"><div className="activity-title"><ShieldCheck size={18} /><h3>Agent activity <span>({analysis.audit_log.length} stages)</span></h3></div><div className="timeline">{analysis.audit_log.map((event, index) => <Activity key={`${event.agent}-${index}`} label={event.agent} detail={event.details} />)}</div></div>
        <div className="evidence"><div className="activity-title"><FileSearch size={18} /><h3>Verified evidence matches <span>({analysis.matches.length} of {analysis.candidate_count} candidates)</span></h3></div>{analysis.matches.length ? analysis.matches.map((match) => <article className="evidence-card" key={`${match.unit_id}-${match.source_url}`}><div><p className="evidence-label">Submission / {match.unit_id} / {match.match_type ?? 'VERIFIED'}</p><p>{match.submission_text}</p></div><div><p className="evidence-label">{match.source_title} / confidence {match.verification_score.toFixed(2)}</p><p>{match.source_text}</p><a href={match.source_url}>{match.source_url}</a></div></article>) : <p className="muted">No candidates passed evidence verification.</p>}</div>
        <div className="report"><p className="eyebrow">Forensic report</p><p className="report-summary">{analysis.summary}</p><ul>{analysis.recommendations.map((recommendation) => <li key={recommendation}>{recommendation}</li>)}</ul></div>
      </section> : <section className="empty-state"><div className="empty-icon"><FileSearch size={24} /></div><h2>Your evidence trail starts here.</h2><p>Upload a sample document to see the first deterministic analysis pass.</p></section>}
      <footer>SEMA TRACE / MVP 0.1 <span>Built for transparent investigation</span></footer>
    </main>
  )
}

function Metric({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return <div className={`metric ${accent ? 'accent' : ''}`}><span>{label}</span><strong>{value}</strong></div>
}

function Activity({ label, detail }: { label: string; detail: string }) {
  return <div className="activity-row"><span className="check">✓</span><div><strong>{label}</strong><span>{detail}</span></div></div>
}
