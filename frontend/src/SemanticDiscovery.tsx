import { useEffect, useState } from 'react'
import { safeExternalUrl } from './evidence'
import { readJsonResponse } from './api'
import { summarizeDiscovery } from './discovery'

export type DiscoveryCandidate = {
  candidate_id: string; submission_start: number; submission_end: number; submission_excerpt: string
  original_submission_start: number; original_submission_end: number; original_submission_excerpt: string
  source_document_id: string; source_revision: string; source_document_sha256: string
  source_start: number; source_end: number; source_excerpt: string; source_title: string; source_url: string
  minilm_rank: number; score: number; citation_context?: string | null
  source_attribution?: { authors: string[]; license_name: string | null; attribution_text: string | null }
}
type ReviewRanking = { ranking: {candidate_id: string; margin: number; truncated: boolean}[]; elapsed_seconds: number; limitation: string }
type Investigation = { explanation: {status: string; similarities?: string; differences?: string; uncertainty?: string; submission_quote?: string; source_quote?: string; warning: string; source_document_id?: string; source_revision?: string}; contexts?: DiscoveryCandidate[]; trace: {node: string; action?: string; tool?: string; reason?: string; details?: Record<string, unknown>}[]; elapsed_seconds: number; model: string }
export type SavedSemanticReview = {rankings?: Record<string, ReviewRanking>; investigations?: Record<string, Investigation>}
const API = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')

export default function SemanticDiscovery({ id, rows, review, onSaved }: {id: string; rows: DiscoveryCandidate[]; review: SavedSemanticReview; onSaved: () => Promise<void>}) {
  const [open, setOpen] = useState(false), [selected, setSelected] = useState(''), [busy, setBusy] = useState(''), [error, setError] = useState('')
  const [caps, setCaps] = useState({roberta_configured: false, investigation_enabled: false})
  const starts = [...new Set(rows.map(r => r.submission_start))]
  const start = starts.includes(Number(selected)) && selected !== '' ? Number(selected) : starts[0]
  const group = rows.filter(r => r.submission_start === start)
  const ranking = review.rankings?.[String(start)]
  const overview = summarizeDiscovery(rows)
  useEffect(() => {fetch(API+'/api/semantic/capabilities').then(readJsonResponse).then(setCaps).catch(() => {})}, [])
  async function act(kind: 'rerank' | 'investigate', candidate: string) {
    setBusy(kind); setError('')
    try {
      const response = await fetch(`${API}/api/analysis/${id}/semantic/${kind}`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({candidate_id:candidate}), signal: AbortSignal.timeout(kind === 'investigate' ? 300_000 : 90_000)})
      await readJsonResponse(response)
      await onSaved()
    } catch(e) {setError(e instanceof Error ? e.message : 'Semantic review unavailable')}
    finally {setBusy('')}
  }
  return <section className="semantic-discovery report" data-testid="semantic-discovery">
    <p className="eyebrow">SEMANTIC DISCOVERY / REVIEW ONLY</p>
    <h3>Retrieved sources and passage comparisons</h3>
    <p>Related source passages were retrieved for {starts.length} submitted text units. Compare the wording and meaning below. These are review candidates; related independent writing can also rank highly.</p>
    <button className="secondary" aria-expanded={open} onClick={() => setOpen(!open)}>{open ? 'Close semantic exploration' : 'Explore semantic matches'}</button>
    {!!overview.topSources.length && <div className="semantic-overview">
      <h4>Sources to inspect</h4>
      <div className="semantic-source-list">{overview.topSources.slice(0,4).map(source => <button key={source.id} onClick={() => {setSelected(String(source.best.submission_start));setOpen(true)}}><strong>{source.title}</strong><span>First-ranked for {source.firstRankedUnits} of {overview.unitCount} text units</span></button>)}</div>
      <p className="muted">These counts describe retrieval rankings, not the number or percentage of reused sentences. The purple contexts receive no matching-word coverage credit.</p>
    </div>}
    {!open && !!overview.previews.length && <div className="semantic-previews">
      <h4>Highest-ranked passage comparisons</h4>
      <p className="muted">Showing {overview.previews.length} of {overview.unitCount} first-ranked contexts, ordered by MiniLM cosine similarity. A high rank does not establish copying.</p>
      {overview.previews.map(r => <article className="semantic-preview" key={r.candidate_id} data-preview-candidate={r.candidate_id}>
        <div className="semantic-pair-heading"><strong>{r.source_title}</strong><span>Cosine ranking signal {r.score.toFixed(3)}</span></div>
        <div className="evidence-card"><div><p className="evidence-label">Submitted passage</p><blockquote><mark className="semantic-context">{r.submission_excerpt}</mark></blockquote></div><div><p className="evidence-label">Actual indexed source passage</p><blockquote><mark className="semantic-context">{r.source_excerpt}</mark></blockquote><p className="muted">Revision {r.source_revision} · source [{r.source_start}, {r.source_end})</p></div></div>
        <button onClick={() => {setSelected(String(r.submission_start));setOpen(true)}}>Inspect this passage and alternatives</button>
      </article>)}
    </div>}
    {open && <>
      {!rows.length ? <p>No saved semantic candidates. Reanalyze an older report to generate them.</p> : <>
        <label className="semantic-selector">Submitted sentence <select value={String(start)} onChange={e => setSelected(e.target.value)}>{starts.map((n,i) => <option key={n} value={String(n)}>{i+1}. {rows.find(r => r.submission_start === n)!.submission_excerpt.slice(0,100)}</option>)}</select></label>
        <p className="muted">Default order: MiniLM cosine ranking. Cosine and RoBERTa logit margins are ranking signals, never plagiarism probabilities or similarity percentages.</p>
        <button className="secondary" disabled={!!busy || !caps.roberta_configured || !!ranking} onClick={() => void act('rerank',group[0].candidate_id)}>{busy === 'rerank' ? 'Comparing up to 8 passages…' : ranking ? 'RoBERTa comparison saved' : 'Compare experimental RoBERTa ranking'}</button>
        <details><summary>Optional RoBERTa comparison status</summary><p className="muted">{!caps.roberta_configured && 'Optional private RoBERTa weights are not configured. '}Transfer check: MiniLM Hit@1 16/16; RoBERTa 14/16 on a small controlled fixture. No source-ranking improvement demonstrated; MiniLM ordering is retained.</p></details>
        {ranking && <p className="muted">Saved RoBERTa comparison: {ranking.elapsed_seconds.toFixed(2)} seconds. {ranking.limitation}</p>}
        {group.map(r => {
          const rr = ranking?.ranking.findIndex(v => v.candidate_id === r.candidate_id)
          const margin = ranking?.ranking.find(v => v.candidate_id === r.candidate_id)
          const investigation = review.investigations?.[r.candidate_id]
          return <article className="semantic-pair" key={r.candidate_id} data-candidate-id={r.candidate_id}>
            <div className="semantic-pair-heading"><strong>Potential correspondence · MiniLM #{r.minilm_rank}</strong><span>cosine {r.score.toFixed(3)}{rr !== undefined && rr >= 0 && <> · RoBERTa #{rr+1} · margin {margin!.margin.toFixed(3)}{margin!.truncated && ' (input truncated)'}</>}</span></div>
            <div className="evidence-card"><div><p className="evidence-label">Submitted sentence · context highlight</p><blockquote><mark className="semantic-context">{r.submission_excerpt}</mark></blockquote><p className="muted">Normalized characters [{r.submission_start}, {r.submission_end})</p><details><summary>Verified original extracted-text interval</summary><p>Unicode characters [{r.original_submission_start}, {r.original_submission_end}). No PDF page coordinates.</p><pre className="original-excerpt">{r.original_submission_excerpt}</pre></details></div>
              <div><p className="evidence-label">{r.source_title} · retrieved passage</p><blockquote><mark className="semantic-context">{r.source_excerpt}</mark></blockquote><p className="muted">Indexed source characters [{r.source_start}, {r.source_end}) · Revision {r.source_revision}</p><p className="muted">{r.source_attribution?.authors.join(', ') || 'Author not supplied'} · {r.source_attribution?.license_name ?? 'Private or unverified rights'}{r.source_attribution?.attribution_text && <> · {r.source_attribution.attribution_text}</>}</p>{safeExternalUrl(r.source_url) && <a href={safeExternalUrl(r.source_url)!} target="_blank" rel="noopener noreferrer">Original source</a>}<details><summary>Immutable source identity</summary><p>{r.source_document_id}</p><p className="hash-text">SHA256 {r.source_document_sha256}</p></details></div></div>
            <p className="muted">Purple marks retrieved context, not proven paraphrase boundaries. Citation marker: {r.citation_context ? `present (${r.citation_context})` : 'not detected'}; claim support is not assessed.</p>
            <button className="secondary" disabled={!!busy || !caps.investigation_enabled || !!investigation} onClick={() => void act('investigate',r.candidate_id)}>{busy === 'investigate' ? 'Investigating locally…' : investigation ? 'Local investigation saved' : 'Investigate this correspondence'}</button>
            {!caps.investigation_enabled && <p className="muted">Optional investigation requires LangGraph and a configured local Ollama model. Core discovery works without it.</p>}
            {investigation && <div className="investigation"><h4>Local investigation · {investigation.model}</h4><p>{investigation.explanation.warning}</p>{investigation.explanation.submission_quote && <><p>Grounded submission quote</p><blockquote>{investigation.explanation.submission_quote}</blockquote><p>Grounded source quote</p><blockquote>{investigation.explanation.source_quote}</blockquote><p className="muted">Source {investigation.explanation.source_document_id} · Revision {investigation.explanation.source_revision}</p><p><strong>Model-reported similarity:</strong> {investigation.explanation.similarities}</p><p><strong>Differences:</strong> {investigation.explanation.differences}</p><p><strong>Uncertainty:</strong> {investigation.explanation.uncertainty}</p></>}<details><summary>Retrieved investigation context ({investigation.contexts?.length ?? 0})</summary>{investigation.contexts?.map((c,i) => <div key={i}><p>{c.source_title} · source [{c.source_start}, {c.source_end}) · revision {c.source_revision}</p><blockquote>{c.source_excerpt}</blockquote><p className="muted hash-text">{c.source_document_id}</p></div>)}</details><details><summary>Conditional tool trace · {investigation.elapsed_seconds.toFixed(1)} seconds</summary><ol>{investigation.trace.map((t,i) => <li key={i}>{t.node}: {t.action ?? t.tool ?? ''} {t.reason}{t.details && <pre className="original-excerpt">{JSON.stringify(t.details,null,2)}</pre>}</li>)}</ol></details></div>}
          </article>
        })}
      </>}
      {error && <p className="error" role="alert">{error}</p>}
    </>}
  </section>
}
