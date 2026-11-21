import { segments, safeExternalUrl } from './evidence'
type Run = { submission_start: number; submission_end: number; source_start: number; source_end: number; words: number }
export type PassageFinding = { finding_id: string; category: string; source_document_id: string; source_revision: string; source_title: string; source_url: string; submission_start: number; submission_end: number; source_start: number; source_end: number; submission_excerpt: string; source_excerpt: string; uncertainty: string; matched_runs: Run[]; citation_context?: string | null; source_attribution?: {authors: string[]; license_name: string | null; license_url: string | null; attribution_text: string | null; attribution_url: string | null} }
export type RelatedSource = { submission_excerpt: string; source_excerpt: string; source_title: string; source_url: string; source_revision: string; source_start: number; source_end: number; uncertainty: string }

function Marked({ text, ranges }: { text: string; ranges: Array<{start: number; end: number}> }) {
  return <blockquote>{segments(text, ranges.map(r => ({...r, kind: 'lexical' as const}))).map(p => p.kind ? <mark key={p.start}>{p.text}</mark> : <span key={p.start}>{p.text}</span>)}</blockquote>
}
export default function PassageFindings({findings, related, excluded}: {findings: PassageFinding[]; related: RelatedSource[]; excluded: string[]}) {
  return <section className="evidence">
    <h3>Aligned passages ({findings.length})</h3>
    <p>Corresponding regions may contain edits. Highlighted word runs show lexical correspondence; the textual similarity report applies coverage exclusions and minimum lengths.</p>
    {findings.map(f => <article className="evidence-card" key={f.finding_id}>
      <div><p className="evidence-label">{f.category === 'EXACT_REUSE' ? 'Exact wording' : 'Modified wording'}{excluded.includes(f.source_document_id) ? ' · Source excluded from coverage' : ''}</p>
        <Marked text={f.submission_excerpt} ranges={f.matched_runs.map(r => ({start: r.submission_start-f.submission_start, end: r.submission_end-f.submission_start}))}/>
        <p className="muted">Submission characters [{f.submission_start}, {f.submission_end})</p></div>
      <div><p className="evidence-label">{f.source_title}</p>
        <Marked text={f.source_excerpt} ranges={f.matched_runs.map(r => ({start: r.source_start-f.source_start, end: r.source_end-f.source_start}))}/>
        <p className="muted">Source characters [{f.source_start}, {f.source_end}) · Revision {f.source_revision}</p>
        {safeExternalUrl(f.source_url) ? <a href={safeExternalUrl(f.source_url)!} target="_blank" rel="noopener noreferrer">Original source</a> : <span>{f.source_url}</span>}
        {f.source_attribution && <p className="muted">{f.source_attribution.authors.join(', ') || 'Author not supplied'} · {safeExternalUrl(f.source_attribution.license_url ?? '') ? <a href={safeExternalUrl(f.source_attribution.license_url ?? '')!} target="_blank" rel="noopener noreferrer">{f.source_attribution.license_name ?? 'Source license'}</a> : f.source_attribution.license_name ?? 'License not supplied'}{f.source_attribution.attribution_text && <> · {f.source_attribution.attribution_text}</>}{safeExternalUrl(f.source_attribution.attribution_url ?? '') && <> · <a href={safeExternalUrl(f.source_attribution.attribution_url ?? '')!} target="_blank" rel="noopener noreferrer">Attribution and history</a></>}</p>}
        <p className="muted">Citation marker: {f.citation_context ? 'present' : 'not detected'}. Source support: NOT_ASSESSED. A marker does not establish claim support.</p>
        <p className="muted">{f.uncertainty}</p></div>
    </article>)}
    {!findings.length && <p>No lexical passage met the aligner's evidence requirements. This does not establish originality.</p>}
    {related.length > 0 && <details className="report"><summary>Related source discovery ({related.length})</summary>
      <p>Transformer-ranked source context for exploration. These results do not establish paraphrased reuse and receive no similarity coverage.</p>
      {related.map((r,i) => <article className="evidence-card" key={i}><div><p>Submitted sentence</p><blockquote>{r.submission_excerpt}</blockquote></div><div><p>{r.source_title}</p><blockquote>{r.source_excerpt}</blockquote><p className="muted">Source [{r.source_start}, {r.source_end}) · {r.uncertainty}</p>{safeExternalUrl(r.source_url) && <a href={safeExternalUrl(r.source_url)!} target="_blank" rel="noopener noreferrer">Original source</a>}</div></article>)}
    </details>}
  </section>
}
