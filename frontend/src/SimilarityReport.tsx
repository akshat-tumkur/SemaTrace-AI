import { useEffect, useState } from 'react'
import { safeExternalUrl, type Evidence } from './evidence'
import { addDocumentExclusion, selectedSourceSegments, sourceEvidence, sourceExcerpt, type SimilarityOptions, type SimilarityReportData } from './similarity'

type Props = {
  report: SimilarityReportData
  text?: string
  evidence: Evidence[]
  onRecalculate: (options: SimilarityOptions) => Promise<void>
}

export default function SimilarityReport({ report, text, evidence, onRecalculate }: Props) {
  const [selected, setSelected] = useState(report.sources[0]?.source_id ?? '')
  const [options, setOptions] = useState<SimilarityOptions>(report.options)
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    setOptions(report.options)
    setSelected(previous => report.sources.some(source => source.source_id === previous) ? previous : report.sources[0]?.source_id ?? '')
  }, [report])
  const source = report.sources.find(item => item.source_id === selected)
  const excerpts = source ? sourceEvidence(evidence, source.source_id).map(item => sourceExcerpt(item, source.source_id)) : []

  async function apply() {
    setError(''); setPending(true)
    try { await onRecalculate(options) }
    catch (e) { setError(e instanceof Error ? e.message : 'Similarity recalculation failed.') }
    finally { setPending(false) }
  }

  function addRange() {
    try {
      if (text === undefined || !start.trim() || !end.trim()) throw new Error('Enter both character offsets.')
      setOptions(current => ({ ...current, excluded_ranges: addDocumentExclusion(current.excluded_ranges, Number(start), Number(end), Array.from(text).length) }))
      setStart(''); setEnd(''); setError('')
    } catch (e) { setError(e instanceof Error ? e.message : 'Invalid exclusion range.') }
  }

  return <section className="similarity-report" aria-label="Preliminary textual similarity report">
    <div className="similarity-heading"><div><p className="eyebrow">Matching-word coverage</p><h3>{report.textual_similarity_percent}% <small>{report.unique_matched_words} unique matched / {report.total_eligible_words} eligible words</small></h3></div><span className="mode-tag">{report.method}</span></div>
    <p>This metric counts grounded matching words across indexed sources. Similarity is not a probability of plagiarism. Related source discovery receives no coverage credit.</p>
    <div className="similarity-grid">
      <div><h4>Source contributions</h4>{report.sources.length ? <div className="similarity-source-list">{report.sources.map(item => <button type="button" key={item.source_id} aria-pressed={selected === item.source_id} onClick={() => setSelected(item.source_id)}>
        <strong>{item.name || item.source_id}</strong><span>{item.attributed_matched_words} attributed words / {item.attributed_percent}% of eligible text</span><span>{item.raw_matched_words} words raw overlap{item.excluded ? ' · excluded from score' : ''}</span>
      </button>)}</div> : <p>No source has score-eligible textual overlap.</p>}<p className="muted">A word contributes once overall. Primary contributions partition the matched words; raw source overlaps can overlap and must not be added together.</p></div>
      <div><h4>{source ? `Matching document spans: ${source.name}` : 'Matching document spans'}</h4>
        {source && source.excluded && <p className="muted">This source is excluded from the score. Its matching intervals remain visible for inspection.</p>}
        {text !== undefined ? <div className="document-text similarity-document">{selectedSourceSegments(text, source).map(part => part.kind ? <mark className="similarity-highlight" key={part.start}>{part.text}</mark> : <span key={part.start}>{part.text}</span>)}</div> : <p>Normalized submission text is unavailable; no span view can be shown.</p>}
        {source && <>
          <p className="muted">{(source.excluded ? source.raw_intervals ?? [] : source.intervals).length} matched interval(s). Positions use Unicode code points, with exclusive end offsets.</p>
          {safeExternalUrl(source.url) ? <a href={safeExternalUrl(source.url)!} target="_blank" rel="noopener noreferrer">Open original source</a> : <span>{source.url}</span>}
          {source.alternative_source_ids.length > 0 && <p className="muted">Overlapping alternatives: {source.alternative_source_ids.map(id => report.sources.find(item => item.source_id === id)?.name ?? id).join(', ')}</p>}
          <details><summary>Retrieved source excerpts ({excerpts.length})</summary>{excerpts.length ? excerpts.map((item, i) => <div key={i}><p>{item.title}</p><blockquote>{item.text}</blockquote>{item.document_id && <p className="muted">{item.document_id} / source characters [{item.start}, {item.end})</p>}</div>) : <p>See the source link and ledger metadata; an original excerpt is not available in this saved record.</p>}</details>
        </>}
      </div>
    </div>
    <details className="similarity-controls"><summary>Exclusions and recalculation</summary>
      <p>Excluding a source removes its matches only. Explicit document ranges remove intersecting words from both matched words and the denominator. Quotes and bibliographies are not automatically identified.</p>
      <label>Minimum contiguous matched words <input type="number" min="3" max="100" value={options.minimum_match_words} onChange={e => setOptions(current => ({ ...current, minimum_match_words: Number(e.target.value) }))} /></label>
      <fieldset><legend>Exclude selected sources</legend>{report.sources.map(item => <label key={item.source_id}><input type="checkbox" checked={options.excluded_source_ids.includes(item.source_id)} onChange={e => setOptions(current => ({ ...current, excluded_source_ids: e.target.checked ? [...current.excluded_source_ids, item.source_id] : current.excluded_source_ids.filter(id => id !== item.source_id) }))} /> {item.name || item.source_id}</label>)}</fieldset>
      <div className="similarity-range-input"><label>Start character <input type="number" min="0" value={start} onChange={e => setStart(e.target.value)} /></label><label>End character (exclusive) <input type="number" min="1" value={end} onChange={e => setEnd(e.target.value)} /></label><button type="button" disabled={text === undefined} onClick={addRange}>Add document exclusion</button></div>
      {options.excluded_ranges.map((range, i) => <p key={i}>[{range.start}, {range.end}) — {range.reason} <button type="button" onClick={() => setOptions(current => ({ ...current, excluded_ranges: current.excluded_ranges.filter((_, index) => i !== index) }))}>Remove</button></p>)}
      <p className="muted">Current report excludes {report.exclusions.excluded_document_words} document word(s). Editing controls does not change this report until recalculated.</p>
      <button type="button" disabled={pending || !Number.isInteger(options.minimum_match_words) || options.minimum_match_words < 3 || options.minimum_match_words > 100} onClick={() => void apply()}>{pending ? 'Recalculating…' : 'Recalculate and save preliminary metric'}</button>
      {error && <p role="alert" className="similarity-error">{error}</p>}
    </details>
    <details><summary>Excluded and uncertain evidence</summary>
      <p>{report.unresolved_findings.length} potential or unresolved finding(s), separate from confirmed textual overlap.</p>
      {report.unresolved_findings.map((finding, i) => <p key={i}>{finding.candidate_id ?? 'Evidence'}{finding.category ? ` / ${finding.category}` : ''}: {finding.reason}</p>)}
      <p>Ledger entries: {report.ledger.length}. Complete intervals, eligibility and provenance are included in the JSON download.</p>
    </details>
    <details><summary>Method and limits</summary><p>{typeof report.methodology === 'string' ? report.methodology : 'Unique eligible matched document words divided by eligible document words, multiplied by 100.'}</p>{typeof report.methodology !== 'string' && <pre>{JSON.stringify(report.methodology, null, 2)}</pre>}<ul>{report.limitations.map((limitation, i) => <li key={i}>{limitation}</li>)}</ul></details>
  </section>
}
