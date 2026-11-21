export type Evidence = {
  candidate_id?: string
  unit_id: string
  submission_text: string
  source_text: string
  source_title: string
  source_url: string
  source_document_id?: string
  source_start_char?: number
  source_end_char?: number
  source_occurrences?: Array<{
    occurrence_id: string
    source_document_id: string
    source_start_char: number
    source_end_char: number
    title: string
    url: string
    authors: string[]
    collection: string
    license_url: string | null
    license_name: string | null
    rights_status: string
    attribution_url: string | null
    attribution_text: string | null
  }>
  retrieval_score: number
  verification_score: number
  match_type?: string
  final_match_status?: string
  verification_method?: string
  explanation?: string
  effective_semantic_support?: { status: 'FULL' | 'PARTIAL' | 'NONE' | 'UNKNOWN'; basis: string; reason_codes: string[]; independently_validated: boolean; semantic_support_coverage_permitted: boolean; raw_verdict: string | null }
  citation_present?: boolean
  citation_support_status?: string
  evidence_assessment?: { lexical_status: string; semantic_status: string }
  dual_assessment?: {
    textual_reuse: { verdict: 'YES' | 'NO' | 'UNCERTAIN'; category: string; confidence: string; evidence: DualPair[]; explanation: string }
    semantic_support: { verdict: 'FULL' | 'PARTIAL' | 'NONE' | 'UNCERTAIN'; confidence: string; supporting_spans: DualPair[]; unsupported_or_changed_details: ChangedDetail[]; explanation: string }
    limitations: string[]
  }
  verification_diagnostics?: Array<{ code: string; field: string; side: string | null; occurrence: number | null; explanation: string; unresolved: boolean }>
  llm_verdict?: string | null
  llm_verification?: {
    confidence: string
    supporting_spans?: Array<{ submission_span: string; source_span: string }>
    contradictions?: string[]
    limitations?: string[]
  }
}

export type Unit = { id: string; text: string; start_char: number; end_char: number }
export type DualPair = { submission_span: string; source_span: string; submission_occurrence: number; source_occurrence: number }
type ChangedDetail = { submission_span: string; submission_occurrence: number; source_span: string | null; source_occurrence: number | null; relationship: string; explanation: string }
export type Highlight = { start: number; end: number; kind: 'lexical' | 'model' | 'partial' | 'unresolved' | 'reuse' }

function occurrenceRange(text: string, quote: string, occurrence: number): [number, number] | null {
  if (typeof quote !== 'string' || !quote.trim() || !Number.isInteger(occurrence) || occurrence < 0) return null
  const points = Array.from(text), needle = Array.from(quote)
  let found = 0
  for (let i = 0; i + needle.length <= points.length; i++) {
    if (needle.every((p, j) => points[i + j] === p) && found++ === occurrence) return [i, i + needle.length]
  }
  return null
}

export function dualGrounded(e: Evidence): boolean {
  const d = e.dual_assessment
  if (!d || e.evidence_assessment?.semantic_status === 'INVALID_EVIDENCE') return false
  const r = d.textual_reuse, s = d.semantic_support
  if (!r || !s || !Array.isArray(r.evidence) || !Array.isArray(s.supporting_spans) || !Array.isArray(s.unsupported_or_changed_details)) return false
  if (!Array.isArray(d.limitations) || d.limitations.some(p => typeof p !== 'string' || !p.trim()) || ![r.explanation, s.explanation].every(p => typeof p === 'string' && p.trim())) return false
  if (!['YES', 'NO', 'UNCERTAIN'].includes(r.verdict) || !['FULL', 'PARTIAL', 'NONE', 'UNCERTAIN'].includes(s.verdict)) return false
  if (![r.confidence, s.confidence].every(c => ['LOW', 'MEDIUM', 'HIGH'].includes(c))) return false
  if (r.verdict === 'YES' && (!['EXACT', 'MINOR_EDIT', 'PARAPHRASE', 'PARTIAL'].includes(r.category) || !r.evidence.length)) return false
  if (r.verdict === 'NO' && (r.category !== 'NONE' || r.evidence.length)) return false
  if (r.verdict === 'UNCERTAIN' && r.category !== 'UNCERTAIN') return false
  if (['FULL', 'PARTIAL'].includes(s.verdict) !== Boolean(s.supporting_spans.length)) return false
  if (s.verdict === 'FULL' && s.unsupported_or_changed_details.length) return false
  if ((s.verdict === 'PARTIAL' || (s.verdict === 'NONE' && r.verdict === 'YES')) && !s.unsupported_or_changed_details.length) return false
  if (![...r.evidence, ...s.supporting_spans].every(p => p && occurrenceRange(e.submission_text, p.submission_span, p.submission_occurrence) && occurrenceRange(e.source_text, p.source_span, p.source_occurrence))) return false
  if (r.category === 'EXACT' && r.evidence.some(p => p.submission_span !== p.source_span)) return false
  for (const p of s.unsupported_or_changed_details) {
    if (!p || typeof p.explanation !== 'string' || !p.explanation.trim()) return false
    const range = occurrenceRange(e.submission_text, p.submission_span, p.submission_occurrence)
    if (!range || (p.source_span === null) !== (p.source_occurrence === null)) return false
    if (!['CHANGED', 'NOT_STATED', 'UNCERTAIN'].includes(p.relationship) || (p.relationship === 'CHANGED' && p.source_span === null) || (p.relationship === 'NOT_STATED' && p.source_span !== null)) return false
    if (p.source_span !== null && !occurrenceRange(e.source_text, p.source_span, p.source_occurrence!)) return false
    if (s.supporting_spans.some(q => { const x = occurrenceRange(e.submission_text, q.submission_span, q.submission_occurrence)!; return x[0] < range[1] && range[0] < x[1] })) return false
    if (r.category === 'PARTIAL' && p.relationship === 'NOT_STATED' && r.evidence.some(q => { const x = occurrenceRange(e.submission_text, q.submission_span, q.submission_occurrence)!; return x[0] < range[1] && range[0] < x[1] })) return false
  }
  const coversWords = (pairs: DualPair[]) => [...e.submission_text.matchAll(/[\p{L}\p{N}_]+/gu)].every(word => {
    const a = Array.from(e.submission_text.slice(0, word.index)).length, b = a + Array.from(word[0]).length
    return pairs.some(p => { const x = occurrenceRange(e.submission_text, p.submission_span, p.submission_occurrence)!; return x[0] <= a && b <= x[1] })
  })
  return (s.verdict !== 'FULL' || coversWords(s.supporting_spans)) && (r.category !== 'EXACT' || coversWords(r.evidence))
}

export function dualQuotes(e: Evidence, kind: 'reuse' | 'support'): DualPair[] {
  if (!dualGrounded(e)) return []
  return kind === 'reuse' ? e.dual_assessment!.textual_reuse.evidence : e.dual_assessment!.semantic_support.supporting_spans
}

export function textualReuse(e: Evidence): string {
  if (!e.dual_assessment) return e.evidence_assessment?.lexical_status ?? 'Unclassified overlap'
  if (!dualGrounded(e)) return 'UNKNOWN — invalid or inconsistent assessment'
  const r = e.dual_assessment.textual_reuse
  return r.confidence === 'LOW' || r.verdict === 'UNCERTAIN' ? 'UNKNOWN — requires review' : `${r.verdict} / ${r.category} — model-reported possible reuse, not proof of copying`
}

export function semanticSupport(e: Evidence): string {
  if (e.dual_assessment) {
    if (!dualGrounded(e)) return 'UNKNOWN — invalid or inconsistent assessment'
    const effective = e.effective_semantic_support
    if (!effective || effective.status === 'UNKNOWN' || effective.independently_validated !== false || effective.semantic_support_coverage_permitted !== false) return 'UNKNOWN — quarantined or effective assessment unavailable'
    const s = e.dual_assessment.semantic_support
    if (s.confidence === 'LOW' || s.verdict === 'UNCERTAIN' || e.evidence_assessment?.semantic_status === 'UNKNOWN') return 'UNKNOWN — requires review'
    return `${effective.status} — model-reported support, not independently verified`
  }
  // No current path supplies independently adjudicated semantic truth. Older
  // saved reports must not acquire confirmation from a score or stale badge.
  if (!e.llm_verdict) return e.verification_method === 'deterministic_fallback' ? 'UNKNOWN — verification unavailable' : 'UNVERIFIED — lexical evidence only'
  if (e.llm_verification?.contradictions?.length) return 'UNKNOWN — model reports a conflict'
  if (e.llm_verification?.confidence === 'LOW' || e.final_match_status === 'MANUAL_REVIEW') return 'UNKNOWN — requires review'
  if (e.llm_verdict === 'PARTIAL_OVERLAP') return 'PARTIAL — model-reported, not independently verified'
  if (e.llm_verdict === 'UNRELATED') return 'UNKNOWN — model reports no specific support'
  if (e.llm_verdict === 'INSUFFICIENT_CONTEXT') return 'UNKNOWN — insufficient context'
  return 'UNVERIFIED — model-reported similarity, not independent confirmation'
}

export function codePointSlice(text: string, start: number, end: number): string {
  return Array.from(text).slice(start, end).join('')
}

function uniqueRange(text: string, quote: string): [number, number] | null {
  const points = Array.from(text), needle = Array.from(quote)
  if (!quote.trim()) return null
  const starts: number[] = []
  for (let i = 0; i + needle.length <= points.length; i++) {
    if (needle.every((p, j) => points[i + j] === p)) starts.push(i)
  }
  return starts.length === 1 ? [starts[0], starts[0] + needle.length] : null
}

export function highlights(text: string, units: Unit[], matches: Evidence[], unresolved: Evidence[] = []): Highlight[] {
  const result: Highlight[] = []
  for (const [evidence, pending] of [...matches.map(e => [e, false] as const), ...unresolved.map(e => [e, true] as const)]) {
    const unit = units.find(u => u.id === evidence.unit_id)
    if (!unit || unit.text !== evidence.submission_text || unit.start_char < 0 || unit.end_char > Array.from(text).length || codePointSlice(text, unit.start_char, unit.end_char) !== unit.text) continue
    if (evidence.dual_assessment) {
      // Reuse highlights never imply that the highlighted claim is supported.
      if (dualGrounded(evidence) && evidence.dual_assessment.textual_reuse.verdict === 'YES' && evidence.dual_assessment.textual_reuse.confidence !== 'LOW') {
        for (const p of dualQuotes(evidence, 'reuse')) {
          const range = occurrenceRange(unit.text, p.submission_span, p.submission_occurrence)!
          result.push({ start: unit.start_char + range[0], end: unit.start_char + range[1], kind: 'reuse' })
        }
      } else if (pending) result.push({ start: unit.start_char, end: unit.end_char, kind: 'unresolved' })
      continue
    }
    if (pending || !evidence.llm_verdict) {
      result.push({ start: unit.start_char, end: unit.end_char, kind: pending ? 'unresolved' : 'lexical' })
    } else {
      for (const pair of literalQuotes(evidence)) {
        const range = uniqueRange(unit.text, pair.submission_span)
        if (range) result.push({ start: unit.start_char + range[0], end: unit.start_char + range[1], kind: evidence.llm_verdict === 'PARTIAL_OVERLAP' ? 'partial' : 'model' })
      }
    }
  }
  return result
}

export function segments(text: string, ranges: Highlight[]) {
  const length = Array.from(text).length
  const safe = ranges.filter(r => Number.isInteger(r.start) && Number.isInteger(r.end) && r.start >= 0 && r.start < r.end && r.end <= length)
  const boundaries = [...new Set([0, length, ...safe.flatMap(r => [r.start, r.end])])].sort((a, b) => a - b)
  return boundaries.slice(0, -1).map((start, i) => {
    const end = boundaries[i + 1]
    const kinds = [...new Set(safe.filter(r => r.start < end && r.end > start).map(r => r.kind))]
    return { start, end, text: codePointSlice(text, start, end), kind: kinds.length > 1 ? 'mixed' : kinds[0] ?? null }
  })
}

export function safeExternalUrl(value: string): string | null {
  try {
    const url = new URL(value)
    return ['https:', 'http:'].includes(url.protocol) && !url.username && !url.password ? url.href : null
  } catch { return null }
}

export function literalQuotes(evidence: Evidence) {
  if (evidence.dual_assessment) return [] // Never project reuse into old semantic quotes.
  if (!evidence.llm_verdict) return []
  return (evidence.llm_verification?.supporting_spans ?? []).filter(pair =>
    pair.submission_span.trim() && pair.source_span.trim() &&
    evidence.submission_text.includes(pair.submission_span) && evidence.source_text.includes(pair.source_span))
}
