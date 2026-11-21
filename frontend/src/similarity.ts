import { segments, type Evidence } from './evidence.ts'

export type SimilarityRange = { start: number; end: number; reason: string }
export type SimilarityOptions = { minimum_match_words: number; excluded_source_ids: string[]; excluded_ranges: SimilarityRange[] }
export type SimilaritySource = {
  source_id: string
  name: string
  url: string
  raw_matched_words: number
  attributed_matched_words: number
  attributed_percent: number
  intervals: Array<{ start: number; end: number }>
  raw_intervals?: Array<{ start: number; end: number }>
  alternative_source_ids: string[]
  excluded?: boolean
}
export type SimilarityReportData = {
  method: string
  document_id: string
  total_eligible_words: number
  unique_matched_words: number
  textual_similarity_percent: number
  sources: SimilaritySource[]
  ledger: Array<Record<string, unknown>>
  exclusions: { excluded_document_words: number; excluded_source_ids: string[]; document_ranges: SimilarityRange[] }
  unresolved_findings: Array<{ candidate_id?: string; category?: string; reason: string }>
  methodology: string | Record<string, unknown>
  limitations: string[]
  options: SimilarityOptions
}

export function selectedSourceSegments(text: string, source?: SimilaritySource) {
  const ranges = source?.excluded ? source.raw_intervals ?? [] : source?.intervals ?? []
  return segments(text, ranges.map(range => ({ ...range, kind: 'lexical' as const })))
}

export function sourceEvidence(evidence: Evidence[], sourceId: string): Evidence[] {
  return evidence.filter(item => {
    const value = item as Evidence & { source_id?: string }
    return value.source_id === sourceId || item.source_document_id === sourceId ||
      item.source_occurrences?.some(origin => origin.source_document_id === sourceId)
  }).filter((item, index, values) => values.findIndex(other => other.source_text === item.source_text && other.source_url === item.source_url) === index)
}

export function sourceExcerpt(evidence: Evidence, sourceId: string) {
  const occurrence = evidence.source_occurrences?.find(origin => origin.source_document_id === sourceId)
  return {
    title: occurrence?.title ?? evidence.source_title,
    text: evidence.source_text,
    document_id: occurrence?.source_document_id ?? evidence.source_document_id,
    start: occurrence?.source_start_char ?? evidence.source_start_char,
    end: occurrence?.source_end_char ?? evidence.source_end_char,
  }
}

export function addDocumentExclusion(ranges: SimilarityRange[], start: number, end: number, textLength: number): SimilarityRange[] {
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || start >= end || end > textLength) {
    throw new Error('Use a nonempty character range within the normalized document.')
  }
  return [...ranges, { start, end, reason: 'USER_SELECTED_TEXT' }]
}
