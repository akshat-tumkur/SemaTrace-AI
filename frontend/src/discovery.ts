/** Ordinal retrieval summaries only. These counts never measure reused text. */
export type RankedContext = {
  candidate_id: string; submission_start: number; source_document_id: string
  source_title: string; source_url: string; source_revision: string; score: number; minilm_rank: number
}
export function summarizeDiscovery<T extends RankedContext>(rows: T[]) {
  const best = new Map<number, T>()
  for (const row of rows) {
    if (!row.candidate_id || !Number.isFinite(row.score)) continue
    const previous = best.get(row.submission_start)
    if (!previous || row.minilm_rank < previous.minilm_rank || (row.minilm_rank === previous.minilm_rank && row.score > previous.score)) best.set(row.submission_start, row)
  }
  const first = [...best.values()]
  const sources = new Map<string, {id: string; title: string; url: string; revision: string; firstRankedUnits: number; best: T}>()
  for (const row of first) {
    const source = sources.get(row.source_document_id)
    if (!source) sources.set(row.source_document_id, {id:row.source_document_id, title:row.source_title, url:row.source_url, revision:row.source_revision, firstRankedUnits:1, best:row})
    else {source.firstRankedUnits++; if (row.score > source.best.score) source.best = row}
  }
  const topSources = [...sources.values()].sort((a,b) => b.firstRankedUnits-a.firstRankedUnits || b.best.score-a.best.score || a.id.localeCompare(b.id))
  const previews = first.sort((a,b) => b.score-a.score || a.submission_start-b.submission_start).slice(0,3)
  return {unitCount:best.size, topSources, previews}
}
