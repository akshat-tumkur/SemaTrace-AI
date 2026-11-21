import { codePointSlice, type Highlight } from './evidence.ts'
export function semanticRanges(text: string, finding: { category: string; matched_passages: Array<{submission_start:number;submission_end:number;submission_excerpt:string}> }): Highlight[] {
  if (finding.category === 'UNCERTAIN') return []
  return finding.matched_passages.filter(p => Number.isInteger(p.submission_start) && Number.isInteger(p.submission_end) && p.submission_start>=0 && p.submission_start<p.submission_end && p.submission_end<=Array.from(text).length && codePointSlice(text,p.submission_start,p.submission_end)===p.submission_excerpt).map(p=>({start:p.submission_start,end:p.submission_end,kind:'model'}))
}
