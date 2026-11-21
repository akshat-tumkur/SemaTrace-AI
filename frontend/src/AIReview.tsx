import { useState } from 'react'
import { safeExternalUrl } from './evidence'
import type { AIAssessment } from './productTypes'

export default function AIReview({assessment,available,onReview,onInvestigate,investigations}:{assessment?:AIAssessment;available:boolean;onReview:()=>Promise<void>;onInvestigate:(candidate:string)=>Promise<void>;investigations?:Record<string,{explanation:{warning:string;similarities?:string;differences?:string;uncertainty?:string};trace:{node:string;tool?:string;action?:string}[]}>}) {
  const [busy,setBusy]=useState(false),[error,setError]=useState(''),[visible,setVisible]=useState(8)
  const [investigating,setInvestigating]=useState('')
  const findings=assessment?.findings ?? []
  async function review(){setBusy(true);setError('');try{await onReview()}catch(e){setError(e instanceof Error?e.message:'AI review failed')}finally{setBusy(false)}}
  return <section className="ai-review" data-testid="ai-review">
    <div className="section-heading"><div><p className="eyebrow">MEANING & PARAPHRASE REVIEW</p><h3>AI-identified correspondences <span>({findings.length})</span></h3></div>
      <button className="secondary" disabled={busy || !available || (!!assessment && assessment.status==='complete' && !assessment.remaining_units)} onClick={()=>void review()}>{busy?'Reviewing source evidence…':assessment?.remaining_units?'Review next passages':assessment?.status==='complete'?'Review complete':'Review selected excerpts with OpenAI'}</button></div>
    <p>These findings compare specific meanings and literal source evidence. They are advisory, can be wrong, and do not prove copying or misconduct.</p>
    {assessment?.reviewed_unit_starts && <p className="scope-note">Reviewed {assessment.reviewed_unit_starts.length} of {assessment.eligible_units} eligible text units · {assessment.remaining_units ?? 0} remaining · {assessment.skipped_units?.length ?? 0} skipped by context bounds · {assessment.model ?? 'AI review'}{assessment.usage && <> · {assessment.usage.total_tokens.toLocaleString()} tokens in the latest review</>}</p>}
    {!available && <p className="notice">AI review needs an OpenAI key in the server's private .env. Local source discovery and matching-word analysis work independently.</p>}
    {available && !findings.length && <p className="notice">{assessment?.status==='complete'?'No supported semantic correspondences were accepted in the reviewed passages. This does not establish originality.':'AI review sends selected submission/source excerpts to OpenAI. Up to48 eligible units are reviewed per request; the entire document still receives local source retrieval.'}</p>}
    {assessment?.warnings?.map((warning,i)=><p className="notice warning" key={i}>{warning}</p>)}
    {error && <p className="error" role="alert">{error}</p>}
    {findings.slice(0,visible).map((finding,i)=><article className="ai-finding" key={finding.finding_id} data-ai-finding={finding.finding_id}>
      <div className="semantic-pair-heading"><strong>{i+1}. {finding.source_title}</strong><span className="finding-badge">{finding.relation==='PARAPHRASE'?'Potential paraphrase':'Partial meaning correspondence'}</span></div>
      <div className="evidence-card"><div><p className="evidence-label">Submitted evidence · [{finding.submission_start}, {finding.submission_end})</p><blockquote><mark className="semantic-context">{finding.submission_excerpt}</mark></blockquote></div><div><p className="evidence-label">Literal source evidence · [{finding.source_start}, {finding.source_end})</p><blockquote><mark className="semantic-context">{finding.source_excerpt}</mark></blockquote>{safeExternalUrl(finding.source_url) && <a href={safeExternalUrl(finding.source_url)!} target="_blank" rel="noopener noreferrer">Open original source ↗</a>}</div></div>
      <p><strong>AI explanation:</strong> {finding.explanation}</p>
      {!!finding.differences.length && <ul className="differences">{finding.differences.map((d,j)=><li key={j}>{d}</li>)}</ul>}
      <details><summary>Context, attribution and source identity</summary><p>Revision {finding.source_revision} · {finding.source_document_id}</p><p>{finding.source_attribution?.authors.join(', ') || 'Author not supplied'} · {finding.source_attribution?.license_name ?? 'Rights unverified / private snapshot'}</p><p>{finding.source_attribution?.attribution_text}</p>{finding.submission_context && <><p>Submitted context</p><blockquote>{finding.submission_context}</blockquote></>}{finding.source_context_excerpt && <><p>Source context</p><blockquote>{finding.source_context_excerpt}</blockquote></>}</details>
      <button className="secondary investigation-action" disabled={!available || !!investigating || !!investigations?.['openai:'+finding.candidate_id]} onClick={async()=>{setInvestigating(finding.candidate_id);setError('');try{await onInvestigate(finding.candidate_id)}catch(e){setError(e instanceof Error?e.message:'Investigation failed')}finally{setInvestigating('')}}}>{investigating===finding.candidate_id?'Investigating source context…':investigations?.['openai:'+finding.candidate_id]?'Investigation saved':'Investigate evidence with LangGraph'}</button>
      {investigations?.['openai:'+finding.candidate_id]&&<div className="investigation"><p>{investigations['openai:'+finding.candidate_id].explanation.warning}</p><p>{investigations['openai:'+finding.candidate_id].explanation.similarities}</p><p>{investigations['openai:'+finding.candidate_id].explanation.differences}</p><p>{investigations['openai:'+finding.candidate_id].explanation.uncertainty}</p><details><summary>Actual conditional tool trace</summary><ol>{investigations['openai:'+finding.candidate_id].trace.map((t,j)=><li key={j}>{t.node}: {t.tool ?? t.action}</li>)}</ol></details></div>}
    </article>)}
    {findings.length>visible && <button className="secondary" onClick={()=>setVisible(n=>n+8)}>Show more findings ({findings.length-visible} remaining)</button>}
  </section>
}
