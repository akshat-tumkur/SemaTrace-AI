import type { PassageFinding, RelatedSource } from './PassageFindings'
import type { DiscoveryCandidate, SavedSemanticReview } from './SemanticDiscovery'
import type { SimilarityReportData } from './similarity'
import type { Evidence, Unit } from './evidence'
export type AIFinding = DiscoveryCandidate & {finding_id:string;relation:string;explanation:string;differences:string[];warning:string;submission_context?:string;source_context_excerpt?:string}
export type AIAssessment = {status:string;findings:AIFinding[];eligible_units?:number;reviewed_unit_starts?:number[];remaining_units?:number;skipped_units?:unknown[];warnings?:string[];model?:string;usage?:{total_tokens:number};note?:string}
export type Analysis = {
  analysis_id:string;filename:string;status:string;created_at?:string;report_version?:string;summary:string
  findings:PassageFinding[];related_sources:(RelatedSource & DiscoveryCandidate)[];semantic_review?:SavedSemanticReview;semantic_assessment?:AIAssessment
  document?:{text:string;note:string};similarity_report?:SimilarityReportData;matches:Evidence[];units?:Unit[];units_analyzed?:number
  verification?:{candidates:Evidence[];unresolved_candidate_ids:string[]};audit_log:{agent:string;event:string;details:string}[]
  corpus?:{source_origins:number;unique_passages:number;dense_status:string};retrieval?:{retained_sources:string[];dropped_sources:string[]}
  processing?:{elapsed_seconds:number};web_discovery?:{status:string;sources:unknown[];queries:string[];warnings:string[]}
}
export type ProductStatus = {cloud_review_available:boolean;web_search_available:boolean;review_model:string;source_uploads_enabled:boolean;privacy:string;limits?:{ai_units_per_review:number}}
export type Job = {job_id:string;status:string;stage:string;progress:number;analysis_id:string|null;error:string|null;filename:string}
