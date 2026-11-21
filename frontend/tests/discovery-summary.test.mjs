import { test } from 'node:test'
import assert from 'node:assert/strict'
import { summarizeDiscovery } from '../src/discovery.ts'
const row = (unit, source, rank, score, id=`${unit}-${source}-${rank}`) => ({candidate_id:id, submission_start:unit*100, source_document_id:source, source_title:source, source_url:'https://example.org/'+source, source_revision:'fixed', score, minilm_rank:rank})
test('retrieval summary counts distinct first-ranked units, not eight candidates as eight matches', () => {
  const rows=[row(0,'Water cycle',1,.7),row(0,'Water cycle',2,.65),row(0,'Wetland',3,.6),row(1,'Water cycle',1,.8),row(2,'Wetland',1,.75)]
  const summary=summarizeDiscovery([...rows, rows[0]])
  assert.equal(summary.unitCount,3)
  assert.equal(summary.topSources[0].title,'Water cycle')
  assert.equal(summary.topSources[0].firstRankedUnits,2)
  assert.deepEqual(summary.previews.map(r => r.submission_start),[100,200,0])
  assert.equal(rows[0].submission_start,0)
  assert.equal('semantic_coverage_percent' in summary,false)
})
test('unrelated queries still have candidates; summary never creates a reuse verdict or percentage', () => {
  const summary=summarizeDiscovery([row(0,'Unrelated source',1,-.1)])
  assert.equal(summary.topSources[0].title,'Unrelated source')
  assert.equal(summary.previews.length,1)
  assert.equal('confirmed_reuse' in summary,false)
  assert.deepEqual(summarizeDiscovery([]),{unitCount:0,topSources:[],previews:[]})
})
