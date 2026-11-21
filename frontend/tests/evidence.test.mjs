import { test } from 'node:test'
import assert from 'node:assert/strict'
import { safeExternalUrl, literalQuotes, semanticSupport, textualReuse, dualQuotes, dualGrounded, highlights, segments } from '../src/evidence.ts'

test('only HTTP(S) source URLs become links', () => {
  for (const value of ['javascript:alert(1)', 'data:text/html,test', 'local://corpus/demo.txt', 'not a URL', 'https://user:pass@example.org']) assert.equal(safeExternalUrl(value), null)
  assert.equal(safeExternalUrl('https://example.org/source'), 'https://example.org/source')
})

test('strong lexical role reversals and model quotes never display VERIFIED support', () => {
  const lexical = { verification_method: 'deterministic', llm_verdict: null }
  assert.match(semanticSupport(lexical), /^UNVERIFIED/)
  assert.match(semanticSupport({ ...lexical, verification_method: 'deterministic_fallback' }), /^UNKNOWN/)
  assert.match(semanticSupport({ llm_verdict: 'POSSIBLE_PARAPHRASE', llm_verification: { confidence: 'HIGH' } }), /^UNVERIFIED/)
})

test('a reported partial match with conflicts does not display unqualified partial support', () => {
  assert.match(semanticSupport({ llm_verdict: 'PARTIAL_OVERLAP', final_match_status: 'PARTIAL_MATCH', llm_verification: { confidence: 'HIGH', contradictions: ['A role differs.'] } }), /^UNKNOWN/)
})

test('Unicode original-unit offsets distinguish repeated sentences', () => {
  const text = '🌍 Repeat. Repeat.'
  const units = [{ id: 'u2', text: 'Repeat.', start_char: 10, end_char: 17 }]
  const match = { unit_id: 'u2', submission_text: 'Repeat.' }
  const ranges = highlights(text, units, [match])
  assert.deepEqual(ranges, [{ start: 10, end: 17, kind: 'lexical' }])
  const parts = segments(text, ranges)
  assert.equal(parts.map(p => p.text).join(''), text)
  assert.equal(parts.find(p => p.kind).text, 'Repeat.')
})

test('partial highlights never extend a quote and ambiguous repeats remain unassigned', () => {
  const text = 'Claim one; claim two.'
  const unit = { id: 'u1', text, start_char: 0, end_char: 21 }
  const match = { unit_id: 'u1', submission_text: text, source_text: 'Claim one', llm_verdict: 'PARTIAL_OVERLAP', llm_verification: { supporting_spans: [{ submission_span: 'Claim one', source_span: 'Claim one' }] } }
  assert.deepEqual(highlights(text, [unit], [match]), [{ start: 0, end: 9, kind: 'partial' }])
  const repeated = { ...unit, text: 'Claim one Claim one', end_char: 19 }
  assert.deepEqual(highlights(repeated.text, [repeated], [{ ...match, submission_text: repeated.text }]), [])
})

test('overlaps render once, preserve all text and keep unresolved visually distinct', () => {
  const parts = segments('abcdef', [{ start: 0, end: 4, kind: 'lexical' }, { start: 2, end: 6, kind: 'unresolved' }, { start: -1, end: 99, kind: 'partial' }])
  assert.deepEqual(parts.map(p => p.text), ['ab', 'cd', 'ef'])
  assert.deepEqual(parts.map(p => p.kind), ['lexical', 'mixed', 'unresolved'])
})

test('displayed semantic quotes must be literal on both sides and current', () => {
  const evidence = { submission_text: 'A copied sentence.', source_text: 'A copied sentence.', llm_verdict: 'POSSIBLE_PARAPHRASE', llm_verification: { supporting_spans: [
    { submission_span: 'copied sentence', source_span: 'copied sentence' },
    { submission_span: 'imagined text', source_span: 'copied sentence' },
  ] } }
  assert.equal(literalQuotes(evidence).length, 1)
  assert.equal(literalQuotes({ ...evidence, llm_verdict: null }).length, 0)
})

function dualEvidence() {
  return { unit_id: 'u1', submission_text: 'The launch was in 2016.', source_text: 'The launch was in 2015.',
    dual_assessment: {
      textual_reuse: { verdict: 'YES', category: 'MINOR_EDIT', confidence: 'HIGH', evidence: [{ submission_span: 'The launch was in 2016.', source_span: 'The launch was in 2015.', submission_occurrence: 0, source_occurrence: 0 }], explanation: 'Wording reused.' },
      semantic_support: { verdict: 'NONE', confidence: 'HIGH', supporting_spans: [], unsupported_or_changed_details: [{ submission_span: '2016', source_span: '2015', submission_occurrence: 0, source_occurrence: 0, relationship: 'CHANGED', explanation: 'Year changed.' }], explanation: 'The source gives a different year.' }, limitations: []
    }
  }
}

test('dual reuse YES with support NONE stays distinct, with no legacy semantic projection', () => {
  const e = dualEvidence()
  assert.equal(dualGrounded(e), true)
  assert.match(textualReuse(e), /^YES \/ MINOR_EDIT/)
  assert.match(semanticSupport(e), /^UNKNOWN/) // Historical dual record needs an effective projection.
  assert.equal(dualQuotes(e, 'reuse').length, 1)
  assert.deepEqual(dualQuotes(e, 'support'), [])
  assert.deepEqual(literalQuotes({ ...e, llm_verdict: 'POSSIBLE_PARAPHRASE' }), [])
  assert.deepEqual(highlights(e.submission_text, [{ id: 'u1', text: e.submission_text, start_char: 0, end_char: 23 }], [e]), [{ start: 0, end: 23, kind: 'reuse' }])
})

test('dual invalid quotes, categories and occurrences display UNKNOWN with no positive quotes', () => {
  for (const change of ['quote', 'occurrence', 'category', 'support']) {
    const e = dualEvidence()
    if (change === 'quote') e.dual_assessment.textual_reuse.evidence[0].source_span = 'An invented quotation.'
    if (change === 'occurrence') e.dual_assessment.textual_reuse.evidence[0].source_occurrence = 1
    if (change === 'category') e.dual_assessment.textual_reuse.category = 'NONE'
    if (change === 'support') e.dual_assessment.semantic_support.verdict = 'FULL'
    assert.equal(dualGrounded(e), false)
    assert.match(textualReuse(e), /^UNKNOWN/)
    assert.match(semanticSupport(e), /^UNKNOWN/)
    assert.deepEqual(dualQuotes(e, 'reuse'), [])
  }
})

test('explicit Unicode quote occurrence highlights only the selected repeated text', () => {
  const e = dualEvidence(), text = '🌍 café; café.'
  e.submission_text = e.source_text = text
  e.dual_assessment.textual_reuse.category = 'PARTIAL'
  e.dual_assessment.textual_reuse.evidence = [{ submission_span: 'café', source_span: 'café', submission_occurrence: 1, source_occurrence: 1 }]
  e.dual_assessment.semantic_support = { verdict: 'UNCERTAIN', confidence: 'LOW', supporting_spans: [], unsupported_or_changed_details: [], explanation: 'Needs context.' }
  assert.deepEqual(highlights(text, [{ id: 'u1', text, start_char: 0, end_char: 13 }], [e]), [{ start: 8, end: 12, kind: 'reuse' }])
  assert.match(semanticSupport(e), /^UNKNOWN/)
})

test('partial support cannot cover a changed clause or silently drop qualifiers', () => {
  const e = dualEvidence()
  e.dual_assessment.semantic_support.verdict = 'PARTIAL'
  e.dual_assessment.semantic_support.supporting_spans = [...e.dual_assessment.textual_reuse.evidence]
  assert.equal(dualGrounded(e), false)
  e.dual_assessment.semantic_support = { verdict: 'FULL', confidence: 'HIGH', supporting_spans: [{ submission_span: 'The launch', source_span: 'The launch', submission_occurrence: 0, source_occurrence: 0 }], unsupported_or_changed_details: [], explanation: 'Incomplete quotation.' }
  assert.equal(dualGrounded(e), false)
})

test('full support remains model-reported, and never independently verified', () => {
  const e = dualEvidence()
  e.dual_assessment.semantic_support = { verdict: 'FULL', confidence: 'HIGH', supporting_spans: [...e.dual_assessment.textual_reuse.evidence], unsupported_or_changed_details: [], explanation: 'A model claim, not human adjudication.' }
  assert.match(semanticSupport(e), /^UNKNOWN/)
  e.effective_semantic_support = { status: 'FULL', basis: 'MODEL_REPORTED_LITERAL_GROUNDED', reason_codes: [], independently_validated: false, semantic_support_coverage_permitted: false, raw_verdict: 'FULL' }
  assert.match(semanticSupport(e), /^FULL.*not independently verified/)
})

test('raw FULL cannot override effective quarantine and reuse remains visible', () => {
  const e = dualEvidence()
  e.dual_assessment.semantic_support = { verdict: 'FULL', confidence: 'HIGH', supporting_spans: [...e.dual_assessment.textual_reuse.evidence], unsupported_or_changed_details: [], explanation: 'Wrong FULL.' }
  e.effective_semantic_support = { status: 'UNKNOWN', basis: 'QUARANTINED_MODEL_REPORT', reason_codes: ['NUMERIC_DETAIL_MISMATCH'], independently_validated: false, semantic_support_coverage_permitted: false, raw_verdict: 'FULL' }
  assert.match(semanticSupport(e), /^UNKNOWN/)
  assert.match(textualReuse(e), /^YES/)
  assert.equal(dualQuotes(e, 'reuse').length, 1)
})

test('malformed saved dual payloads degrade to UNKNOWN without a rendering exception', () => {
  for (const corrupt of [e => { delete e.dual_assessment.limitations }, e => { e.dual_assessment.textual_reuse.evidence = [null] }, e => { e.dual_assessment.textual_reuse.evidence[0].source_span = 123 }, e => { e.dual_assessment.semantic_support.unsupported_or_changed_details = [null] }]) {
    const e = dualEvidence(); corrupt(e)
    assert.equal(dualGrounded(e), false)
    assert.match(semanticSupport(e), /^UNKNOWN/)
    assert.deepEqual(dualQuotes(e, 'reuse'), [])
  }
})
