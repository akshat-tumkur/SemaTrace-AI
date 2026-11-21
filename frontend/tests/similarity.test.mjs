import { test } from 'node:test'
import assert from 'node:assert/strict'
import { addDocumentExclusion, selectedSourceSegments, sourceEvidence, sourceExcerpt } from '../src/similarity.ts'

test('source selection highlights only supplied Unicode intervals and preserves original text', () => {
  const text = '🌍 alpha beta alpha beta'
  const source = { intervals: [{ start: 13, end: 23 }] }
  const parts = selectedSourceSegments(text, source)
  assert.equal(parts.map(part => part.text).join(''), text)
  assert.deepEqual(parts.filter(part => part.kind).map(part => part.text), ['alpha beta'])
  assert.equal(selectedSourceSegments(text).filter(part => part.kind).length, 0)
})

test('overlapping source intervals render once and invalid ranges are ignored', () => {
  const parts = selectedSourceSegments('abcdef', { intervals: [{ start: 0, end: 4 }, { start: 2, end: 6 }, { start: -1, end: 20 }] })
  assert.equal(parts.map(part => part.text).join(''), 'abcdef')
  assert.equal(parts.filter(part => part.kind).reduce((sum, part) => sum + part.text.length, 0), 6)
})

test('source excerpts preserve original occurrence identity and deduplicate repeated candidates', () => {
  const evidence = { source_text: 'Original source.', source_url: 'https://example.org', source_document_id: 'doc-a' }
  const alias = { source_text: 'Another original.', source_url: 'https://example.org/b', source_occurrences: [{ source_document_id: 'doc-b' }] }
  assert.deepEqual(sourceEvidence([evidence, evidence, alias], 'doc-a'), [evidence])
  assert.deepEqual(sourceEvidence([evidence, alias], 'doc-b'), [alias])
  assert.deepEqual(sourceEvidence([evidence, alias], 'missing'), [])
})

test('explicit exclusions require nonempty in-bounds code-point ranges', () => {
  const initial = [{ start: 0, end: 1, reason: 'Existing' }]
  assert.deepEqual(addDocumentExclusion(initial, 2, 5, 6), [...initial, { start: 2, end: 5, reason: 'USER_SELECTED_TEXT' }])
  assert.equal(initial.length, 1)
  for (const [start, end] of [[-1, 3], [3, 3], [4, 7], [1.2, 3], [NaN, 2]]) assert.throws(() => addDocumentExclusion([], start, end, 6))
})

test('selected alias excerpt uses that original occurrence, not the canonical source coordinates', () => {
  const evidence = { source_text: 'Identical passage.', source_title: 'Canonical', source_document_id: 'doc-a', source_start_char: 10, source_end_char: 28,
    source_occurrences: [{ title: 'Original B', source_document_id: 'doc-b', source_start_char: 400, source_end_char: 418 }] }
  assert.deepEqual(sourceExcerpt(evidence, 'doc-b'), { title: 'Original B', text: 'Identical passage.', document_id: 'doc-b', start: 400, end: 418 })
})

test('an excluded source shows raw matching intervals for inspection without restoring contribution', () => {
  const source = { excluded: true, intervals: [], raw_intervals: [{ start: 0, end: 3 }], attributed_matched_words: 0 }
  assert.deepEqual(selectedSourceSegments('abc def', source).filter(part => part.kind).map(part => part.text), ['abc'])
  assert.equal(source.attributed_matched_words, 0)
})
