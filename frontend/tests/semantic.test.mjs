import {test} from 'node:test'
import assert from 'node:assert/strict'
import {semanticRanges} from '../src/semanticRanges.ts'
test('semantic quote highlights preserve Unicode offsets and do not extend unsupported clauses',()=>{
 const text='🌍 Shared idea; unsupported tail.'
 const f={category:'PARTIAL_REUSE',matched_passages:[{submission_start:2,submission_end:13,submission_excerpt:'Shared idea'}]}
 assert.deepEqual(semanticRanges(text,f),[{start:2,end:13,kind:'model'}])
})
test('context-only uncertainty and invented quotes never gain semantic highlights',()=>{
 const f={category:'UNCERTAIN',matched_passages:[{submission_start:0,submission_end:4,submission_excerpt:'Text'}]}
 assert.deepEqual(semanticRanges('Text',f),[])
 assert.deepEqual(semanticRanges('Text',{...f,category:'PARAPHRASE',matched_passages:[{submission_start:0,submission_end:4,submission_excerpt:'fake'}]}),[])
})
