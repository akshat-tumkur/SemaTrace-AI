import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readJsonResponse } from '../src/api.ts'

test('empty Vite proxy 502 identifies HTTP failure and backend configuration', async () => {
  await assert.rejects(readJsonResponse(new Response('', {status:502})), /HTTP 502.*empty response.*backend.*proxy/)
})
test('non-JSON server and unexpected successful empty responses remain failures', async () => {
  await assert.rejects(readJsonResponse(new Response('<html>Bad gateway</html>', {status:502})), /non-JSON.*HTTP 502/)
  await assert.rejects(readJsonResponse(new Response('', {status:202})), /empty response.*HTTP 202/)
  await assert.rejects(readJsonResponse(new Response(null, {status:204})), /empty response.*HTTP 204/)
})
test('API JSON errors retain useful detail and valid report JSON is unchanged', async () => {
  await assert.rejects(readJsonResponse(Response.json({detail:'Optional model unavailable'}, {status:503})), /Optional model unavailable/)
  const saved={analysis_id:'example',related_sources:[],semantic_review:{}}
  assert.deepEqual(await readJsonResponse(Response.json(saved)), saved)
  assert.deepEqual(await readJsonResponse(Response.json([])), [])
})
