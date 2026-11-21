/** Read an API response without hiding HTTP/proxy failures behind JSON errors. */
export async function readJsonResponse(response: Response) {
  const body = await response.text()
  let payload
  if (body.trim()) {
    try { payload = JSON.parse(body) }
    catch {
      throw new Error(`The API returned a non-JSON response (HTTP ${response.status}). Check that the backend is running and the frontend API address or proxy points to its port.`)
    }
  }
  if (!response.ok) {
    const detail = typeof payload?.detail === 'string' ? payload.detail : null
    throw new Error(detail ?? `API request failed (HTTP ${response.status})${body.trim() ? '.' : ' with an empty response.'} Check that the backend is running and the frontend API address or proxy points to its port.`)
  }
  if (payload === undefined || payload === null) {
    throw new Error(`The API returned an empty response (HTTP ${response.status}). Check the backend and frontend API address or proxy; no successful result was received.`)
  }
  return payload
}
