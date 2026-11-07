import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, getForecast, getHistory, getQuotes } from '@/services/api'

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('api service', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('history never requests a broker sync by default', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { status: 'ok', candles: [] }))
    vi.stubGlobal('fetch', fetchMock)

    await getHistory('NSE:SBIN-EQ', { resolution: '1' })

    const url = String(fetchMock.mock.calls[0][0])
    expect(url).toContain('/history/NSE%3ASBIN-EQ')
    expect(url).toContain('sync=false')
    expect(url).toContain('resolution=1')
  })

  it('quotes encodes the symbol list', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { status: 'ok', count: 0, data: [] }))
    vi.stubGlobal('fetch', fetchMock)

    await getQuotes(['NSE:SBIN-EQ', 'NSE:TCS-EQ'])

    expect(String(fetchMock.mock.calls[0][0])).toContain('symbols=NSE%3ASBIN-EQ%2CNSE%3ATCS-EQ')
  })

  it('maps backend error detail to ApiError', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(502, { detail: 'token expired' })))

    const err = await getForecast('NSE:SBIN-EQ').catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).status).toBe(502)
    expect((err as ApiError).message).toBe('token expired')
  })

  it('falls back to the status text when the body is not JSON', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('oops', { status: 500, statusText: 'Server Error' })))

    const err = await getForecast('NSE:SBIN-EQ').catch((e: unknown) => e)
    expect((err as ApiError).message).toBe('Server Error')
  })
})
