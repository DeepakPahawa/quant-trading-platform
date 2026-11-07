import type {
  AuthExchangeResponse,
  ForecastResponse,
  HealthResponse,
  HistoryResponse,
  LoginUrlResponse,
  QuotesResponse,
} from '@/types/api'

/** Backend base URL. Override with VITE_API_BASE_URL in frontend/.env.local. */
export const API_BASE: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://127.0.0.1:8000/api'

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
  })

  if (!res.ok) {
    let detail = res.statusText || `HTTP ${res.status}`
    try {
      const body = (await res.json()) as { detail?: unknown }
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // Body was not JSON; keep the status text.
    }
    throw new ApiError(res.status, detail)
  }
  return (await res.json()) as T
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

export function getLoginUrl(): Promise<LoginUrlResponse> {
  return request<LoginUrlResponse>('/auth/login-url')
}

/** Exchanges the FYERS auth code (or the full redirected URL). The code is never stored. */
export function exchangeAuthCode(authCodeOrUrl: string): Promise<AuthExchangeResponse> {
  return request<AuthExchangeResponse>('/auth/exchange', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ auth_code_or_url: authCodeOrUrl }),
  })
}

/** Live quotes for several symbols. Requires an authenticated FYERS session. */
export function getQuotes(symbols: readonly string[]): Promise<QuotesResponse> {
  const query = encodeURIComponent(symbols.join(','))
  return request<QuotesResponse>(`/market/quotes?symbols=${query}`)
}

export interface HistoryOptions {
  resolution?: string
  rangeFrom?: string
  rangeTo?: string
  /** Default false: page loads read the local archive only and never hit the broker. */
  sync?: boolean
}

export function getHistory(symbol: string, options: HistoryOptions = {}): Promise<HistoryResponse> {
  const params = new URLSearchParams({
    resolution: options.resolution ?? '1',
    sync: String(options.sync ?? false),
  })
  if (options.rangeFrom) params.set('range_from', options.rangeFrom)
  if (options.rangeTo) params.set('range_to', options.rangeTo)
  return request<HistoryResponse>(`/history/${encodeURIComponent(symbol)}?${params.toString()}`)
}

/** TFT quantile forecast from archived bars. Returns BUFFERING until enough bars exist. */
export function getForecast(symbol: string): Promise<ForecastResponse> {
  return request<ForecastResponse>(`/forecast/${encodeURIComponent(symbol)}`)
}
