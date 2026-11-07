// Response types for the FastAPI backend. Keep in sync with the Pydantic models:
//   backend/app/main.py (health), backend/app/routers/market.py (history, quotes, auth),
//   backend/app/routers/forecast.py and backend/app/models/tft_model.py (forecast).

export interface HealthResponse {
  status: 'ok' | 'unauthenticated' | 'error'
  authenticated: boolean
  profile: Record<string, unknown> | null
  error: string | null
  timestamp: number
}

export interface LoginUrlResponse {
  login_url: string
}

export interface AuthExchangeResponse {
  status: string
  authenticated: boolean
  profile: Record<string, unknown> | null
  message: string
}

/** Each candle is [epoch_seconds, open, high, low, close, volume, open_interest]. */
export type Candle = [number, number, number, number, number, number, number]

export interface HistoryResponse {
  status: string
  symbol: string
  resolution: string
  count: number
  candles: Candle[]
}

export interface QuoteData {
  symbol: string
  ltp: number
  change?: number
  change_pct?: number
  volume?: number
  depth_imbalance?: number
  micro_price?: number
}

export interface QuotesResponse {
  status: string
  count: number
  data: QuoteData[]
}

export interface HorizonForecast {
  p10: number
  p50: number
  p90: number
  p10_return: number
  p50_return: number
  p90_return: number
  directional_bias: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
}

export type ForecastStatus = 'READY' | 'BUFFERING'

export interface ForecastResponse {
  status: ForecastStatus
  symbol: string | null
  timestamp: number | null
  current_price: number | null
  p10: number | null
  p50: number | null
  p90: number | null
  directional_bias: 'BULLISH' | 'BEARISH' | 'NEUTRAL' | null
  horizons: Record<string, HorizonForecast> | null
  buffer_count: number | null
  required_bars: number | null
}
