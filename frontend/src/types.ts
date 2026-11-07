// ─── WebSocket message types from /ws/market ────────────────────────────────

export interface WsConnectionAck {
  type: 'connection_ack'
  status: string
  channel: string
}

export interface WsTick {
  type: 'tick'
  symbol: string
  timestamp: number
  data: Record<string, number | string>
}

export interface WsBar {
  type: 'bar'
  symbol: string
  timestamp: number
  data: {
    timestamp: number
    open: number
    high: number
    low: number
    close: number
    volume: number
    vwap?: number
    vwap_upper?: number
    vwap_lower?: number
  }
}

export interface WsDepthMetrics {
  symbol: string | null
  timestamp: number | null
  best_bid: number
  best_ask: number
  mid_price: number
  depth_imbalance: number   // [-1, 1]
  micro_price: number
  normalized_spread: number
  total_bid_qty: number
  total_ask_qty: number
  bid_qty_1: number
  ask_qty_1: number
}

export interface DepthLevel {
  price: number
  quantity: number
  orders: number | null
}

export interface WsDepth {
  type: 'depth'
  symbol: string
  timestamp: number
  metrics: WsDepthMetrics
  data: {
    bids?: DepthLevel[]
    asks?: DepthLevel[]
    bid_price1?: number
    bid_size1?: number
    ask_price1?: number
    ask_size1?: number
    bid_price2?: number
    bid_size2?: number
    ask_price2?: number
    ask_size2?: number
    bid_price3?: number
    bid_size3?: number
    ask_price3?: number
    ask_size3?: number
    bid_price4?: number
    bid_size4?: number
    ask_price4?: number
    ask_size4?: number
    bid_price5?: number
    bid_size5?: number
    ask_price5?: number
    ask_size5?: number
  }
}

export interface WsTftForecast {
  type: 'tft_forecast'
  symbol: string
  timestamp: number
  horizon: number
  predictions: Array<{
    t: number       // epoch seconds (future)
    p10: number
    p50: number
    p90: number
  }>
}

export type WsMessage = WsConnectionAck | WsTick | WsBar | WsDepth | WsTftForecast

// ─── REST response types ─────────────────────────────────────────────────────

export interface OptionQuote {
  strike: number
  option_type: 'CE' | 'PE'
  bid: number
  ask: number
  ltp: number
  iv: number
  delta: number
  gamma: number
  theta: number
  vega: number
  oi: number
  volume: number
}

export interface SviCurvePoint {
  strike: number
  log_moneyness: number
  iv: number
  model_iv: number
}

export interface OptionsChainResponse {
  status: string
  symbol: string
  data_source: string
  spot: number
  forward: number
  time_to_expiry_days: number
  time_to_expiry_years: number
  risk_free_rate: number
  svi_parameters: {
    a: number
    b: number
    rho: number
    m: number
    sigma: number
  }
  curve_points: SviCurvePoint[]
  quotes: OptionQuote[]
}

export interface BacktestRunRequest {
  symbol: string
  initial_capital: number
  stop_loss_pct: number | null
  take_profit_pct: number | null
  instrument_type: string
  strategy: string
  default_quantity: number
}

export interface EquityCurvePoint {
  timestamp: number
  equity: number
  drawdown: number
}

export interface TotalDeductionsBreakdown {
  stt: number
  exchange_charges: number
  sebi_charges: number
  stamp_duty: number
  gst: number
  brokerage: number
  total: number
}

export interface BacktestRunResponse {
  status: string
  symbol: string
  initial_capital: number
  final_capital: number
  gross_profit: number
  net_profit: number
  net_cagr: number
  sharpe_ratio: number
  sortino_ratio: number
  max_drawdown: number
  profit_factor: number
  total_trades: number
  winning_trades: number
  losing_trades: number
  win_rate: number
  deductions_breakdown: TotalDeductionsBreakdown
  trades: Array<Record<string, number | string>>
  equity_curve: EquityCurvePoint[]
}

