/**
 * BacktestPanel.tsx
 *
 * - Shows data-availability status for the selected symbol.
 * - "Sync Data" button calls GET /api/history/{symbol}?sync=true to ingest from FYERS.
 * - Parameter form: SL %, TP %, initial capital, strategy, instrument, date range.
 * - POST /api/backtest/run on submit.
 * - Renders:
 *     • Metric cards: Net CAGR, Sharpe, Max Drawdown, Profit Factor, Win Rate, Taxes Paid
 *     • Equity curve rendered as SVG polyline
 *     • Trade stats bar (total / winning / losing)
 */

import { useState, useCallback, useEffect } from 'react'
import type { BacktestRunResponse, EquityCurvePoint } from '../types'

// ─── API helpers ──────────────────────────────────────────────────────────────

interface HistoryCheckResponse {
  status: string
  count: number
  candles: number[][]
}

interface AvailableSymbolsResponse {
  status: string
  symbols: string[]
  count: number
}

async function checkSymbolData(symbol: string): Promise<number> {
  /** Returns number of locally-stored bars (0 = none). */
  const res = await fetch(
    `/api/history/${encodeURIComponent(symbol)}?resolution=1`,
  )
  if (!res.ok) return 0
  const data = (await res.json()) as HistoryCheckResponse
  return data.count ?? 0
}

async function fetchAvailableSymbols(): Promise<string[]> {
  const res = await fetch('/api/symbols')
  if (!res.ok) return []
  const data = (await res.json()) as AvailableSymbolsResponse
  return data.symbols ?? []
}

async function syncHistoricalData(
  symbol: string,
  rangeFrom: string,
  rangeTo: string,
): Promise<number> {
  /** Triggers a live fetch from FYERS and stores to DuckDB. Returns bar count. */
  const params = new URLSearchParams({
    resolution: '1',
    sync: 'true',
    ...(rangeFrom ? { range_from: rangeFrom } : {}),
    ...(rangeTo ? { range_to: rangeTo } : {}),
  })
  const res = await fetch(
    `/api/history/${encodeURIComponent(symbol)}?${params.toString()}`,
  )
  if (!res.ok) {
    const err = (await res.json().catch(() => ({ detail: res.statusText }))) as { detail?: string }
    throw new Error(err.detail ?? res.statusText)
  }
  const data = (await res.json()) as HistoryCheckResponse
  return data.count ?? 0
}

interface RunParams {
  symbol: string
  initial_capital: number
  stop_loss_pct: number | null
  take_profit_pct: number | null
  instrument_type: string
  strategy: string
  default_quantity: number
  start_date: string | null
  end_date: string | null
}

async function runBacktest(params: RunParams): Promise<BacktestRunResponse> {
  const res = await fetch('/api/backtest/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!res.ok) {
    const err = (await res.json().catch(() => ({ detail: res.statusText }))) as { detail?: string }
    throw new Error(err.detail ?? res.statusText)
  }
  return res.json() as Promise<BacktestRunResponse>
}

// ─── Equity Curve SVG ─────────────────────────────────────────────────────────

function EquityCurve({ points }: { points: EquityCurvePoint[] }): React.JSX.Element {
  if (points.length < 2) {
    return (
      <svg style={{ width: '100%', height: 100 }}>
        <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle"
          fontSize={10} fill="#334155">No equity data</text>
      </svg>
    )
  }

  const PAD = { l: 44, r: 8, t: 8, b: 20 }
  const W = 540; const H = 110
  const equities = points.map(p => p.equity)
  const minE = Math.min(...equities)
  const maxE = Math.max(...equities)
  const range = maxE - minE || 1

  const ptStr = points
    .map((p, i) => {
      const x = PAD.l + (i / (points.length - 1)) * (W - PAD.l - PAD.r)
      const y = PAD.t + (1 - (p.equity - minE) / range) * (H - PAD.t - PAD.b)
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')

  const fillPts = `${PAD.l},${H - PAD.b} ` + ptStr + ` ${W - PAD.r},${H - PAD.b}`
  const ticks = [minE, (minE + maxE) / 2, maxE]
  const isProfit = equities[equities.length - 1] >= equities[0]
  const lineColor = isProfit ? '#22c55e' : '#ef4444'
  const gradId = isProfit ? 'eqGradGreen' : 'eqGradRed'
  const gradColor = isProfit ? '#22c55e' : '#ef4444'

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 110 }}>
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={gradColor} stopOpacity="0.25" />
          <stop offset="100%" stopColor={gradColor} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      {ticks.map((v, i) => {
        const y = PAD.t + (1 - (v - minE) / range) * (H - PAD.t - PAD.b)
        return (
          <g key={i}>
            <line x1={PAD.l} y1={y} x2={W - PAD.r} y2={y} stroke="#1e293b" strokeWidth={1} />
            <text x={PAD.l - 4} y={y + 3} textAnchor="end" fontSize={8} fill="#475569">
              {(v / 1000).toFixed(0)}K
            </text>
          </g>
        )
      })}
      <polygon points={fillPts} fill={`url(#${gradId})`} />
      <polyline points={ptStr} fill="none" stroke={lineColor} strokeWidth={1.5} />
    </svg>
  )
}

// ─── Metric Card ──────────────────────────────────────────────────────────────

interface MetricCardProps {
  label: string
  value: string
  sub?: string
  color?: string
}

function MetricCard({ label, value, sub, color = '#e2e8f0' }: MetricCardProps): React.JSX.Element {
  return (
    <div style={metricCardStyle}>
      <div style={{ fontSize: 9, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 700, color, marginTop: 2 }}>{value}</div>
      {sub && <div style={{ fontSize: 9, color: '#334155', marginTop: 1 }}>{sub}</div>}
    </div>
  )
}

// ─── Data status indicator ────────────────────────────────────────────────────

type DataStatus = 'checking' | 'available' | 'missing' | 'syncing' | 'sync_error'

interface DataStatusBarProps {
  status: DataStatus
  barCount: number
  symbol: string
  errorMsg: string
  onSync: () => void
}

function DataStatusBar({ status, barCount, symbol, errorMsg, onSync }: DataStatusBarProps): React.JSX.Element {
  const configs: Record<DataStatus, { color: string; icon: string; msg: string }> = {
    checking:   { color: '#475569', icon: '…', msg: `Checking local data for ${symbol}…` },
    available:  { color: '#22c55e', icon: '✓', msg: `${barCount.toLocaleString()} 1-min bars cached locally` },
    missing:    { color: '#ef4444', icon: '⚠', msg: `No local data for ${symbol}. Sync from FYERS first.` },
    syncing:    { color: '#facc15', icon: '↻', msg: 'Fetching from FYERS and storing to DuckDB…' },
    sync_error: { color: '#ef4444', icon: '✗', msg: errorMsg },
  }
  const { color, icon, msg } = configs[status]

  return (
    <div style={{ ...dataBar, borderColor: color + '30' }}>
      <span style={{ color, fontSize: 11, flexShrink: 0 }}>{icon}</span>
      <span style={{ color: '#94a3b8', fontSize: 10, flex: 1 }}>{msg}</span>
      {(status === 'missing' || status === 'sync_error') && (
        <button onClick={onSync} style={syncBtnStyle}>
          ⬇ Sync Data
        </button>
      )}
      {status === 'syncing' && (
        <span style={{ fontSize: 9, color: '#facc15' }}>please wait…</span>
      )}
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

interface FormState {
  symbol: string
  capital: string
  slPct: string
  tpPct: string
  strategy: string
  instrument: string
  qty: string
  dateFrom: string
  dateTo: string
}

type PanelStatus = 'idle' | 'loading' | 'ok' | 'error'

function today(): string {
  return new Date().toISOString().slice(0, 10)
}
function daysAgo(n: number): string {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return d.toISOString().slice(0, 10)
}

export default function BacktestPanel(): React.JSX.Element {
  const [form, setForm] = useState<FormState>({
    symbol: 'NSE:NIFTY50-INDEX',
    capital: '100000',
    slPct: '1.0',
    tpPct: '2.0',
    strategy: 'vwap_bands',
    instrument: 'equity',
    qty: '1',
    dateFrom: daysAgo(5),
    dateTo: today(),
  })

  const [runStatus, setRunStatus] = useState<PanelStatus>('idle')
  const [result, setResult] = useState<BacktestRunResponse | null>(null)
  const [runError, setRunError] = useState('')

  // Data availability state
  const [dataStatus, setDataStatus] = useState<DataStatus>('checking')
  const [barCount, setBarCount] = useState(0)
  const [syncError, setSyncError] = useState('')

  // Check local data on mount + symbol change
  const checkData = useCallback(async (sym: string) => {
    setDataStatus('checking')
    try {
      const count = await checkSymbolData(sym)
      setBarCount(count)
      setDataStatus(count > 0 ? 'available' : 'missing')
    } catch {
      setDataStatus('missing')
      setBarCount(0)
    }
  }, [])

  useEffect(() => {
    void checkData(form.symbol)
  }, [form.symbol, checkData])

  // Sync data from FYERS
  const handleSync = useCallback(async () => {
    setDataStatus('syncing')
    setSyncError('')
    try {
      const count = await syncHistoricalData(form.symbol, form.dateFrom, form.dateTo)
      setBarCount(count)
      setDataStatus(count > 0 ? 'available' : 'missing')
    } catch (err) {
      setSyncError(String(err))
      setDataStatus('sync_error')
    }
  }, [form.symbol, form.dateFrom, form.dateTo])

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setForm(f => ({ ...f, [name]: value }))
  }, [])

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault()
    if (dataStatus === 'missing') {
      await handleSync()
      return
    }
    setRunStatus('loading')
    setRunError('')

    const slPct = parseFloat(form.slPct) / 100
    const tpPct = parseFloat(form.tpPct) / 100

    try {
      const res = await runBacktest({
        symbol: form.symbol,
        initial_capital: parseFloat(form.capital),
        stop_loss_pct: isNaN(slPct) ? null : slPct,
        take_profit_pct: isNaN(tpPct) ? null : tpPct,
        instrument_type: form.instrument,
        strategy: form.strategy,
        default_quantity: parseInt(form.qty, 10),
        start_date: form.dateFrom || null,
        end_date: form.dateTo || null,
      })
      setResult(res)
      setRunStatus('ok')
    } catch (err) {
      setRunError(String(err))
      setRunStatus('error')
    }
  }, [form, dataStatus, handleSync])

  const taxesPaid = result
    ? result.deductions_breakdown.total.toLocaleString('en-IN', { maximumFractionDigits: 0 })
    : '—'

  return (
    <div style={panelContainer}>
      {/* Header */}
      <div style={panelHeader}>
        <span style={dot('#facc15')} />
        <span>Backtest Engine</span>
        {runStatus === 'loading' && (
          <span style={{ marginLeft: 'auto', color: '#facc15', fontSize: 10 }}>↻ running simulation…</span>
        )}
        {runStatus === 'error' && (
          <span style={{ marginLeft: 'auto', color: '#ef4444', fontSize: 10 }} title={runError}>⚠ {runError.slice(0, 60)}</span>
        )}
      </div>

      {/* Data status */}
      <DataStatusBar
        status={dataStatus}
        barCount={barCount}
        symbol={form.symbol}
        errorMsg={syncError}
        onSync={() => { void handleSync() }}
      />

      {/* Form */}
      <form
        onSubmit={(e) => { void handleSubmit(e) }}
        style={formStyle}
      >
        <div style={formGrid}>
          <label style={labelStyle}>
            Symbol
            <input name="symbol" value={form.symbol} onChange={handleChange} style={inputStyle} />
          </label>
          <label style={labelStyle}>
            Capital (₹)
            <input name="capital" value={form.capital} onChange={handleChange}
              style={inputStyle} type="number" min="1000" />
          </label>
          <label style={labelStyle}>
            SL %
            <input name="slPct" value={form.slPct} onChange={handleChange}
              style={inputStyle} type="number" step="0.1" min="0" />
          </label>
          <label style={labelStyle}>
            TP %
            <input name="tpPct" value={form.tpPct} onChange={handleChange}
              style={inputStyle} type="number" step="0.1" min="0" />
          </label>
          <label style={labelStyle}>
            From
            <input name="dateFrom" value={form.dateFrom} onChange={handleChange}
              style={inputStyle} type="date" />
          </label>
          <label style={labelStyle}>
            To
            <input name="dateTo" value={form.dateTo} onChange={handleChange}
              style={inputStyle} type="date" />
          </label>
          <label style={labelStyle}>
            Strategy
            <select name="strategy" value={form.strategy} onChange={handleChange} style={inputStyle}>
              <option value="vwap_bands">VWAP Bands</option>
              <option value="sma_cross">SMA Cross</option>
              <option value="rsi_momentum">RSI Momentum</option>
            </select>
          </label>
          <label style={labelStyle}>
            Instrument
            <select name="instrument" value={form.instrument} onChange={handleChange} style={inputStyle}>
              <option value="equity">Equity</option>
              <option value="options">Options</option>
            </select>
          </label>
          <label style={labelStyle}>
            Qty / Lots
            <input name="qty" value={form.qty} onChange={handleChange}
              style={inputStyle} type="number" min="1" />
          </label>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 6 }}>
            {dataStatus === 'missing' || dataStatus === 'sync_error' ? (
              <button type="button" onClick={() => { void handleSync() }}
                style={{ ...runBtnStyle, background: '#b45309' }}>
                ⬇ Sync & Run
              </button>
            ) : (
              <button type="submit"
                disabled={runStatus === 'loading' || dataStatus === 'syncing' || dataStatus === 'checking'}
                style={runBtnStyle}>
                {runStatus === 'loading' ? '…' : '▶ Run'}
              </button>
            )}
          </div>
        </div>
      </form>

      {/* Results */}
      {result && (
        <div style={{ flex: 1, overflowY: 'auto', padding: '8px 12px' }}>
          <div style={metricsGrid}>
            <MetricCard
              label="Net CAGR"
              value={`${(result.net_cagr * 100).toFixed(2)}%`}
              color={result.net_cagr >= 0 ? '#22c55e' : '#ef4444'}
            />
            <MetricCard
              label="Sharpe"
              value={result.sharpe_ratio.toFixed(2)}
              color={result.sharpe_ratio >= 1 ? '#22c55e' : result.sharpe_ratio >= 0 ? '#facc15' : '#ef4444'}
            />
            <MetricCard
              label="Max Drawdown"
              value={`${(result.max_drawdown * 100).toFixed(2)}%`}
              color="#ef4444"
            />
            <MetricCard
              label="Profit Factor"
              value={result.profit_factor.toFixed(2)}
              color={result.profit_factor >= 1 ? '#22c55e' : '#ef4444'}
            />
            <MetricCard
              label="Win Rate"
              value={`${(result.win_rate * 100).toFixed(1)}%`}
              sub={`${result.winning_trades}W / ${result.losing_trades}L`}
              color="#94a3b8"
            />
            <MetricCard
              label="Taxes & Charges"
              value={`₹${taxesPaid}`}
              sub={`STT ₹${result.deductions_breakdown.stt.toFixed(0)} · GST ₹${result.deductions_breakdown.gst.toFixed(0)}`}
              color="#fb923c"
            />
          </div>

          <div style={tradeBar}>
            <span style={{ color: '#94a3b8' }}>Trades: {result.total_trades}</span>
            <span style={{ color: '#22c55e' }}>▲ {result.winning_trades}</span>
            <span style={{ color: '#ef4444' }}>▼ {result.losing_trades}</span>
            <span style={{ marginLeft: 'auto', color: '#94a3b8' }}>
              Net P&L:{' '}
              <span style={{ color: result.net_profit >= 0 ? '#22c55e' : '#ef4444' }}>
                ₹{result.net_profit.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
              </span>
            </span>
          </div>

          <div style={{ marginTop: 8, borderRadius: 4, overflow: 'hidden', background: '#0a0c10', padding: 4 }}>
            <div style={{ fontSize: 9, color: '#334155', marginBottom: 2, padding: '0 4px', textTransform: 'uppercase' }}>
              Equity Curve
            </div>
            <EquityCurve points={result.equity_curve} />
          </div>
        </div>
      )}

      {runStatus === 'idle' && !result && (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#1e293b', fontSize: 11 }}>
          {dataStatus === 'available'
            ? 'Configure params and press ▶ Run'
            : dataStatus === 'missing'
            ? 'Press ⬇ Sync Data to fetch candles from FYERS, then run backtest.'
            : ''}
        </div>
      )}
    </div>
  )
}

// ─── styles ───────────────────────────────────────────────────────────────────

const panelContainer: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  background: '#0d0f14',
  overflow: 'hidden',
}

const panelHeader: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '5px 12px',
  fontSize: 11,
  fontWeight: 600,
  letterSpacing: '0.08em',
  color: '#94a3b8',
  borderBottom: '1px solid #1e293b',
  textTransform: 'uppercase' as const,
  flexShrink: 0,
}

const dataBar: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '5px 12px',
  borderBottom: '1px solid #1e293b',
  flexShrink: 0,
  borderLeft: '2px solid transparent',
  background: '#080a0e',
}

const formStyle: React.CSSProperties = {
  padding: '7px 12px',
  borderBottom: '1px solid #1e293b',
  flexShrink: 0,
}

const formGrid: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(110px, 1fr))',
  gap: 7,
  alignItems: 'end',
}

const labelStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 3,
  fontSize: 9,
  color: '#475569',
  textTransform: 'uppercase',
  letterSpacing: '0.06em',
}

const inputStyle: React.CSSProperties = {
  background: '#0a0c10',
  border: '1px solid #1e293b',
  borderRadius: 3,
  color: '#e2e8f0',
  fontSize: 11,
  padding: '3px 6px',
  outline: 'none',
  fontFamily: 'inherit',
  width: '100%',
  colorScheme: 'dark',
}

const runBtnStyle: React.CSSProperties = {
  background: '#1d4ed8',
  border: 'none',
  borderRadius: 3,
  color: '#fff',
  fontSize: 11,
  fontWeight: 700,
  padding: '4px 12px',
  cursor: 'pointer',
  fontFamily: 'inherit',
  letterSpacing: '0.04em',
  width: '100%',
  whiteSpace: 'nowrap',
}

const syncBtnStyle: React.CSSProperties = {
  background: '#78350f',
  border: '1px solid #b45309',
  borderRadius: 3,
  color: '#fde68a',
  fontSize: 9,
  fontWeight: 600,
  padding: '2px 8px',
  cursor: 'pointer',
  fontFamily: 'inherit',
  whiteSpace: 'nowrap',
  flexShrink: 0,
}

const metricsGrid: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(3, 1fr)',
  gap: 7,
  marginBottom: 7,
}

const metricCardStyle: React.CSSProperties = {
  background: '#0a0c10',
  border: '1px solid #1e293b',
  borderRadius: 5,
  padding: '7px 10px',
}

const tradeBar: React.CSSProperties = {
  display: 'flex',
  gap: 12,
  fontSize: 10,
  color: '#94a3b8',
  padding: '3px 2px',
}

function dot(color: string): React.CSSProperties {
  return {
    display: 'inline-block',
    width: 7,
    height: 7,
    borderRadius: '50%',
    background: color,
    flexShrink: 0,
  }
}

// silence unused import warning — only used in effect
void fetchAvailableSymbols
