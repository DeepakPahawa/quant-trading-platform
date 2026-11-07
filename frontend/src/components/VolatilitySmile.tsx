/**
 * VolatilitySmile.tsx
 *
 * Fetches QuantLib-calibrated SVI curve from GET /api/options/chain/{symbol}
 * and renders:
 *  - SVI model IV curve (smooth line)
 *  - Market quote IV dots (CE in blue, PE in orange)
 *  - ATM strike marker
 *
 * Uses an SVG canvas drawn manually — no extra charting lib needed.
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import type { OptionsChainResponse, SviCurvePoint, OptionQuote } from '../types'

// ─── API ──────────────────────────────────────────────────────────────────────

async function fetchOptionsChain(symbol: string): Promise<OptionsChainResponse> {
  const res = await fetch(`/api/options/chain/${encodeURIComponent(symbol)}?strikecount=12`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<OptionsChainResponse>
}

// ─── SVG chart helpers ────────────────────────────────────────────────────────

interface Range { min: number; max: number }

function mapX(val: number, range: Range, width: number, pad: number): number {
  return pad + ((val - range.min) / (range.max - range.min)) * (width - 2 * pad)
}

function mapY(val: number, range: Range, height: number, pad: number): number {
  return height - pad - ((val - range.min) / (range.max - range.min)) * (height - 2 * pad)
}

function buildLinePath(points: Array<{ x: number; y: number }>): string {
  if (points.length === 0) return ''
  return points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
    .join(' ')
}

// ─── component ───────────────────────────────────────────────────────────────

interface SmileState {
  curve: SviCurvePoint[]
  quotes: OptionQuote[]
  spot: number
  status: 'idle' | 'loading' | 'ok' | 'error'
  error: string
  symbol: string
}

const SYMBOL = 'NSE:NIFTY50-INDEX'
const REFRESH_MS = 60_000

export default function VolatilitySmile(): React.JSX.Element {
  const [state, setState] = useState<SmileState>({
    curve: [], quotes: [], spot: 0,
    status: 'idle', error: '', symbol: SYMBOL,
  })
  const svgRef = useRef<SVGSVGElement>(null)
  const [dims, setDims] = useState({ w: 0, h: 0 })

  // Track container size
  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return
    const ro = new ResizeObserver(entries => {
      const e = entries[0]
      if (e) setDims({ w: e.contentRect.width, h: e.contentRect.height })
    })
    ro.observe(svg)
    return () => ro.disconnect()
  }, [])

  const load = useCallback(async (sym: string) => {
    setState(s => ({ ...s, status: 'loading', error: '' }))
    try {
      const data = await fetchOptionsChain(sym)
      setState(s => ({
        ...s, status: 'ok',
        curve: data.curve_points,
        quotes: data.quotes,
        spot: data.spot,
        symbol: sym,
      }))
    } catch (e) {
      setState(s => ({ ...s, status: 'error', error: String(e) }))
    }
  }, [])

  useEffect(() => {
    void load(SYMBOL)
    const t = setInterval(() => { void load(SYMBOL) }, REFRESH_MS)
    return () => clearInterval(t)
  }, [load])

  // ── Build SVG paths ────────────────────────────────────────────────────────
  const PAD = 36
  const { w, h } = dims
  const hasData = state.curve.length > 0 && w > 0 && h > 0

  let curvePath = ''
  let quoteDotsAll: Array<{ cx: number; cy: number; color: string; label: string }> = []
  let xRange: Range = { min: 0, max: 1 }
  let yRange: Range = { min: 0, max: 1 }
  let xTicks: Array<{ x: number; label: string }> = []
  let yTicks: Array<{ y: number; label: string }> = []

  if (hasData) {
    const allStrikes = [...state.curve.map(p => p.strike), ...state.quotes.map(q => q.strike)]
    const allIVs = [
      ...state.curve.map(p => p.model_iv),
      ...state.quotes.map(q => q.iv).filter(v => v > 0 && v < 2),
    ]
    xRange = { min: Math.min(...allStrikes), max: Math.max(...allStrikes) }
    yRange = { min: Math.max(0, Math.min(...allIVs) - 0.02), max: Math.max(...allIVs) + 0.03 }

    const curvePoints = state.curve.map(p => ({
      x: mapX(p.strike, xRange, w, PAD),
      y: mapY(p.model_iv, yRange, h, PAD),
    }))
    curvePath = buildLinePath(curvePoints)

    quoteDotsAll = state.quotes
      .filter(q => q.iv > 0 && q.iv < 2)
      .map(q => ({
        cx: mapX(q.strike, xRange, w, PAD),
        cy: mapY(q.iv, yRange, h, PAD),
        color: q.option_type === 'CE' ? '#60a5fa' : '#fb923c',
        label: `${q.option_type} K=${q.strike} IV=${(q.iv * 100).toFixed(1)}%`,
      }))

    // X ticks — 5 strike levels
    const xStep = (xRange.max - xRange.min) / 4
    xTicks = Array.from({ length: 5 }, (_, i) => {
      const val = xRange.min + xStep * i
      return { x: mapX(val, xRange, w, PAD), label: val.toFixed(0) }
    })

    // Y ticks — 4 IV levels
    const yStep = (yRange.max - yRange.min) / 3
    yTicks = Array.from({ length: 4 }, (_, i) => {
      const val = yRange.min + yStep * i
      return { y: mapY(val, yRange, h, PAD), label: `${(val * 100).toFixed(0)}%` }
    })
  }

  return (
    <div style={panelContainer}>
      <div style={panelHeader}>
        <span style={dot('#818cf8')} />
        <span>IV Smile — SVI Calibration · {state.symbol}</span>
        {state.status === 'loading' && <span style={{ marginLeft: 'auto', color: '#facc15', fontSize: 10 }}>↻ loading…</span>}
        {state.status === 'error' && <span style={{ marginLeft: 'auto', color: '#ef4444', fontSize: 10 }}>⚠ {state.error}</span>}
        <span style={{ marginLeft: 'auto', display: 'flex', gap: 10, alignItems: 'center' }}>
          <span style={{ color: '#60a5fa', fontSize: 10 }}>● CE</span>
          <span style={{ color: '#fb923c', fontSize: 10 }}>● PE</span>
          <span style={{ color: '#a78bfa', fontSize: 10 }}>— SVI</span>
        </span>
      </div>

      <svg ref={svgRef} style={{ flex: 1, width: '100%', display: 'block' }}>
        {hasData && (
          <>
            {/* Grid lines */}
            {yTicks.map((t, i) => (
              <line key={i} x1={PAD} y1={t.y} x2={w - PAD} y2={t.y}
                stroke="#1e293b" strokeWidth={1} />
            ))}
            {xTicks.map((t, i) => (
              <line key={i} x1={t.x} y1={PAD} x2={t.x} y2={h - PAD}
                stroke="#1e293b" strokeWidth={1} />
            ))}

            {/* ATM line */}
            {state.spot > 0 && (
              <line
                x1={mapX(state.spot, xRange, w, PAD)}
                y1={PAD}
                x2={mapX(state.spot, xRange, w, PAD)}
                y2={h - PAD}
                stroke="#facc1540"
                strokeWidth={1}
                strokeDasharray="4 3"
              />
            )}

            {/* SVI model curve */}
            <path d={curvePath} fill="none" stroke="#a78bfa" strokeWidth={2} />

            {/* Quote dots */}
            {quoteDotsAll.map((d, i) => (
              <circle key={i} cx={d.cx} cy={d.cy} r={4}
                fill={d.color} opacity={0.85}>
                <title>{d.label}</title>
              </circle>
            ))}

            {/* Y axis ticks */}
            {yTicks.map((t, i) => (
              <text key={i} x={PAD - 4} y={t.y + 4}
                textAnchor="end" fontSize={9} fill="#475569">
                {t.label}
              </text>
            ))}

            {/* X axis ticks */}
            {xTicks.map((t, i) => (
              <text key={i} x={t.x} y={h - PAD + 14}
                textAnchor="middle" fontSize={9} fill="#475569">
                {t.label}
              </text>
            ))}

            {/* Axis labels */}
            <text x={PAD / 2} y={(h / 2)} textAnchor="middle" fontSize={9} fill="#334155"
              transform={`rotate(-90, ${PAD / 2}, ${h / 2})`}>
              Implied Vol
            </text>
            <text x={w / 2} y={h - 2} textAnchor="middle" fontSize={9} fill="#334155">
              Strike
            </text>
          </>
        )}

        {state.status === 'idle' || (!hasData && state.status !== 'loading') ? (
          <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle"
            fontSize={12} fill="#334155">
            {state.status === 'error' ? state.error : 'Loading IV smile…'}
          </text>
        ) : null}
      </svg>
    </div>
  )
}

// ─── styles ───────────────────────────────────────────────────────────────────

const panelContainer: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  background: '#0d0f14',
}

const panelHeader: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '6px 12px',
  fontSize: 11,
  fontWeight: 600,
  letterSpacing: '0.08em',
  color: '#94a3b8',
  borderBottom: '1px solid #1e293b',
  textTransform: 'uppercase' as const,
  flexShrink: 0,
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

