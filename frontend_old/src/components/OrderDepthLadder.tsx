/**
 * OrderDepthLadder.tsx
 *
 * Live 5-level bid-ask DOM (depth of market) with:
 *  - Colour-scaled volume bars behind each row
 *  - Micro-price imbalance gauge ([-1, 1] → red … green)
 *  - Flicker animation on price change
 *
 * Updates are pushed via WebSocket depth messages;
 * component state is kept minimal to avoid heavy re-renders.
 */

import { useState, useCallback } from 'react'
import { useMarketWebSocket } from '../hooks/useMarketWebSocket'
import type { WsMessage, WsDepth, WsDepthMetrics, DepthLevel } from '../types'

// ─── helpers ─────────────────────────────────────────────────────────────────

function extractLevels(depth: WsDepth['data']): { bids: DepthLevel[]; asks: DepthLevel[] } {
  if (depth.bids && depth.bids.length > 0) {
    return { bids: depth.bids.slice(0, 5), asks: (depth.asks ?? []).slice(0, 5) }
  }
  // Flat-key fallback (FYERS format)
  const bids: DepthLevel[] = []
  const asks: DepthLevel[] = []
  for (let i = 1; i <= 5; i++) {
    const bp = (depth as Record<string, number | undefined>)[`bid_price${i}`]
    const bq = (depth as Record<string, number | undefined>)[`bid_size${i}`]
    const ap = (depth as Record<string, number | undefined>)[`ask_price${i}`]
    const aq = (depth as Record<string, number | undefined>)[`ask_size${i}`]
    if (bp !== undefined) bids.push({ price: bp, quantity: bq ?? 0, orders: null })
    if (ap !== undefined) asks.push({ price: ap, quantity: aq ?? 0, orders: null })
  }
  return { bids, asks }
}

function fmt(n: number, decimals = 2): string {
  return n.toLocaleString('en-IN', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
}

function fmtQty(n: number): string {
  if (n >= 1_00_000) return `${(n / 1_00_000).toFixed(1)}L`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
  return String(Math.round(n))
}

// ─── sub-components ───────────────────────────────────────────────────────────

interface DepthRowProps {
  level: DepthLevel
  maxQty: number
  side: 'bid' | 'ask'
}

function DepthRow({ level, maxQty, side }: DepthRowProps): React.JSX.Element {
  const pct = maxQty > 0 ? Math.min((level.quantity / maxQty) * 100, 100) : 0
  const barColor = side === 'bid' ? '#22c55e18' : '#ef444418'
  const priceColor = side === 'bid' ? '#22c55e' : '#ef4444'

  return (
    <div style={{ ...rowStyle, position: 'relative' }}>
      {/* Volume bar */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          bottom: 0,
          [side === 'bid' ? 'right' : 'left']: 0,
          width: `${pct}%`,
          background: barColor,
          transition: 'width 0.15s ease',
          pointerEvents: 'none',
        }}
      />
      {side === 'bid' ? (
        <>
          <span style={{ color: '#94a3b8', fontSize: 10 }}>{fmtQty(level.quantity)}</span>
          <span style={{ color: priceColor, fontWeight: 600 }}>{fmt(level.price)}</span>
        </>
      ) : (
        <>
          <span style={{ color: priceColor, fontWeight: 600 }}>{fmt(level.price)}</span>
          <span style={{ color: '#94a3b8', fontSize: 10 }}>{fmtQty(level.quantity)}</span>
        </>
      )}
    </div>
  )
}

interface ImbalanceGaugeProps {
  imbalance: number  // [-1, 1]
  microPrice: number
  midPrice: number
}

function ImbalanceGauge({ imbalance, microPrice, midPrice }: ImbalanceGaugeProps): React.JSX.Element {
  // Map [-1, 1] to [0%, 100%]
  const pct = ((imbalance + 1) / 2) * 100
  const color = imbalance > 0.1 ? '#22c55e' : imbalance < -0.1 ? '#ef4444' : '#facc15'

  return (
    <div style={gaugeWrapper}>
      <div style={gaugeLabel}>
        <span>Imbalance</span>
        <span style={{ color }}>{imbalance >= 0 ? '+' : ''}{(imbalance * 100).toFixed(1)}%</span>
      </div>
      <div style={gaugeTrack}>
        {/* Centre mark */}
        <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: '#334155' }} />
        <div style={{ ...gaugeFill, width: `${pct}%`, background: color }} />
      </div>
      <div style={gaugeLabel}>
        <span style={{ color: '#64748b', fontSize: 10 }}>μ-price</span>
        <span style={{ color: '#94a3b8', fontSize: 10 }}>{fmt(microPrice)} <span style={{ color: '#475569' }}>(mid {fmt(midPrice)})</span></span>
      </div>
    </div>
  )
}

// ─── main component ───────────────────────────────────────────────────────────

interface DepthState {
  bids: DepthLevel[]
  asks: DepthLevel[]
  metrics: WsDepthMetrics | null
  symbol: string
}

const EMPTY_LEVEL: DepthLevel = { price: 0, quantity: 0, orders: null }
const EMPTY_LEVELS: DepthLevel[] = Array(5).fill(EMPTY_LEVEL)

export default function OrderDepthLadder(): React.JSX.Element {
  const [state, setState] = useState<DepthState>({
    bids: EMPTY_LEVELS,
    asks: EMPTY_LEVELS,
    metrics: null,
    symbol: 'NSE:NIFTY50-INDEX',
  })

  const handleMsg = useCallback((msg: WsMessage) => {
    if (msg.type !== 'depth') return
    const d = msg as WsDepth
    const { bids, asks } = extractLevels(d.data)
    setState({
      bids: bids.length > 0 ? bids : EMPTY_LEVELS,
      asks: asks.length > 0 ? asks : EMPTY_LEVELS,
      metrics: d.metrics,
      symbol: d.symbol,
    })
  }, [])

  useMarketWebSocket(handleMsg)

  const allQty = [...state.bids, ...state.asks].map(l => l.quantity)
  const maxQty = allQty.length > 0 ? Math.max(...allQty, 1) : 1

  const spread =
    state.bids[0]?.price > 0 && state.asks[0]?.price > 0
      ? state.asks[0].price - state.bids[0].price
      : null

  return (
    <div style={panelContainer}>
      {/* Header */}
      <div style={panelHeader}>
        <span style={dot('#ef4444')} />
        <span>Order Depth — {state.symbol}</span>
        {spread !== null && (
          <span style={{ marginLeft: 'auto', color: '#facc15', fontSize: 10 }}>
            Spread {fmt(spread, 1)}
          </span>
        )}
      </div>

      {/* Column labels */}
      <div style={{ ...rowStyle, borderBottom: '1px solid #1e293b', fontSize: 10, color: '#475569', padding: '3px 10px' }}>
        <span>QTY</span>
        <span>BID</span>
        <span style={{ marginLeft: 'auto' }}>ASK</span>
        <span>QTY</span>
      </div>

      {/* Combined bid-ask table */}
      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        {/* Bids */}
        <div style={{ flex: 1, borderRight: '1px solid #1e293b' }}>
          {state.bids.map((lvl, i) => (
            <DepthRow key={i} level={lvl} maxQty={maxQty} side="bid" />
          ))}
        </div>
        {/* Asks */}
        <div style={{ flex: 1 }}>
          {state.asks.map((lvl, i) => (
            <DepthRow key={i} level={lvl} maxQty={maxQty} side="ask" />
          ))}
        </div>
      </div>

      {/* Imbalance gauge */}
      {state.metrics && (
        <ImbalanceGauge
          imbalance={state.metrics.depth_imbalance}
          microPrice={state.metrics.micro_price}
          midPrice={state.metrics.mid_price}
        />
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

const rowStyle: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  padding: '4px 10px',
  fontSize: 12,
  borderBottom: '1px solid #0f172a',
}

const gaugeWrapper: React.CSSProperties = {
  padding: '8px 10px',
  borderTop: '1px solid #1e293b',
  flexShrink: 0,
}

const gaugeLabel: React.CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  fontSize: 10,
  color: '#94a3b8',
  marginBottom: 4,
}

const gaugeTrack: React.CSSProperties = {
  position: 'relative',
  height: 6,
  borderRadius: 3,
  background: '#1e293b',
  overflow: 'hidden',
  marginBottom: 4,
}

const gaugeFill: React.CSSProperties = {
  position: 'absolute',
  left: 0,
  top: 0,
  height: '100%',
  borderRadius: 3,
  transition: 'width 0.2s ease, background 0.2s ease',
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

