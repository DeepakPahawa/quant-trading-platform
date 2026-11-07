/**
 * CandlestickChart.tsx  (lightweight-charts v5 API)
 *
 * Renders:
 *  - 1-minute OHLCV candlestick series
 *  - Session VWAP line + ±1σ band (line series)
 *  - TFT p10–p90 forward funnel (area series) projecting from the latest candle
 *
 * Chart instance lives in a useRef and is NEVER torn down on re-render.
 * WebSocket bar + TFT forecast messages are pushed directly into the series refs.
 */

import { useEffect, useRef } from 'react'
import {
  createChart,
  CrosshairMode,
  LineStyle,
  CandlestickSeries,
  LineSeries,
  AreaSeries,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type LineData,
  type AreaData,
  type Time,
  type DeepPartial,
  type ChartOptions,
} from 'lightweight-charts'
import { useMarketWebSocket } from '../hooks/useMarketWebSocket'
import type { WsMessage, WsBar, WsTftForecast } from '../types'

// ─── helpers ─────────────────────────────────────────────────────────────────

function toTime(epoch: number): Time {
  return epoch as Time
}

// ─── chart options ────────────────────────────────────────────────────────────

const CHART_OPTIONS: DeepPartial<ChartOptions> = {
  layout: {
    background: { color: '#0d0f14' },
    textColor: '#94a3b8',
  },
  grid: {
    vertLines: { color: '#1e293b' },
    horzLines: { color: '#1e293b' },
  },
  crosshair: { mode: CrosshairMode.Normal },
  rightPriceScale: { borderColor: '#334155' },
  timeScale: {
    borderColor: '#334155',
    timeVisible: true,
    secondsVisible: false,
  },
}

// ─── component ───────────────────────────────────────────────────────────────

export default function CandlestickChart(): React.JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const vwapSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const vwapUpperRef = useRef<ISeriesApi<'Line'> | null>(null)
  const vwapLowerRef = useRef<ISeriesApi<'Line'> | null>(null)
  const p10SeriesRef = useRef<ISeriesApi<'Area'> | null>(null)
  const p90SeriesRef = useRef<ISeriesApi<'Area'> | null>(null)

  // ── Mount chart once ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      ...CHART_OPTIONS,
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
    })
    chartRef.current = chart

    // v5: chart.addSeries(SeriesDefinition, options)
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    })
    candleSeriesRef.current = candleSeries

    const vwapSeries = chart.addSeries(LineSeries, {
      color: '#facc15',
      lineWidth: 1,
      lineStyle: LineStyle.Solid,
      priceLineVisible: false,
      lastValueVisible: false,
    })
    vwapSeriesRef.current = vwapSeries

    const vwapUpper = chart.addSeries(LineSeries, {
      color: '#fbbf2440',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      priceLineVisible: false,
      lastValueVisible: false,
    })
    vwapUpperRef.current = vwapUpper

    const vwapLower = chart.addSeries(LineSeries, {
      color: '#fbbf2440',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      priceLineVisible: false,
      lastValueVisible: false,
    })
    vwapLowerRef.current = vwapLower

    // TFT p90 band (top envelope)
    const p90Series = chart.addSeries(AreaSeries, {
      lineColor: '#818cf880',
      topColor: '#818cf820',
      bottomColor: '#818cf800',
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    })
    p90SeriesRef.current = p90Series

    // TFT p10 band (bottom envelope)
    const p10Series = chart.addSeries(AreaSeries, {
      lineColor: '#818cf840',
      topColor: '#818cf810',
      bottomColor: '#818cf800',
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    })
    p10SeriesRef.current = p10Series

    // Responsive resize
    const ro = new ResizeObserver(entries => {
      const entry = entries[0]
      if (entry) {
        chart.applyOptions({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        })
      }
    })
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      chart.remove()
      chartRef.current = null
    }
  }, [])

  // ── WebSocket handler (stable across renders — uses refs) ─────────────────
  useMarketWebSocket((msg: WsMessage) => {
    if (msg.type === 'bar') {
      const bar = msg as WsBar
      const d = bar.data

      const candle: CandlestickData = {
        time: toTime(d.timestamp),
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }
      candleSeriesRef.current?.update(candle)

      if (d.vwap !== undefined) {
        const lp: LineData = { time: toTime(d.timestamp), value: d.vwap }
        vwapSeriesRef.current?.update(lp)
      }
      if (d.vwap_upper !== undefined) {
        vwapUpperRef.current?.update({ time: toTime(d.timestamp), value: d.vwap_upper })
      }
      if (d.vwap_lower !== undefined) {
        vwapLowerRef.current?.update({ time: toTime(d.timestamp), value: d.vwap_lower })
      }
    }

    if (msg.type === 'tft_forecast') {
      const fc = msg as WsTftForecast
      const p90Data: AreaData[] = fc.predictions.map(p => ({ time: toTime(p.t), value: p.p90 }))
      const p10Data: AreaData[] = fc.predictions.map(p => ({ time: toTime(p.t), value: p.p10 }))
      p90SeriesRef.current?.setData(p90Data)
      p10SeriesRef.current?.setData(p10Data)
    }
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={panelHeader}>
        <span style={dot('#22c55e')} />
        <span>NIFTY 1-min · VWAP Bands · TFT Funnel</span>
      </div>
      <div ref={containerRef} style={{ flex: 1, minHeight: 0 }} />
    </div>
  )
}

// ─── inline styles ────────────────────────────────────────────────────────────

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
  textTransform: 'uppercase',
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
