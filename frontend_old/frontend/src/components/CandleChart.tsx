import { useEffect, useRef } from 'react'
import {
  CandlestickSeries,
  ColorType,
  HistogramSeries,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type Time,
} from 'lightweight-charts'
import { DOWN_COLOR, UP_COLOR, toCandlestickData, toVolumeData } from '@/lib/candles'
import { formatIst } from '@/lib/format'
import type { Candle } from '@/types/api'

interface CandleChartProps {
  rows: Candle[]
}

/**
 * Candlestick chart with a volume histogram.
 * The chart is created once. Data updates go through setData, never by re-creating the chart.
 */
export function CandleChart({ rows }: CandleChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volumeRef = useRef<ISeriesApi<'Histogram'> | null>(null)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const chart = createChart(el, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: 'rgba(148, 163, 184, 0.12)' },
        horzLines: { color: 'rgba(148, 163, 184, 0.12)' },
      },
      localization: {
        // Bars are stored as UTC epoch seconds. Show session time in IST.
        timeFormatter: (time: Time) => (typeof time === 'number' ? formatIst(time) : String(time)),
      },
      timeScale: { timeVisible: true, secondsVisible: false, borderVisible: false },
      rightPriceScale: { borderVisible: false },
    })

    candleRef.current = chart.addSeries(CandlestickSeries, {
      upColor: UP_COLOR,
      downColor: DOWN_COLOR,
      borderVisible: false,
      wickUpColor: UP_COLOR,
      wickDownColor: DOWN_COLOR,
    })
    volumeRef.current = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    })
    chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } })

    chartRef.current = chart
    return () => {
      chart.remove()
      chartRef.current = null
      candleRef.current = null
      volumeRef.current = null
    }
  }, [])

  useEffect(() => {
    candleRef.current?.setData(toCandlestickData(rows))
    volumeRef.current?.setData(toVolumeData(rows))
    chartRef.current?.timeScale().fitContent()
  }, [rows])

  return (
    <div
      ref={containerRef}
      className="chart"
      role="img"
      aria-label={`Candlestick chart with ${rows.length} bars`}
    />
  )
}
