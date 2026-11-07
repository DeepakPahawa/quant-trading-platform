import { bandPosition, DISCLAIMER } from '@/lib/forecast'
import { formatInr } from '@/lib/format'
import type { ForecastResponse, HorizonForecast } from '@/types/api'

const BIAS_LABEL: Record<HorizonForecast['directional_bias'], string> = {
  BULLISH: '▲ Bullish',
  BEARISH: '▼ Bearish',
  NEUTRAL: '● Neutral',
}

interface ForecastPanelProps {
  forecast: ForecastResponse
}

function HorizonRow({ label, h, current }: { label: string; h: HorizonForecast; current: number | null }) {
  const pos = bandPosition(h.p50, h.p10, h.p90)
  return (
    <div className="horizon">
      <div className="horizon-head">
        <strong>{label}</strong>
        <span className={`bias bias-${h.directional_bias.toLowerCase()}`}>{BIAS_LABEL[h.directional_bias]}</span>
      </div>
      <div className="band" aria-label={`${label} band from ${formatInr(h.p10)} to ${formatInr(h.p90)}`}>
        <div className="band-range" />
        <div className="band-median" style={{ left: `${pos}%` }} />
      </div>
      <div className="band-labels">
        <span>p10 {formatInr(h.p10)}</span>
        <span>p50 {formatInr(h.p50)}</span>
        <span>p90 {formatInr(h.p90)}</span>
      </div>
      {current !== null && <div className="muted small">Reference price {formatInr(current)}</div>}
    </div>
  )
}

/** Shows the TFT quantile band for each horizon, or buffering progress. */
export function ForecastPanel({ forecast }: ForecastPanelProps) {
  if (forecast.status === 'BUFFERING') {
    const have = forecast.buffer_count ?? 0
    const need = forecast.required_bars ?? 60
    return (
      <section className="card" aria-live="polite">
        <h2>Forecast</h2>
        <p>Collecting bars: {have} of {need}. The model needs {need} one-minute bars before it can forecast.</p>
        <p className="disclaimer">{DISCLAIMER}</p>
      </section>
    )
  }

  const horizons = forecast.horizons ?? {}
  const entries = Object.entries(horizons)
  return (
    <section className="card">
      <h2>Forecast</h2>
      {entries.length === 0 ? (
        <p className="muted">No horizon data returned.</p>
      ) : (
        entries.map(([label, h]) => (
          <HorizonRow key={label} label={label} h={h} current={forecast.current_price} />
        ))
      )}
      <p className="disclaimer">{DISCLAIMER}</p>
    </section>
  )
}
