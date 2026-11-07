import { useMemo } from 'react'
import { useParams } from 'react-router-dom'
import { CandleChart } from '@/components/CandleChart'
import { ForecastPanel } from '@/components/ForecastPanel'
import { usePolling } from '@/hooks/usePolling'
import { getForecast, getHistory } from '@/services/api'

const FORECAST_REFRESH_MS = 60_000

export function SymbolPage() {
  const { symbol: rawSymbol = '' } = useParams()
  const symbol = decodeURIComponent(rawSymbol)

  const history = usePolling(
    () => getHistory(symbol, { resolution: '1' }),
    FORECAST_REFRESH_MS,
    symbol,
  )
  const forecast = usePolling(
    () => getForecast(symbol),
    FORECAST_REFRESH_MS,
    symbol,
  )

  // Keep the array identity stable between polls that return the same data.
  const candles = useMemo(() => history.data?.candles ?? [], [history.data])

  return (
    <section>
      <div className="page-head">
        <h1>{symbol}</h1>
        <button type="button" onClick={() => { history.reload(); forecast.reload() }}>Refresh</button>
      </div>

      <div className="grid-2">
        <section className="card">
          <h2>1-minute bars (IST)</h2>
          {history.loading && !history.data && <p className="muted">Loading bars…</p>}
          {history.error !== null && history.error !== undefined && (
            <div className="notice" role="alert">Could not load bars: {(history.error as Error).message}</div>
          )}
          {history.data && candles.length === 0 && (
            <p className="muted">No archived bars for this symbol yet. Sync them with the backend first.</p>
          )}
          {candles.length > 0 && <CandleChart rows={candles} />}
        </section>

        <div>
          {forecast.loading && !forecast.data && <p className="muted">Loading forecast…</p>}
          {forecast.error !== null && forecast.error !== undefined && (
            <div className="notice" role="alert">Could not load forecast: {(forecast.error as Error).message}</div>
          )}
          {forecast.data && <ForecastPanel forecast={forecast.data} />}
        </div>
      </div>
    </section>
  )
}
