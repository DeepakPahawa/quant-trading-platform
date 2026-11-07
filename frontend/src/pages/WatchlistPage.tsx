import { Link } from 'react-router-dom'
import { ApiError, getQuotes } from '@/services/api'
import { usePolling } from '@/hooks/usePolling'
import { formatInr } from '@/lib/format'
import { DEFAULT_SYMBOLS } from '@/lib/symbols'
import type { QuoteData } from '@/types/api'

const REFRESH_MS = 5000

function changeClass(pct: number | undefined): string {
  if (pct === undefined) return 'muted'
  if (pct > 0) return 'up'
  if (pct < 0) return 'down'
  return 'muted'
}

function QuoteRow({ q }: { q: QuoteData }) {
  const pct = q.change_pct
  return (
    <tr>
      <th scope="row">
        <Link to={`/symbol/${encodeURIComponent(q.symbol)}`}>{q.symbol}</Link>
      </th>
      <td className="num">{formatInr(q.ltp)}</td>
      <td className={`num ${changeClass(pct)}`}>
        {pct === undefined ? '—' : `${pct > 0 ? '+' : ''}${pct.toFixed(2)}%`}
      </td>
      <td className="num">{q.depth_imbalance === undefined ? '—' : q.depth_imbalance.toFixed(3)}</td>
      <td className="num">{formatInr(q.micro_price)}</td>
    </tr>
  )
}

export function WatchlistPage() {
  const { data, error, loading, reload } = usePolling(
    () => getQuotes(DEFAULT_SYMBOLS),
    REFRESH_MS,
    'watchlist',
  )

  const unauthenticated = error instanceof ApiError && (error.status === 401 || error.status === 502)

  return (
    <section>
      <div className="page-head">
        <h1>Watchlist</h1>
        <button type="button" onClick={reload}>Refresh</button>
      </div>

      {loading && !data && <p className="muted">Loading quotes…</p>}

      {unauthenticated && (
        <div className="notice" role="alert">
          FYERS is not connected. <Link to="/login">Log in to FYERS</Link> to see live quotes.
        </div>
      )}

      {error !== null && error !== undefined && !unauthenticated && (
        <div className="notice" role="alert">Could not load quotes: {(error as Error).message}</div>
      )}

      {data && (
        <div className="table-wrap">
          <table>
            <caption className="sr-only">Live quotes</caption>
            <thead>
              <tr>
                <th scope="col">Symbol</th>
                <th scope="col" className="num">LTP</th>
                <th scope="col" className="num">Change</th>
                <th scope="col" className="num">Depth imbalance</th>
                <th scope="col" className="num">Micro-price</th>
              </tr>
            </thead>
            <tbody>
              {data.data.map((q) => (
                <QuoteRow key={q.symbol} q={q} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
