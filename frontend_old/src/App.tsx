/**
 * App.tsx — dark-mode terminal layout
 *
 * ┌─────────────────────────────┬─────────────────┐
 * │  CandlestickChart (60%)     │ OrderDepthLadder │
 * │                             │      (20%)       │
 * ├──────────────────────┬──────┴─────────────────-┤
 * │  BacktestPanel (50%) │ VolatilitySmile (50%)   │
 * └──────────────────────┴─────────────────────────┘
 */

import CandlestickChart from './components/CandlestickChart'
import OrderDepthLadder from './components/OrderDepthLadder'
import VolatilitySmile from './components/VolatilitySmile'
import BacktestPanel from './components/BacktestPanel'

export default function App(): React.JSX.Element {
  return (
    <div style={rootStyle}>
      {/* Top bar */}
      <header style={topBar}>
        <span style={logo}>⬡ QUANT TERMINAL</span>
        <span style={subLabel}>NSE · Real-time · TFT Forecasting · SVI Vol Surface</span>
        <span style={{ marginLeft: 'auto', fontSize: 10, color: '#334155' }}>
          ws://localhost:8000/ws/market
        </span>
      </header>

      {/* Main grid */}
      <div style={mainGrid}>
        {/* Row 1 */}
        <div style={{ ...cell, gridColumn: '1 / 3', gridRow: '1 / 2' }}>
          <CandlestickChart />
        </div>
        <div style={{ ...cell, gridColumn: '3 / 4', gridRow: '1 / 2' }}>
          <OrderDepthLadder />
        </div>

        {/* Row 2 */}
        <div style={{ ...cell, gridColumn: '1 / 2', gridRow: '2 / 3' }}>
          <BacktestPanel />
        </div>
        <div style={{ ...cell, gridColumn: '2 / 4', gridRow: '2 / 3' }}>
          <VolatilitySmile />
        </div>
      </div>
    </div>
  )
}

// ─── styles ───────────────────────────────────────────────────────────────────

const rootStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  height: '100vh',
  background: '#080a0e',
  color: '#e2e8f0',
  fontFamily: "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
  overflow: 'hidden',
}

const topBar: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 16,
  padding: '5px 16px',
  background: '#0d0f14',
  borderBottom: '1px solid #1e293b',
  flexShrink: 0,
  height: 36,
}

const logo: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 800,
  color: '#facc15',
  letterSpacing: '0.12em',
}

const subLabel: React.CSSProperties = {
  fontSize: 10,
  color: '#334155',
  letterSpacing: '0.06em',
}

const mainGrid: React.CSSProperties = {
  flex: 1,
  display: 'grid',
  gridTemplateColumns: '3fr 2fr 1.4fr',
  gridTemplateRows: '60% 40%',
  gap: 1,
  background: '#0f172a',   // gap colour
  minHeight: 0,
  overflow: 'hidden',
}

const cell: React.CSSProperties = {
  background: '#0d0f14',
  overflow: 'hidden',
  minHeight: 0,
  minWidth: 0,
}

