import { BrowserRouter, Link, NavLink, Route, Routes } from 'react-router-dom'
import { LoginPage } from '@/pages/LoginPage'
import { SymbolPage } from '@/pages/SymbolPage'
import { WatchlistPage } from '@/pages/WatchlistPage'

export default function App() {
  return (
    <BrowserRouter>
      <header className="topbar">
        <Link to="/" className="brand">Quant Platform</Link>
        <nav aria-label="Main">
          <NavLink to="/">Watchlist</NavLink>
          <NavLink to="/symbol/NSE:SBIN-EQ">Symbol</NavLink>
          <NavLink to="/login">FYERS login</NavLink>
        </nav>
      </header>
      <main className="container">
        <Routes>
          <Route path="/" element={<WatchlistPage />} />
          <Route path="/symbol/:symbol" element={<SymbolPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="*" element={<p>Page not found.</p>} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}
