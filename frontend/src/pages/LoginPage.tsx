import { useEffect, useState, type FormEvent } from 'react'
import { exchangeAuthCode, getHealth, getLoginUrl } from '@/services/api'
import type { HealthResponse } from '@/types/api'

/**
 * FYERS login. The backend performs the exchange and holds the session.
 * The pasted code or URL is sent once and is never stored in the browser.
 */
export function LoginPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [loginUrl, setLoginUrl] = useState<string | null>(null)
  const [redirected, setRedirected] = useState('')
  const [message, setMessage] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const refreshHealth = () => {
    getHealth()
      .then((h) => { setHealth(h); setHealthError(null) })
      .catch((err: Error) => setHealthError(err.message))
  }

  useEffect(refreshHealth, [])

  const openLogin = async () => {
    setMessage(null)
    try {
      const { login_url } = await getLoginUrl()
      setLoginUrl(login_url)
      window.open(login_url, '_blank', 'noopener,noreferrer')
    } catch (err) {
      setMessage(`Could not get login URL: ${(err as Error).message}`)
    }
  }

  const submit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!redirected.trim()) return
    setBusy(true)
    setMessage(null)
    try {
      const res = await exchangeAuthCode(redirected.trim())
      setMessage(res.authenticated ? 'Connected to FYERS.' : res.message || 'Login was not completed.')
      setRedirected('')
      refreshHealth()
    } catch (err) {
      setMessage(`Exchange failed: ${(err as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section>
      <h1>FYERS login</h1>

      <section className="card">
        <h2>Status</h2>
        {healthError && <div className="notice" role="alert">Backend unreachable: {healthError}</div>}
        {health && (
          <p>
            {health.authenticated ? 'Connected' : 'Not connected'}
            {health.error && !health.authenticated && <span className="muted small"> ({health.error})</span>}
          </p>
        )}
      </section>

      <section className="card">
        <h2>1. Authorise</h2>
        <p className="muted">Open the FYERS login page, sign in, and copy the URL you are redirected to.</p>
        <button type="button" onClick={openLogin}>Open FYERS login</button>
        {loginUrl && <p className="small muted">If no window opened, use this link: <a href={loginUrl} rel="noopener noreferrer">FYERS login</a></p>}
      </section>

      <section className="card">
        <h2>2. Finish</h2>
        <form onSubmit={submit}>
          <label htmlFor="redirected">Redirected URL or auth code</label>
          <textarea
            id="redirected"
            value={redirected}
            onChange={(e) => setRedirected(e.target.value)}
            rows={3}
            autoComplete="off"
            spellCheck={false}
            placeholder="https://trade.fyers.in/api-login/redirect-uri/index.html?auth_code=…"
          />
          <button type="submit" disabled={busy || !redirected.trim()}>
            {busy ? 'Connecting…' : 'Connect'}
          </button>
        </form>
        {message && <p role="status" className="small">{message}</p>}
      </section>
    </section>
  )
}
