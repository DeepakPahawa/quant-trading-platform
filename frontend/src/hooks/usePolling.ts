import { useCallback, useEffect, useRef, useState } from 'react'

export interface PollState<T> {
  data: T | undefined
  error: unknown
  loading: boolean
  reload: () => void
}

/**
 * Runs `load` immediately, then every `intervalMs`. Keeps the last good data on failure.
 * Restarts when `key` changes (for example, a new symbol) and stops on unmount.
 */
export function usePolling<T>(load: () => Promise<T>, intervalMs: number, key: string): PollState<T> {
  const [data, setData] = useState<T | undefined>(undefined)
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(true)
  const [tick, setTick] = useState(0)
  const loadRef = useRef(load)

  useEffect(() => {
    loadRef.current = load
  })

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    const run = async () => {
      try {
        const result = await loadRef.current()
        if (!cancelled) {
          setData(result)
          setError(null)
        }
      } catch (err) {
        if (!cancelled) setError(err)
      } finally {
        if (!cancelled) {
          setLoading(false)
          timer = setTimeout(run, intervalMs)
        }
      }
    }

    void run()
    return () => {
      cancelled = true
      if (timer !== undefined) clearTimeout(timer)
    }
  }, [intervalMs, key, tick])

  const reload = useCallback(() => setTick((t) => t + 1), [])
  return { data, error, loading, reload }
}
