import { useEffect, useRef, useCallback } from 'react'
import type { WsMessage } from '../types'

const WS_URL = 'ws://localhost:8000/ws/market'
const HEARTBEAT_INTERVAL = 15_000

type MessageHandler = (msg: WsMessage) => void

/**
 * Singleton WebSocket connection shared across all components in the page.
 * Returns a stable `subscribe` function that accepts a message handler.
 */
let ws: WebSocket | null = null
const handlers = new Set<MessageHandler>()
let heartbeatTimer: ReturnType<typeof setInterval> | null = null

function ensureConnected(): void {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return

  ws = new WebSocket(WS_URL)

  ws.onopen = () => {
    ws!.send(JSON.stringify({ action: 'subscribe', symbols: ['NSE:NIFTY50-INDEX'] }))
    heartbeatTimer = setInterval(() => {
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'ping', timestamp: Date.now() }))
      }
    }, HEARTBEAT_INTERVAL)
  }

  ws.onmessage = (ev: MessageEvent<string>) => {
    try {
      const msg = JSON.parse(ev.data) as WsMessage
      handlers.forEach(h => h(msg))
    } catch {
      // ignore malformed frames
    }
  }

  ws.onclose = () => {
    if (heartbeatTimer) clearInterval(heartbeatTimer)
    ws = null
    // Reconnect after 2 s if any handlers still active
    if (handlers.size > 0) setTimeout(ensureConnected, 2000)
  }

  ws.onerror = () => ws?.close()
}

export function useMarketWebSocket(onMessage: MessageHandler): void {
  const handlerRef = useRef<MessageHandler>(onMessage)
  handlerRef.current = onMessage

  const stable = useCallback((msg: WsMessage) => handlerRef.current(msg), [])

  useEffect(() => {
    handlers.add(stable)
    ensureConnected()
    return () => {
      handlers.delete(stable)
      if (handlers.size === 0) {
        ws?.close()
        ws = null
      }
    }
  }, [stable])
}

