import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ForecastPanel } from '@/components/ForecastPanel'
import { bandPosition, DISCLAIMER } from '@/lib/forecast'
import type { ForecastResponse } from '@/types/api'

const ready: ForecastResponse = {
  status: 'READY',
  symbol: 'NSE:SBIN-EQ',
  timestamp: 1772422500,
  current_price: 800,
  p10: 795,
  p50: 801,
  p90: 808,
  directional_bias: 'BULLISH',
  buffer_count: null,
  required_bars: null,
  horizons: {
    't+5m': {
      p10: 797,
      p50: 800.5,
      p90: 803,
      p10_return: -0.001,
      p50_return: 0.0006,
      p90_return: 0.0037,
      directional_bias: 'BULLISH',
    },
  },
}

describe('bandPosition', () => {
  it('places the median inside the band and clamps outside it', () => {
    expect(bandPosition(5, 0, 10)).toBe(50)
    expect(bandPosition(-1, 0, 10)).toBe(0)
    expect(bandPosition(11, 0, 10)).toBe(100)
  })

  it('centres a degenerate band', () => {
    expect(bandPosition(5, 5, 5)).toBe(50)
  })
})

describe('ForecastPanel', () => {
  it('shows buffering progress without a band', () => {
    render(
      <ForecastPanel
        forecast={{ ...ready, status: 'BUFFERING', buffer_count: 12, required_bars: 60, horizons: null }}
      />,
    )
    expect(screen.getByText(/Collecting bars: 12 of 60/)).toBeInTheDocument()
    expect(screen.queryByLabelText(/band from/)).not.toBeInTheDocument()
  })

  it('renders each horizon band with a bias label and the disclaimer', () => {
    render(<ForecastPanel forecast={ready} />)
    expect(screen.getByText('t+5m')).toBeInTheDocument()
    expect(screen.getByText('▲ Bullish')).toBeInTheDocument()
    expect(screen.getByLabelText(/t\+5m band from ₹797.00 to ₹803.00/)).toBeInTheDocument()
    expect(screen.getByText(DISCLAIMER)).toBeInTheDocument()
  })
})
