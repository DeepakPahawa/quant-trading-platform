import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// Vitest does not auto-clean the DOM unless globals are enabled.
afterEach(() => {
  cleanup()
})
