---
name: quantlib-vol-surface
description: Inverts option market prices using QuantLib-Python to calculate Implied Volatility (IV), compute Delta/Gamma/Theta/Vega, and calibrate SVI curves for Nifty/BankNifty options. Use for options pricing and volatility smile modeling.
---
# QuantLib Volatility Surface Protocol

### Objective
Construct an arbitrage-free volatility surface from live or historical Indian options chains.

### Execution Directives
1. **IV Solver (`backend/app/models/volatility.py`):**
   - Use `QuantLib.EuropeanOption` with the `QuantLib.BlackScholesMertonProcess` or Black-76 engine.
   - Invert market mid-prices to extract Black-Scholes Implied Volatility (IV) for call and put strikes.
   - Compute Greeks: Delta ($\Delta$), Gamma ($\Gamma$), Theta ($\Theta$), Vega ($\nu$).

2. **Surface Fitting:**
   - Map strikes into log-moneyness: $k = \ln(K / F)$, where $F$ is forward spot.
   - Fit an SVI (Stochastic Volatility Inspired) curve across moneyness per expiry slice.
   - Discard option quotes where bid-ask spread exceeds 15% of the mid-price to eliminate illiquid noise.