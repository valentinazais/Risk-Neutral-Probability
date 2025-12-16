# Risk-Neutral PDF from Options IV Smile

Extracts risk-neutral density (PDF) from SOFI 1M call chain using Breeden-Litzenberger (∂²C/∂K²). Aligns ATM to E[ST]. Plots PNL + PDF, animates cumulative sweep (GIF output).

## Overview
- Input: SOFI calls (2025-09-25, S=28.70, τ=1/12y, r=4.09%).
- Computes IV smile → smooth spline → dense calls → PDF.
- ATM = ∫ST pdf(ST) dST (forward-aligned).
- PNL: max(ST-K,0) - premium.
- Static: PNL top, PDF bottom.
- Animation: Line sweeps left→right, fills PDF (blue), PNL gains/losses (green/red).

## Outputs
- Console: ATM strike (~28.xx), premium (~0.xx).
- `dynamic_probability_pnl.gif`: 200ms/frame sweep (subsampled).

## Steps
1. Load strikes/calls.
2. IV via Newton-BS solver.
3. Gaussian smooth + CubicSpline.
4. Dense pricing.
5. Numerical 2nd deriv → PDF (normalize).
6. E[ST] → ATM K.
7. Interp premium → PNL.
8. Plot + animate fills.

## Run
```bash
pip install numpy pandas scipy matplotlib
python script.py
