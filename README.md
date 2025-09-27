Breeden-Litzenberger PDF Extractor: Predicting Stock Prices from Option Prices

This repository contains a Python implementation of the Breeden-Litzenberger (BL) formula to extract the risk-neutral probability density function (PDF) from option prices. The goal is to explore whether stock prices can be "predicted" (or at least implied) from market option data, revealing market sentiment, tail risks, and distribution shapes. This is inspired by quantitative finance techniques for risk management, arbitrage detection, and sentiment analysis.

The code processes real option data (e.g., for SOFI stock), computes implied volatilities, interpolates and smooths data, applies the BL formula, calculates expected terminal prices, and visualizes PNL alongside the PDF with an animated GIF.

Important Note: This extracts a risk-neutral PDF for pricing purposes (under a no-arbitrage equilibrium). It does not represent real-world (risk-averse) probabilities. Be cautious not to confuse the two—risk-neutral PDFs embed market fears and premia, often exaggerating tails compared to actual outcomes. For real-world forecasting, adjust for risk premia using additional models.

Background 

Can We Predict Stock Prices from Option Prices?
As discussed in the post : https://www.linkedin.com/feed/update/urn:li:activity:7377265675808972800/

This code implements the "deeper step-by-step process" mentioned in the post, using SOFI stock as an example (with 1M tenor calls as of 2025-09-25).

Key Concepts

- Risk-Neutral PDF: Derived from option prices, assuming a world where assets grow at the risk-free rate (no risk premia). Useful for pricing and hedging.
- Applications: Infer market-implied tails/skew, improve VaR models, detect sentiment (e.g., crash fears), or align with ATM calls for PNL analysis.
- Limitations: Handles issues like low liquidity via smoothing/interpolation, but results depend on data quality. Not for direct real-world predictions without adjustments.

Code Overview 

The script performs the following steps:

- Input Data: Latest stock price (S), strikes, call prices, time to expiry (τ), and risk-free rate (r).
- Implied Volatility Calculation: Uses Black-Scholes and Newton's method.
- Smoothing & Interpolation: Gaussian filter and cubic splines for robust IV smile.
- BL Formula Application: Computes second derivative of interpolated calls to get the PDF.
- Normalization & Expectations: Normalizes PDF and computes E[ST] (expected terminal price).
- PNL Calculation: For an ATM call aligned with E[ST].
- Visualization: Static plots of PNL and PDF, plus an animated GIF showing cumulative probabilities and PNL fills (blue for PDF, green/red for positive/negative PNL).

Output:

- Console prints (e.g., ATM strike, premium).
- Static plot.
- Animated GIF: dynamic_probability_pnl.gif (vertical line sweeps, filling areas cumulatively).
- Example Animation Preview (run the code to generate):
- Dynamic Probability PNL GIF
