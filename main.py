import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import newton
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

# Step 1: Update current S (latest price) from real-time data
S = 28.70  # Updated SOFI price as of 2025-09-25

# Step 2: Define strikes and calls (1M tenor)  from user input
strikes = np.array([21.5, 22, 22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 30, 30.5, 31, 32, 33, 34, 35, 36])
calls = np.array([5.82, 5.35, 4.85, 4.35, 3.87, 3.5, 3, 2.56, 2.08, 1.69, 1.3, 0.98, 0.71, 0.51, 0.36, 0.26, 0.18, 0.14, 0.09, 0.08, 0.05, 0.03, 0.03, 0.02, 0.01])

# Step 3: Define parameters 
tau = 1 / 12  # 1M to expiry in years
r = 0.0409  # 1M US Treasury rate as of September 2025

# Step 4: Black-Scholes call price function 
def black_scholes_call(S, K, tau, r, sigma):
    if tau == 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

# Step 5: Implied volatility using Newton's method 
def implied_volatility(call_price, S, K, tau, r, initial_guess=0.2):
    def objective(sigma):
        return black_scholes_call(S, K, tau, r, sigma) - call_price
    try:
        return newton(objective, initial_guess, tol=1e-6, maxiter=100)
    except:
        return np.nan  # Return NaN for convergence failure

# Step 6: Compute IV for each strike - 
ivs = np.array([implied_volatility(c, S, k, tau, r) for c, k in zip(calls, strikes)])

# Step 7: Filter out NaN and clip deep OTM/ITM (IV > 2 or < 0) 
valid_mask = ~np.isnan(ivs) & (ivs > 0) & (ivs < 2)
strikes_valid = strikes[valid_mask]
ivs_valid = ivs[valid_mask]

# Step 8: Smooth IV data with Gaussian filter 
ivs_smoothed = gaussian_filter1d(ivs_valid, sigma=5)

# Step 9: Interpolate in IV space using CubicSpline 
spline = CubicSpline(strikes_valid, ivs_smoothed)

# Step 10: Generate dense strikes - 
dense_strikes = np.linspace(strikes.min() - 5, strikes.max() + 5, 1000)

# Step 11: Compute continuous calls from interpolated IV 
dense_calls = np.array([black_scholes_call(S, k, tau, r, spline(k)) for k in dense_strikes])

# Step 12: Compute second derivative of C with respect to K (numerical) 
first_deriv = np.gradient(dense_calls, dense_strikes)
second_deriv = np.gradient(first_deriv, dense_strikes)

# Step 13: Apply Breeden-Litzenberger formula for PDF 
pdf = np.exp(r * tau) * second_deriv

# Step 14: Normalize PDF to integrate to 1 
integral = np.trapz(pdf, dense_strikes)
pdf_normalized = pdf / integral if integral != 0 else pdf

# This is E[ST] = ∫ ST * pdf(ST) dST ≈ sum(dense_strikes * pdf_normalized * delta_strike), but using trapz for accuracy
expected_st = np.trapz(dense_strikes * pdf_normalized, dense_strikes)

# Set atm_strike to this expected value (to avoid risk-free arbitrage, as it aligns with the forward price under RN measure)
atm_strike = expected_st
print(f"Computed ATM Strike (E[ST] under RN): {atm_strike:.2f}")

# Compute the market-implied premium (call price) at this atm_strike using interpolation from dense_calls
call_interp = interp1d(dense_strikes, dense_calls, kind='cubic', fill_value='extrapolate')
premium = call_interp(atm_strike)
print(f"Interpolated Premium at K={atm_strike:.2f}: {premium:.2f}")

# Compute PNL: intrinsic payoff minus premium
pnl_full = np.maximum(dense_strikes - atm_strike, 0) - premium

# New: Create figure with two subplots (PNL above PDF)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

# Plot PNL on top subplot
ax1.plot(dense_strikes, pnl_full, 'k-', label=f'Call PNL (K={atm_strike:.2f}, Premium={premium:.2f})')
ax1.axhline(0, color='gray', linestyle='--')
ax1.fill_between(dense_strikes, pnl_full, 0, where=(pnl_full > 0), facecolor='green', alpha=0.3, interpolate=True)
ax1.fill_between(dense_strikes, pnl_full, 0, where=(pnl_full < 0), facecolor='red', alpha=0.3, interpolate=True)
ax1.set_ylabel('PNL')
ax1.legend()
ax1.grid(True)
ax1.set_title('Call PNL Aligned with Strike Prices')

# Plot PDF on bottom subplot (blue curve)
ax2.plot(dense_strikes, pdf_normalized, 'b-', label='Risk-Neutral PDF')
ax2.set_xlabel('Strike Price (or Terminal Price ST)')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# Animation: Vertical line moves left to right, filling PDF area cumulatively (blue fill),
# and filling PNL areas (red below 0, green above 0 up to the line)

# Prepare figure for animation (separate from static plot)
fig_anim, (ax1_anim, ax2_anim) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

# Initial empty plots
pnl_line, = ax1_anim.plot(dense_strikes, pnl_full, 'k-')
ax1_anim.axhline(0, color='gray', linestyle='--')
ax1_anim.set_ylabel('Payoff')
ax1_anim.set_title('Dynamic Probability and PNL Visualization')
ax1_anim.set_ylim(pnl_full.min() - 1, max(pnl_full.max(), 1) + 1)  # Adjust y-limits for visibility

pdf_line, = ax2_anim.plot(dense_strikes, pdf_normalized, 'b-')
ax2_anim.set_xlabel('Strike Price (or Stock Price at Maturity)')
ax2_anim.set_ylabel('Density')
ax2_anim.set_xlim(dense_strikes.min(), dense_strikes.max())
ax2_anim.set_ylim(0, pdf_normalized.max() * 1.1)

# Initialize elements for animation
vline = ax2_anim.axvline(x=dense_strikes.min(), color='black', linestyle='--')

# PDF fill (blue cumulative)
pdf_fill = Polygon([[dense_strikes.min(), 0]], closed=True, facecolor='blue', alpha=0.3)
ax2_anim.add_patch(pdf_fill)

# PNL fills (red for negative, green for positive, cumulative)
red_fill = Polygon([[dense_strikes.min(), 0]], closed=False, facecolor='red', alpha=0.3)
green_fill = Polygon([[dense_strikes.min(), 0]], closed=False, facecolor='green', alpha=0.3)
ax1_anim.add_patch(red_fill)
ax1_anim.add_patch(green_fill)

def init():
    vline.set_xdata([dense_strikes.min()] * 2)
    pdf_fill.set_xy([[dense_strikes.min(), 0], [dense_strikes.min(), 0]])
    red_fill.set_xy([[dense_strikes.min(), 0]])
    green_fill.set_xy([[dense_strikes.min(), 0]])
    return vline, pdf_fill, red_fill, green_fill

def animate(i):
    # Current position
    x_pos = dense_strikes[i]
    vline.set_xdata([x_pos] * 2)
    
    # Cumulative data up to i
    idx = i + 1
    x_cum = dense_strikes[:idx]
    y_pdf = pdf_normalized[:idx]
    y_pnl = pnl_full[:idx]
    
    # Update PDF fill
    xy_pdf = np.c_[x_cum, y_pdf]
    xy_pdf = np.r_[xy_pdf, [[x_cum[-1], 0], [x_cum[0], 0]]]
    pdf_fill.set_xy(xy_pdf)
    
    # Update red fill (where y_pnl < 0, between y_pnl and 0)
    red_mask = y_pnl < 0
    if np.any(red_mask):
        x_red = x_cum[red_mask]
        y_red = y_pnl[red_mask]
        # For fill: lower is y_red, upper is 0
        xy_red_lower = np.c_[x_red, y_red]
        xy_red_upper = np.c_[x_red[::-1], np.zeros_like(x_red[::-1])]  # Reverse for closing
        xy_red = np.r_[xy_red_lower, xy_red_upper]
        red_fill.set_xy(xy_red)
    else:
        red_fill.set_xy([[x_cum[0], 0]])
    
    # Update green fill (where y_pnl > 0, between 0 and y_pnl)
    green_mask = y_pnl > 0
    if np.any(green_mask):
        x_green = x_cum[green_mask]
        y_green = y_pnl[green_mask]
        # For fill: lower is 0, upper is y_green
        xy_green_lower = np.c_[x_green, np.zeros_like(x_green)]
        xy_green_upper = np.c_[x_green[::-1], y_green[::-1]]  # Reverse for closing
        xy_green = np.r_[xy_green_lower, xy_green_upper]
        green_fill.set_xy(xy_green)
    else:
        green_fill.set_xy([[x_cum[0], 0]])
    
    return vline, pdf_fill, red_fill, green_fill

# Create animation (subsample for speed)
frames = np.arange(0, len(dense_strikes), 5)
anim = FuncAnimation(fig_anim, animate, init_func=init, frames=frames, interval=50, blit=True)

# Save as GIF
anim.save('dynamic_probability_pnl.gif', writer='pillow')

# Display message
print("Animated GIF saved as 'dynamic_probability_pnl.gif'")
