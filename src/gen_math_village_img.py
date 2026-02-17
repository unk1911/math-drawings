"""
Generate the snow village image from the mathematical formula in formula.tex.
Computes a 2000x1200 image where each pixel's RGB is defined by H_0, H_1, H_2.
Uses NumPy vectorization: loops over s (houses/terrain terms), not over pixels.
"""

import os
import numpy as np
from PIL import Image
import time

W_IMG, H_IMG = 2000, 1200

print(f"Generating {W_IMG}x{H_IMG} image...")
t0 = time.time()

# Pixel coordinate grids
m = np.arange(1, W_IMG + 1, dtype=np.float64)   # shape (W,)
n = np.arange(1, H_IMG + 1, dtype=np.float64)   # shape (H,)

# Normalized coordinates: x in [-5/3, 5/3], y in [-5/6, 5/6] approx
# x[m] = (m-1000)/600, y[n] = (501-n)/600
# Broadcast to (H, W)
x = ((m - 1000) / 600)[np.newaxis, :]   # (1, W)
y = ((501 - n) / 600)[:, np.newaxis]    # (H, 1)


# ── Helper ──────────────────────────────────────────────────────────────────

def safe_exp(z):
    """Clamp exponent to avoid overflow in exp()."""
    return np.exp(np.clip(z, -700, 700))


# ── E(x,y): fractal terrain texture ─────────────────────────────────────────

print("  Computing E...")
E = np.zeros((H_IMG, W_IMG), dtype=np.float64)
for s in range(1, 41):
    coeff = (24 / 25) ** s
    scale = (6 / 5) ** s * 10
    cs2 = np.cos(s ** 2)
    ss2 = np.sin(s ** 2)
    arg1 = scale * (-cs2 * x + ss2 * y + 2 * np.cos(17 * s)) + 2 * np.cos(5 * s)
    arg2 = scale * ( cs2 * y + ss2 * x + 2 * np.cos(15 * s)) + 2 * np.cos(7 * s)
    E += coeff * np.cos(arg1) * np.cos(arg2)


# ── B(x,y): background / ground mask ────────────────────────────────────────

print("  Computing B...")
log_B = np.zeros((H_IMG, W_IMG), dtype=np.float64)
for s in range(1, 41):
    r = (49 / 50) ** s
    q = (107 / 100) ** s * 30
    denom = 3 - 2 * y                                  # (H, 1) broadcast
    base_x = (2 * x + 2 * np.cos(5 * s ** 2)) / denom
    base_y = 2 / denom

    ang = q * (np.cos(15 * s ** 2) * base_x + np.sin(15 * s ** 2) * base_y)
    ang2 = q * (np.sin(15 * s ** 2) * base_x - np.cos(15 * s ** 2) * base_y)

    inner = (np.cos(ang + 2 * np.cos(17 * s ** 2))
             * np.cos(ang2 + 2 * np.cos(18 * s ** 2))
             + (E - 99) / 100)

    exponent = (28 - 20 * y) * r * inner
    log_B -= safe_exp(np.clip(exponent, -700, 700))

B = safe_exp(log_B)


# ── M(x,y): snow silhouette mask ─────────────────────────────────────────────

print("  Computing M...")
snow_line = (y + x / 15 - x ** 2 / 7
             + np.cos(3 * x + np.cos(2 * x)) / 12
             + 57 / 50)
M = safe_exp(-safe_exp(-50 * (np.abs(snow_line) - 0.1 - E / 200)))


# ── Per-house computation (s = 1..67) ────────────────────────────────────────

print("  Computing houses (67 iterations)...")

# Accumulators for A_v (one per channel v=0,1,2) and Z
A = [np.zeros((H_IMG, W_IMG)) for _ in range(3)]
Z = np.ones((H_IMG, W_IMG))   # Z_0 = 1

for s in range(1, 68):
    if s % 10 == 0:
        print(f"    s={s}/67  ({time.time()-t0:.1f}s elapsed)")

    # ── C_s, V_s, L_s, U_s ──────────────────────────────────────────────────
    C = (400 + s) / (480 - 320 * y)                        # (H, 1)
    V = np.arctan(np.tan(40 * C))
    L = x * C + 0.25 * np.cos(7 * (40 * C - V) ** 2)
    U = np.arctan(np.tan(10 * L))

    # ── Rotated local coords ─────────────────────────────────────────────────
    dU = 10 * L - U
    dV = 40 * C - V
    N = dU ** 2 + dV ** 2
    cosN = np.cos(N)
    sinN = np.sin(N)
    P = cosN * V - sinN * U
    Q = cosN * U + sinN * V
    R = 1 - 0.3 * np.cos(2 * dU ** 2 + 3 * dV ** 2)

    # ── W(s): step function ──────────────────────────────────────────────────
    Ws = float(safe_exp(-safe_exp(50500 - 1000 * s)))  # scalar, near 0 for s<50

    # ── T_s: "is pixel in house s?" ─────────────────────────────────────────
    cos_house = np.cos(6 * dU ** 2 + 8 * dV ** 2)
    cos_roof  = np.cos(2 * dU ** 2 +     dV ** 2)
    roof_dist = (500 * (6 + cos_roof) * dU ** 2
                 + 1000 * (40 * (C - 27/20) - V) ** 2
                 - 1e6)
    T_exp = (-100 * (s - 0.5)
             - safe_exp(1000 * (cos_house - 0.8))
             - safe_exp(np.clip(roof_dist, -700, 700)))
    T = safe_exp(-safe_exp(np.clip(T_exp, -700, 700)))

    # ── J_s: window component ───────────────────────────────────────────────
    J_exp = (-safe_exp(100 * np.abs(Q) - 2 * s * R - 15)
             - safe_exp(100 * (np.abs(P) - R)))
    J = T * (1 - Ws) * safe_exp(J_exp)

    # ── K_s: wall component ──────────────────────────────────────────────────
    wall_exp = (200 * np.abs(Q)
                - (4 * s * (1 - Ws) + (190 + 4 * E) * Ws) * R)
    K_exp = (-safe_exp(wall_exp)
             - safe_exp(100 * np.abs(P) - R * (100 + 2 * E)))
    K = T * safe_exp(K_exp)

    # ── Accumulate A_v for each colour channel ───────────────────────────────
    # Detail mask (strips horizontal bands inside house)
    detail_mask = safe_exp(-safe_exp(
        1e3 * (np.cos(8 * dU ** 2 + 3 * dV ** 2) - 0.4)
    ))
    snow_fade = safe_exp(-safe_exp(
        20 * np.abs(Q) - 6 * R - safe_exp(2 * abs(s - 40) - 10)
    ))
    window_boost = 1 + 6 * snow_fade

    Z_prev = Z.copy()
    contrib_base = (J + K * (1 - J)) * detail_mask * window_boost * ((100 - s) / 100)

    for v in range(3):
        hue = (np.cos(12 * s ** 2) - v + 5) / 20
        A[v] += Z_prev * contrib_base * hue

    # ── Update Z (occlusion) ─────────────────────────────────────────────────
    if s < 67:
        Z *= (1 - J) * (1 - K)

Z67 = Z  # Z_{67}


# ── H_v and F ────────────────────────────────────────────────────────────────

print("  Computing H_v and F...")
channels = []
for v in range(3):
    coeff = (13 * v**2 - 16 * v + 103) / 50000
    H = (0.9 * (1 - B)
         + coeff * B * (10 * A[v] + 9 * Z67)
         * (5 + 2*v + (45 - 2*v) * M))
    # F(x) = floor(255 * exp(-exp(-1000x)) * |x|^exp(-exp(1000(x-1))))
    inner = safe_exp(-1000 * H)
    power_exp = safe_exp(-safe_exp(1000 * (H - 1)))
    F = 255 * safe_exp(-inner) * (np.abs(H) ** power_exp)
    channels.append(np.clip(np.floor(F), 0, 255).astype(np.uint8))

# Stack RGB
img_array = np.stack(channels, axis=2)   # (H, W, 3)


# ── Save ─────────────────────────────────────────────────────────────────────

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "math-village-generated.png")
Image.fromarray(img_array, mode="RGB").save(out_path)
elapsed = time.time() - t0
print(f"\nDone! Saved to {out_path}  ({elapsed:.1f}s)")
