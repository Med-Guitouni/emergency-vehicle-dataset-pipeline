"""
calibrate_picker.py

Step 2-4 of the calibration protocol. Interactive point-picking on a chosen
frame to solve for the real camera parameters: horizon_ratio, cx, focal_px,
camera_height.

Run once per chosen frame:
    python3 calibrate_picker.py 5
    python3 calibrate_picker.py 90

WHAT YOU WILL CLICK, IN ORDER (a window opens for each step):

  STEP A - VANISHING POINT (2 clicks per lane line, 2 lines = 4 clicks)
    Click two points along the LEFT lane line (near and far), then two points
    along the RIGHT lane line (near and far) of the SAME lane (the one the
    ego is in, or the one with the clearest dashes).
    -> the two lines are extended and intersected to find the true horizon
       row and column (cx).

  STEP B - LANE WIDTH (2 clicks: left edge, right edge, AT THE SAME ROW)
    Pick a row close to the bottom of the frame (near = more reliable).
    Click the left lane line and the right lane line of the SAME lane at
    that row.
    -> known real width 3.75m (RAA, BAB heavy-vehicle lane) solves focal_px.

  STEP C - DASH RULER (click the START of each visible dash, near to far)
    Click the near-end (closer to ego) of each white dash you can see,
    in order from nearest to farthest. Each dash START is exactly 18m
    (6m stripe + 12m gap) from the previous dash START (RMS standard).
    Click at least 4 dashes for a usable fit.
    -> solves camera_height given focal_px and horizon_row from steps A/B.

Close each plot window after clicking (or press Enter in the terminal) to
move to the next step. Results are printed AND saved to calibration_log.json
so multiple frames can be combined.

Requires: matplotlib (already on your system via earlier installs)
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

LANE_WIDTH_M   = 3.75     # RAA heavy-vehicle lane width, BAB
DASH_PERIOD_M  = 18.0     # RMS: 6m stripe + 12m gap, Autobahn
LOG_FILE       = "calibration_log.json"

# current pipeline constants, for comparison
CURRENT_HORIZON_RATIO = 0.55
CURRENT_FOCAL_FACTOR  = 0.8
CURRENT_CAM_HEIGHT    = 1.4


def click_points(img, n_points, title, instructions):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{title}\n{instructions}", fontsize=10)
    pts = plt.ginput(n_points, timeout=0)
    plt.close(fig)
    return [(float(x), float(y)) for x, y in pts]


def line_intersection(p1, p2, p3, p4):
    """Intersection of line p1-p2 and line p3-p4."""
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return px, py


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 calibrate_picker.py <timestamp_seconds>")
    ts = int(sys.argv[1])

    img_path = os.path.join("calibration_candidates", f"t{ts:04d}.jpg")
    if not os.path.exists(img_path):
        raise SystemExit(f"Frame not found: {img_path} (run extract_candidate_frames.py first)")

    img = cv2.imread(img_path)
    ih, iw = img.shape[:2]
    print(f"Frame t={ts}s   size={iw}x{ih}")

    # ---------------- STEP A: vanishing point ----------------
    print("\nSTEP A — click 2 points on the LEFT lane line (near, then far),")
    print("         then 2 points on the RIGHT lane line (near, then far).")
    print("\nSTEP A — click 2 points on EACH of 3 lane lines (near, then far).")
    print("         Use 3 DIFFERENT lines if visible (e.g. left edge, the dash")
    print("         line left-of-ego-lane, the dash line right-of-ego-lane).")
    print("         For each line: click as close to the BOTTOM of the frame")
    print("         as you can for 'near', and as far up that SAME line as you")
    print("         can still confidently place it for 'far'. A bigger gap")
    print("         between near/far makes the line direction far more accurate.")
    N_LINES = 3
    all_pts = click_points(
        img, N_LINES * 2, f"t={ts}s STEP A: vanishing point (3 lines)",
        f"For each of {N_LINES} lines click NEAR then FAR. "
        f"({N_LINES*2} clicks total, line by line)"
    )
    lines = [(all_pts[2*i], all_pts[2*i+1]) for i in range(N_LINES)]

    # all pairwise intersections - should all agree if clicks are good
    from itertools import combinations
    intersections = []
    for (p1, p2), (p3, p4) in combinations(lines, 2):
        ip = line_intersection(p1, p2, p3, p4)
        if ip is not None:
            intersections.append(ip)

    if len(intersections) < 2:
        raise SystemExit("Lines too parallel to intersect reliably — redo with "
                          "more separated near/far points.")

    ix = np.array([p[0] for p in intersections])
    iy = np.array([p[1] for p in intersections])
    vp_x, vp_y = float(np.median(ix)), float(np.median(iy))
    spread_x, spread_y = float(np.std(ix)), float(np.std(iy))

    print(f"\n  Pairwise intersections ({len(intersections)}):")
    for (x, y) in intersections:
        print(f"    ({x:.1f}, {y:.1f})")
    print(f"  Median vanishing point: ({vp_x:.1f}, {vp_y:.1f})"
          f"   spread: ({spread_x:.1f}, {spread_y:.1f})px")

    # SANITY CHECK - refuse to proceed on an obviously broken vp
    if not (0 <= vp_y <= ih * 1.3) or not (-0.5*iw <= vp_x <= 1.5*iw):
        raise SystemExit(
            f"\n  VANISHING POINT IMPLAUSIBLE: ({vp_x:.1f}, {vp_y:.1f}) for a "
            f"{iw}x{ih} frame.\n  Expected roughly inside or just above the "
            f"frame, near horizontal centre.\n  This usually means near/far "
            f"points on a line were too close together, or a line's near/far "
            f"order was reversed.\n  REDO Step A — space the near/far points "
            f"on each line as far apart as possible."
        )
    if max(spread_x, spread_y) > 0.15 * max(iw, ih):
        print(f"\n  WARNING: pairwise intersections disagree by up to "
              f"{max(spread_x, spread_y):.0f}px — one of the 3 lines was likely "
              f"clicked imprecisely. Results below use the median (more robust "
              f"than 2-line), but consider redoing if this run's combine_calibration"
              f" disagrees with other frames.")

    measured_horizon_ratio = vp_y / ih
    measured_cx_ratio      = vp_x / iw
    print(f"\n  -> horizon_ratio = {measured_horizon_ratio:.4f}  "
          f"(current code: {CURRENT_HORIZON_RATIO})")
    print(f"  -> cx            = {vp_x:.1f}px  "
          f"(current code assumes width/2 = {iw/2:.1f}px)")

    # ---------------- STEP B: lane width ----------------
    print("\nSTEP B — click the LEFT edge then the RIGHT edge of ONE lane,")
    print("         at the SAME row, as close to the bottom as you can.")
    pts_b = click_points(
        img, 2, f"t={ts}s STEP B: lane width",
        "Click left edge of lane, then right edge of SAME lane, same row. (2 clicks)"
    )
    (lx, ly), (rx, ry) = pts_b
    row_b = (ly + ry) / 2.0
    lane_width_px = abs(rx - lx)
    print(f"  Lane width: {lane_width_px:.1f}px at row {row_b:.1f}")

    # ---------------- STEP C: dash ruler ----------------
    print("\nSTEP C — click the NEAR (closer-to-ego) end of each visible dash,")
    print("         in order from nearest to farthest. At least 4 dashes.")
    n_dashes = int(input("How many dashes can you clearly see? (>=4 recommended): "))
    pts_c = click_points(
        img, n_dashes, f"t={ts}s STEP C: dash ruler",
        f"Click the NEAR end of each dash, nearest to farthest ({n_dashes} clicks)"
    )

    # Solve for nearest-dash distance y0 and product K = camera_height * focal_px
    # using: delta_k = bottom_y_k - horizon_row = K / (y0 + 18*k)
    horizon_row_px = vp_y
    deltas = np.array([y - horizon_row_px for (_, y) in pts_c])
    k_idx  = np.arange(len(pts_c))   # 0,1,2,... nearest to farthest

    if np.any(deltas <= 0):
        print("  WARNING: some dash clicks are above the horizon — check your clicks.")

    from scipy.optimize import least_squares

    def residuals(params):
        y0, K = params
        y_model = y0 + DASH_PERIOD_M * k_idx
        delta_model = K / y_model
        return delta_model - deltas

    f0 = iw * CURRENT_FOCAL_FACTOR
    K0 = CURRENT_CAM_HEIGHT * f0
    y0_0 = K0 / max(deltas[0], 1.0)
    sol = least_squares(residuals, x0=[y0_0, K0])
    y0_fit, K_fit = sol.x
    resid_final = residuals(sol.x)

    print(f"\n  Dash fit: nearest dash distance y0 = {y0_fit:.2f}m"
          f"  (camera_height * focal_px) = {K_fit:.1f}")
    print(f"  Residuals (px): {np.round(resid_final, 2)}")
    print(f"  Max residual: {np.max(np.abs(resid_final)):.2f}px "
          f"({'OK' if np.max(np.abs(resid_final)) < 5 else 'HIGH - check clicks/curve/grade'})")

    # ---------------- combine B + C to solve f and camera_height ----------------
    delta_b = row_b - horizon_row_px
    y_at_b  = K_fit / delta_b if delta_b > 0 else float("nan")
    f_fit = lane_width_px * y_at_b / LANE_WIDTH_M if y_at_b == y_at_b else float("nan")
    cam_h_fit = K_fit / f_fit if f_fit == f_fit and f_fit != 0 else float("nan")

    print(f"\n{'='*55}")
    print(f"RESULTS for t={ts}s")
    print(f"{'='*55}")
    print(f"  horizon_ratio   = {measured_horizon_ratio:.4f}   (current: {CURRENT_HORIZON_RATIO})")
    print(f"  cx              = {vp_x:.1f}px  ({measured_cx_ratio:.4f} of width)  "
          f"(current: {iw/2:.1f}px = 0.5000)")
    print(f"  focal_px        = {f_fit:.1f}px  "
          f"(current: {f0:.1f}px = width*{CURRENT_FOCAL_FACTOR})")
    if f_fit == f_fit and f_fit != 0:
        print(f"  focal_factor    = {f_fit/iw:.4f}   (current: {CURRENT_FOCAL_FACTOR})")
    print(f"  camera_height   = {cam_h_fit:.3f}m   (current: {CURRENT_CAM_HEIGHT}m)")
    print(f"  forward dist at lane-width row (y={row_b:.0f}px): {y_at_b:.2f}m")
    print(f"{'='*55}\n")

    # ---------------- save to log ----------------
    entry = {
        "timestamp": ts, "frame_w": iw, "frame_h": ih,
        "vp_x": vp_x, "vp_y": vp_y,
        "horizon_ratio": measured_horizon_ratio,
        "cx_ratio": measured_cx_ratio,
        "lane_width_px": lane_width_px, "lane_width_row": row_b,
        "n_dashes": n_dashes, "dash_y0_m": y0_fit, "dash_K": K_fit,
        "max_dash_residual_px": float(np.max(np.abs(resid_final))),
        "focal_px_fit": f_fit, "focal_factor_fit": (f_fit / iw if f_fit == f_fit else None),
        "camera_height_fit": cam_h_fit,
    }
    log = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            log = json.load(f)
    log = [e for e in log if e.get("timestamp") != ts]  # replace if re-run
    log.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved to {LOG_FILE} ({len(log)} frame(s) logged so far)")
    if len(log) >= 2:
        print("Run combine_calibration.py to average across frames and get the final answer.")


if __name__ == "__main__":
    main()