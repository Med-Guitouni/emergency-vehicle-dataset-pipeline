import numpy as np


class RTSSmoother:
    """
     Runs AFTER tracking is complete,
    BEFORE metrics are computed.

    WHY
    -----------------
    Our raw positions come from projecting the bounding-box bottom pixel to
    the road. Two noise sources corrupt them:
      1. Detection jitter - the bbox edge wobbles a few pixels per frame,
         worst for clipped boxes (vehicles beside the ambulance).
      2. Pixel quantisation - at 25m one pixel equals ~1m of distance, so far
         vehicles snap between discrete positions (e.g. 24.46 <-> 25.33).
    Both turn into fake velocity, fake acceleration and fake jerk, which can
    trigger wrong behaviour labels.

    A Kalman filter alone only looks BACKWARD in time (it cannot use future
    frames). RTS adds a second, backward pass over the finished trajectory:
    every position estimate is corrected using what happened AFTER it. The
    result is the statistically optimal smooth trajectory given all frames.
    This is the same method highD (Krajewski et al. 2018) and INTERACTION
    (Zhan et al. 2019)(critian paper) use before publishing their trajectories.


    -----------------
    Measurements flagged position_reliable=False (clipped bounding boxes) get
    a much larger measurement noise R, so the smoother trusts the motion model
    more than the bad measurement there. This is what tames the +-3m oscillation
    of trucks driving right beside the ambulance.

    MODEL
    -----
    Per track, per axis (x and y independently): constant-velocity model,
    state [position, velocity]. Handles gaps in a track naturally by using
    the real time difference dt between consecutive observations.
    """

    def __init__(self,
                 process_noise=1.0,        # q: how much we allow velocity to wander (m^2/s^3)
                 meas_noise_reliable=1.0,  # R for trusted positions (std ~1m)
                 meas_noise_unreliable=25.0):  # R for clipped boxes (std ~5m)
        self.q      = process_noise
        self.R_rel  = meas_noise_reliable
        self.R_unrel = meas_noise_unreliable

    def _smooth_axis(self, times, values, reliable):
        """
        Forward Kalman filter + backward RTS pass on one axis of one track.
        times:    list of timestamps (seconds), strictly increasing
        values:   measured positions (metres)
        reliable: list of bools (position_reliable per observation)
        Returns the smoothed positions (same length).
        """
        n = len(values)
        H = np.array([[1.0, 0.0]])

        x = np.array([values[0], 0.0])       # initial state: first position, zero velocity
        P = np.diag([self.R_rel, 10.0])      # uncertain about initial velocity

        xs_pred, Ps_pred = [], []            # predicted (prior) per step
        xs_filt, Ps_filt = [], []            # filtered (posterior) per step

        for k in range(n):
            if k == 0:
                x_pred, P_pred = x.copy(), P.copy()
            else:
                dt = max(times[k] - times[k - 1], 1e-6)
                F = np.array([[1.0, dt], [0.0, 1.0]])
                # white-noise acceleration process noise
                Q = self.q * np.array([[dt**3 / 3, dt**2 / 2],
                                       [dt**2 / 2, dt]])
                x_pred = F @ x
                P_pred = F @ P @ F.T + Q

            R = self.R_rel if reliable[k] else self.R_unrel
            innovation = values[k] - x_pred[0]
            S = P_pred[0, 0] + R
            K = P_pred[:, 0] / S
            x = x_pred + K * innovation
            P = P_pred - np.outer(K, P_pred[0, :])

            xs_pred.append(x_pred); Ps_pred.append(P_pred)
            xs_filt.append(x);      Ps_filt.append(P)

        # backward RTS pass - correct each estimate using the future
        xs_smooth = [None] * n
        xs_smooth[-1] = xs_filt[-1]
        P_smooth_next = Ps_filt[-1]
        for k in range(n - 2, -1, -1):
            dt = max(times[k + 1] - times[k], 1e-6)
            F = np.array([[1.0, dt], [0.0, 1.0]])
            C = Ps_filt[k] @ F.T @ np.linalg.inv(Ps_pred[k + 1])
            xs_smooth[k] = xs_filt[k] + C @ (xs_smooth[k + 1] - xs_pred[k + 1])
            P_smooth_next = Ps_filt[k] + C @ (P_smooth_next - Ps_pred[k + 1]) @ C.T

        return [float(s[0]) for s in xs_smooth]

    def smooth(self, track_observations):
        """
        track_observations: dict
            track_id -> list of (timestamp, x_m, y_m, position_reliable),
            sorted by timestamp.

        Returns: dict
            track_id -> dict { timestamp -> (x_smoothed, y_smoothed) }

        Tracks with fewer than 3 observations are passed through unchanged -
        there is not enough information to smooth them meaningfully.
        """
        out = {}
        for tid, obs in track_observations.items():
            if len(obs) < 3:
                out[tid] = {t: (x, y) for (t, x, y, r) in obs}
                continue

            times    = [o[0] for o in obs]
            xs       = [o[1] for o in obs]
            ys       = [o[2] for o in obs]
            reliable = [o[3] for o in obs]

            xs_s = self._smooth_axis(times, xs, reliable)
            ys_s = self._smooth_axis(times, ys, reliable)

            out[tid] = {
                t: (round(xv, 2), round(max(yv, 0.0), 2))
                for t, xv, yv in zip(times, xs_s, ys_s)
            }
        return out