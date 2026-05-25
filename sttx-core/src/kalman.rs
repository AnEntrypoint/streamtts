//! 2-state constant-velocity Kalman filter for smoothing per-loop confidence.
//!
//! DeepConf reads a noisy confidence signal off the model's logits; across
//! adaptive loops that signal arrives one measurement per loop. This filter
//! treats the loop index as time and estimates the *true* confidence plus its
//! velocity (rate of change across loops). The velocity term is what lets the
//! caller detect a plateau — when confidence stops climbing, more loops won't
//! help, so we exit (the diminishing-returns regime the Ouro paper observed
//! past 3-4 loops).
//!
//! State vector: x = [confidence, velocity].
//! Transition (dt = 1 loop):  F = [[1, 1], [0, 1]]
//! Measurement (confidence only): H = [1, 0]
//!
//! Everything is plain f32 scalar arithmetic — the 2x2 covariance update has a
//! closed form, so no matrix-inverse library is needed, and the filter never
//! touches the candle autograd graph (confidence is read out of logits as f32).

/// A 2-state (position + velocity) Kalman filter over a scalar measurement.
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    /// Estimated state: [confidence, velocity].
    x: [f32; 2],
    /// State covariance, row-major 2x2: [[p00, p01], [p10, p11]].
    p: [[f32; 2]; 2],
    /// Process noise scale (how much we trust the constant-velocity model).
    q: f32,
    /// Measurement noise variance (how noisy each confidence reading is).
    r: f32,
    /// Whether we've ingested at least one measurement yet.
    initialized: bool,
}

impl KalmanFilter {
    /// Construct from fitted parameters.
    ///
    /// * `q` – process noise scale (larger = filter tracks faster, trusts model less)
    /// * `r` – measurement noise variance (larger = filter smooths harder)
    /// * `init_var` – initial covariance on the diagonal (prior uncertainty)
    pub fn from_params(q: f32, r: f32, init_var: f32) -> Self {
        Self {
            x: [0.0, 0.0],
            p: [[init_var, 0.0], [0.0, init_var]],
            q,
            r,
            initialized: false,
        }
    }

    /// Ingest one confidence measurement, advancing the filter by one loop.
    ///
    /// The first measurement seeds the position directly (no prediction step),
    /// so a single-measurement filter never divides by an uninitialized
    /// covariance and never produces NaN.
    pub fn update(&mut self, measurement: f32) {
        if !self.initialized {
            self.x[0] = measurement;
            self.x[1] = 0.0;
            self.initialized = true;
            return;
        }

        // --- Predict (constant-velocity model, dt = 1) ---
        // x = F x
        let pred_pos = self.x[0] + self.x[1];
        let pred_vel = self.x[1];
        self.x[0] = pred_pos;
        self.x[1] = pred_vel;

        // P = F P F^T + Q
        // F P:
        let p = self.p;
        let fp = [
            [p[0][0] + p[1][0], p[0][1] + p[1][1]],
            [p[1][0], p[1][1]],
        ];
        // (F P) F^T  (F^T = [[1,0],[1,1]])
        let mut np = [
            [fp[0][0] + fp[0][1], fp[0][1]],
            [fp[1][0] + fp[1][1], fp[1][1]],
        ];
        // Add process noise (diagonal q).
        np[0][0] += self.q;
        np[1][1] += self.q;
        self.p = np;

        // --- Update (measure position only, H = [1, 0]) ---
        // Innovation covariance is scalar: S = H P H^T + R = P00 + R
        let s = self.p[0][0] + self.r;
        // Kalman gain K = P H^T / S = [P00/S, P10/S]
        let k0 = self.p[0][0] / s;
        let k1 = self.p[1][0] / s;
        // Innovation y = z - H x = z - x0
        let y = measurement - self.x[0];
        // State update x = x + K y
        self.x[0] += k0 * y;
        self.x[1] += k1 * y;
        // Covariance update P = (I - K H) P
        // (I - K H) = [[1-k0, 0], [-k1, 1]]
        let p = self.p;
        self.p = [
            [(1.0 - k0) * p[0][0], (1.0 - k0) * p[0][1]],
            [-k1 * p[0][0] + p[1][0], -k1 * p[0][1] + p[1][1]],
        ];
    }

    /// Filtered (denoised) confidence estimate.
    pub fn filtered_confidence(&self) -> f32 {
        self.x[0]
    }

    /// Estimated velocity (confidence change per loop). Near zero = plateau.
    pub fn velocity(&self) -> f32 {
        self.x[1]
    }

    /// Whether at least one measurement has been ingested.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rising_sequence_tracks_upward() {
        // Noisy but clearly rising confidence: filtered estimate should climb
        // and velocity should be positive.
        let mut kf = KalmanFilter::from_params(0.01, 0.1, 1.0);
        let seq = [0.10, 0.22, 0.29, 0.41, 0.52, 0.58];
        for &m in &seq {
            kf.update(m);
        }
        assert!(kf.filtered_confidence() > 0.4, "filtered={}", kf.filtered_confidence());
        assert!(kf.velocity() > 0.0, "velocity={}", kf.velocity());
    }

    #[test]
    fn plateau_drives_velocity_to_zero() {
        // After a rise that flattens, velocity should decay toward ~0.
        let mut kf = KalmanFilter::from_params(0.01, 0.1, 1.0);
        for &m in &[0.1, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] {
            kf.update(m);
        }
        assert!(kf.velocity().abs() < 0.05, "velocity={}", kf.velocity());
    }

    #[test]
    fn single_measurement_no_nan() {
        let mut kf = KalmanFilter::from_params(0.01, 0.1, 1.0);
        kf.update(0.42);
        assert!(kf.filtered_confidence().is_finite());
        assert!(kf.velocity().is_finite());
        assert_eq!(kf.filtered_confidence(), 0.42);
        assert_eq!(kf.velocity(), 0.0);
    }

    #[test]
    fn uninitialized_is_finite() {
        let kf = KalmanFilter::from_params(0.01, 0.1, 1.0);
        assert!(!kf.is_initialized());
        assert!(kf.filtered_confidence().is_finite());
    }
}
