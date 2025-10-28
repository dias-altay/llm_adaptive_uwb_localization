import numpy as np

class CVKalman2D:
    def __init__(self, dt=0.05, process_var=0.5, meas_var=0.04):
        self.dt = float(dt)
        self.process_var = float(process_var)
        self.meas_var = float(meas_var)

        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 10.0
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], float)
        self._rebuild_Q()
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], float)
        self._rebuild_R()
        self.I = np.eye(4)
        self._v_damp = 1.0
        self._pos_scale = 1.0

    def _rebuild_Q(self):
        q = self.process_var
        dt = self.dt
        G = np.array([[0.5 * dt * dt, 0],
                      [0, 0.5 * dt * dt],
                      [dt, 0],
                      [0, dt]], float)
        self.Q = (q ** 2) * (G @ G.T)

    def _rebuild_R(self):
        r = self.meas_var
        self.R = np.eye(2) * (r ** 2)

    def set_noise(self, process_var: float | None = None, meas_var: float | None = None):
        if process_var is not None:
            self.process_var = float(process_var)
            self._rebuild_Q()
        if meas_var is not None:
            self.meas_var = float(meas_var)
            self._rebuild_R()

    def set_velocity_damping(self, v_damp: float):
        try:
            v = float(v_damp)
        except Exception:
            v = 1.0
        self._v_damp = max(0.0, min(1.0, v))

    def set_position_scaling(self, pos_scale: float):
        try:
            v = float(pos_scale)
        except Exception:
            v = 1.0
        self._pos_scale = max(0.1, min(1.0, v))

    def predict(self):
        try:
            self.x[2, 0] *= self._v_damp
            self.x[3, 0] *= self._v_damp
        except Exception:
            pass
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.asarray(z, float).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        if self._pos_scale < 0.999:
            K[0:2, :] *= self._pos_scale

        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

    def state(self):
        return float(self.x[0, 0]), float(self.x[1, 0])
