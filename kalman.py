import cv2
import numpy as np

class KalmanImageFilter:
    def __init__(self, dt=1, acceleration=1, measurement_uncertainty=50):
        self.dt = dt
        self.acc = acceleration
        self.R = measurement_uncertainty

        # State: pixel intensity + velocity
        self.initialized = False

    def initialize(self, frame):
        h, w = frame.shape
        self.x = np.zeros((h, w, 2))  # [value, velocity]
        self.P = np.ones((h, w, 2, 2)) * 1000

        self.initialized = True

    def process(self, frame):
        frame = frame.astype(np.float32)

        if not self.initialized:
            self.initialize(frame)

        dt = self.dt

        # Matrices
        F = np.array([[1, dt],
                      [0, 1]])

        Q = self.acc * np.array([[dt**4/4, dt**3/2],
                                 [dt**3/2, dt**2]])

        H = np.array([[1, 0]])
        R = self.R

        h, w = frame.shape
        output = np.zeros_like(frame)

        for i in range(h):
            for j in range(w):

                x = self.x[i, j]
                P = self.P[i, j]

                # Predict
                x = F @ x
                P = F @ P @ F.T + Q

                # Update
                z = frame[i, j]
                y = z - (H @ x)
                S = H @ P @ H.T + R
                K = P @ H.T / S

                x = x + (K.flatten() * y)
                P = (np.eye(2) - K @ H) @ P

                self.x[i, j] = x
                self.P[i, j] = P

                output[i, j] = x[0]

        return np.clip(output, 0, 255).astype(np.uint8)


def kalman(Images, sol=None):
    if sol is None:
        sol = [2, 4, 60]
    dt = int(sol[0])
    acceleration = int(sol[1])
    measurement_noise = int(sol[2])
    kf = KalmanImageFilter(dt, acceleration, measurement_noise)
    Filtered_Images = []
    for i in range(len(Images)):
        frame = Images[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filtered = kf.process(gray)
        Filtered_Images.append(filtered)
    return Filtered_Images
