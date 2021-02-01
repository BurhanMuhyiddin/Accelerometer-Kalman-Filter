import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, FT, st, amp):
        # create time range
        self.t = np.arange(0, FT, st)

        # initialize variables
        self._originalSignal = np.zeros((np.shape(self.t)))
        self._filteredSignal = np.zeros((np.shape(self.t)))

        # generate noisy signal
        self.generate_noisy_signal(st, amp)

        self.R = np.cov(self._originalSignal) * 30
        self.H = 1
        self.Q = 10
        self.P = 0
        self.U_hat = 0
        self.K = 0

    def generate_noisy_signal(self, st, amp):
        # create noise
        noise = np.random.random(np.shape(self.t))

        # get noisy signal amplitude
        self._originalSignal = np.sin(self.t) * amp + noise

    def update(self, U):
        self.K = self.P * self.H / (self.H * self.P * self.H + self.R)
        self.U_hat = self.U_hat + self.K * (U - self.H * self.U_hat)

        self.P = (1 - self.K * self.H) * self.P + self.Q

    def predict(self):
        pass

    def start_kalman_filter(self):
        for i in range(0, np.size(self.t)):
            self.update(self._originalSignal[i])
            self._filteredSignal[i] = self.U_hat

        self.plot_signal()

    def plot_signal(self):
        plt.plot(self.t, self._originalSignal)
        plt.plot(self.t, self._filteredSignal)
        plt.legend(["OriginalSignal", "Filtered Signal"])
        plt.grid(True, which='both')
        plt.show()


if __name__ == '__main__':
    kf = KalmanFilter(20, 0.1, 2)
    kf.start_kalman_filter()