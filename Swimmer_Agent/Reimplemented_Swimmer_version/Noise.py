import numpy as np

'''you can tune also time step decayed noise factor to reach a better exploration/exploitation'''

# Noise step decay
reduction_factor = 0.9
time_step_decay = 150

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.2, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


# Definition noise factor decay
def random_noise(n_f, actual_step):
    if (actual_step % time_step_decay) == 0:
        n_f *= reduction_factor
        return n_f
    else:
        return n_f
