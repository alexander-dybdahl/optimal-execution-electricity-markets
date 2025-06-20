import torch

class TimeWeightedAgent:
    """
    Agent that trades using a time-weighted average price (TWAP) strategy.
    At each time step, it liquidates the remaining inventory evenly over the remaining time.
    """
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.N = dynamics.N  # number of time steps
        self.T = dynamics.T  # total time
        self.dt = dynamics.dt
        self.device = dynamics.device

    def predict(self, t, y):
        """
        Predict the trading rate (q) for the current state.
        Args:
            t: tensor of shape (batch, 1) - current time
            y: tensor of shape (batch, dim) - current state, y[:, 0] is X (position), y[:, 2] is D (demand)
        Returns:
            q: tensor of shape (batch, 1) - trading rate
        """
        X = y[:, 0:1]  # current position
        D = y[:, 2:3]  # forecasted demand
        # Compute number of remaining steps (assume t in [0, T])
        steps_remaining = torch.clamp(((self.T - t) / self.dt).ceil(), min=1)
        # Evenly distribute the difference to target D by T
        q = (D - X) / steps_remaining
        return q
