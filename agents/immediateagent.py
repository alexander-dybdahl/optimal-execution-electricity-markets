import torch

class ImmediateAgent:
    """
    Agent that trades immediately to meet the target demand.
    At each time step, it trades the entire difference between the current position X and the target D.
    """
    def __init__(self, dynamics):
        self.dynamics = dynamics
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
        
        # Trade the entire difference immediately
        # We need to divide by dt to get the rate q, since dX = q * dt
        q = (D - X) / self.dt
        return q
