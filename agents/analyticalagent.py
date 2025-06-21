import torch

class AnalyticalAgent:
    """
    Agent that uses the known analytical solution for the optimal control.
    """
    def __init__(self, dynamics):
        self.dynamics = dynamics

    def predict(self, t, y):
        """
        Predict the trading rate (q) using the analytical formula.

        Args:
            t: tensor of shape (batch, 1) - current time
            y: tensor of shape (batch, dim) - current state
        Returns:
            q: tensor of shape (batch, 1) - trading rate
        """
        return self.dynamics.optimal_control_analytic(t, y)

    def predict_Y(self, t, y):
        """
        Predict the value function (Y) using the analytical formula.

        Args:
            t: tensor of shape (batch, 1) - current time
            y: tensor of shape (batch, dim) - current state
        Returns:
            Y: tensor of shape (batch, 1) - value function
        """
        return self.dynamics.value_function_analytic(t, y)