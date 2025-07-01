import torch

class HybridAgent:
    """
    Agent that uses the known analytical solution for the optimal control.
    """
    def __init__(self, dynamics):
        self.dynamics = dynamics

    def predict(self, t, y):
        """
        Predict the trading rate (q) using a hybrid of the analytical formula and a no-trade region.

        Args:
            t: tensor of shape (batch, 1) - current time
            y: tensor of shape (batch, dim) - current state
        Returns:
            q: tensor of shape (batch, 1) - trading rate
        """
        X = y[:, 0:1]
        P = y[:, 1:2]
        D = y[:, 2:3]
        tau = self.dynamics.T - t

        # Compute Lambda
        dV_dX = -self.dynamics.eta * (D - X) / ((self.dynamics.eta + self.dynamics.nu) * tau + 2 * self.dynamics.gamma)
        dV_dP = -1 / ((self.dynamics.eta + self.dynamics.nu) * tau + 2 * self.dynamics.gamma)
        Lambda = P + dV_dX + self.dynamics.nu * dV_dP

        # Compute analytical control
        q_analytic = self.dynamics.optimal_control_analytic(t, y)

        # Apply no-trade region logic
        psi = self.dynamics.psi
        no_trade = Lambda.abs() <= psi
        q = torch.where(no_trade, torch.zeros_like(q_analytic), q_analytic)
        return q

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