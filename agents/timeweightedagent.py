import torch

class TimeWeightedAgent:
    """
    Agent that trades using a layered time-weighted average price (TWAP) strategy.
    The agent's goal is to change its current position `X` to meet a target position `D` by time `T`.
    At each time step, any change in the target `D` is treated as a new order
    to be executed as a TWAP strategy over the remaining time.
    The final trading rate is the sum of all such active TWAP strategies.
    """
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.N = dynamics.N  # number of time steps
        self.T = dynamics.T  # total time
        self.dt = dynamics.dt
        self.device = dynamics.device

        # Agent state, reset at the beginning of a simulation run
        self.q_schedule = None
        self.last_D = None
        self.initial_X = None

    def predict(self, t, y):
        """
        Predict the trading rate (q) for the current state.

        The trading rate is calculated by layering multiple TWAP strategies.
        A new TWAP strategy is initiated whenever the target demand `D` changes.

        Args:
            t: tensor of shape (batch, 1) - current time
            y: tensor of shape (batch, dim) - current state, y[:, 0] is X (position), y[:, 2] is D (demand)
        Returns:
            q: tensor of shape (batch, 1) - trading rate
        """
        X = y[:, 0:1]  # current position
        D = y[:, 2:3]  # forecasted demand
        
        current_step_index = int(round(t[0].item() / self.dt))

        if current_step_index == 0:
            # Reset state at the beginning of a simulation run
            batch_size = X.shape[0]
            self.q_schedule = torch.zeros(batch_size, self.N, device=self.device)
            self.initial_X = X
            
            # Initial order: liquidate initial position X to meet target D
            order_size = D - self.initial_X
            q_component = (order_size / self.N) / self.dt # convert quantity per step to rate
            self.q_schedule += q_component
            self.last_D = D
        else:
            # Subsequent steps: check for changes in D and create new orders
            order_size = D - self.last_D
            
            if torch.any(order_size != 0):
                steps_remaining = self.N - current_step_index
                if steps_remaining > 0:
                    q_component = (order_size / steps_remaining) / self.dt # convert quantity per step to rate
                    
                    update_indices = torch.arange(current_step_index, self.N, device=self.device)
                    update_values = q_component.repeat(1, len(update_indices))
                    self.q_schedule.index_add_(1, update_indices, update_values)

            self.last_D = D

        # The trade for the current step is the scheduled quantity
        q = self.q_schedule[:, current_step_index:current_step_index+1]
        return q
