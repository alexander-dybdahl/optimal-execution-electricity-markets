import torch

class StartEndAgent:
    """
    Agent that only trades at the start and end of the trading period to close the G-X gap.
    - At the beginning (t=0): trades half of the G-X gap
    - At the end (t=T): trades the remaining G-X gap to exactly meet generation target
    
    Where G is generation target and X is current position.
    """
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.dt = dynamics.dt
        self.T = dynamics.T
        self.N = dynamics.N
        self.device = dynamics.device
        
        # Track if we've already traded at the start
        self.start_trade_done = None
        self.initial_gap = None

    def predict(self, t, y):
        """
        Predict the trading rate (q) for the current state.

        Args:
            t: tensor of shape (batch, 1) - current time
            y: tensor of shape (batch, dim) - current state, y[:, 0] is X (position), y[:, 2] is G (generation target)
        Returns:
            q: tensor of shape (batch, 1) - trading rate
        """
        X = y[:, 0:1]  # current position
        G = y[:, 2:3]  # generation target (using G instead of D for generation)
        
        current_time = t[0].item()
        current_step_index = int(round(current_time / self.dt))
        
        # Initialize tracking variables at the start of simulation
        if current_step_index == 0:
            batch_size = X.shape[0]
            self.start_trade_done = torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device)
            self.initial_gap = G - X  # Store the initial G-X gap
        
        # Trade at the start (t=0)
        if current_step_index == 0 and not torch.all(self.start_trade_done):
            # Trade half of the initial gap at the start
            # We divide by dt to get the rate q, since dX = q * dt
            trade_fraction = 1.0
            q = (self.initial_gap * trade_fraction) / self.dt
            self.start_trade_done.fill_(True)
            return q
        
        # Trade at the end (last time step)
        elif current_step_index == self.N - 1:
            # Trade the remaining gap to exactly meet the generation target
            # This should be approximately the other half, but we calculate it exactly
            # to account for any accumulated drift or changes
            remaining_gap = G - X
            q = remaining_gap / self.dt
            return q
        
        # No trading during intermediate time steps
        else:
            batch_size = X.shape[0]
            return torch.zeros(batch_size, 1, device=self.device)

    def reset(self):
        """Reset the agent state for a new simulation run."""
        self.start_trade_done = None
        self.initial_gap = None
