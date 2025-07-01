import torch

class KTimesAgent:
    """
    Agent that trades exactly K times during the trading interval.
    The trades are evenly distributed across the time horizon, and each trade
    closes (1/K) of the total G-X gap.
    
    Where G is generation target and X is current position.
    """
    def __init__(self, dynamics, K=5):
        self.dynamics = dynamics
        self.dt = dynamics.dt
        self.T = dynamics.T
        self.N = dynamics.N
        self.device = dynamics.device
        self.K = K  # Number of times to trade
        
        # Calculate the time steps at which to trade
        if K > self.N:
            raise ValueError(f"K ({K}) cannot be larger than the number of time steps ({self.N})")
        
        # Distribute K trades evenly across N time steps
        self.trade_steps = []
        if K == 1:
            # Single trade at the middle
            self.trade_steps = [self.N // 2]
        else:
            # Distribute evenly from start to end
            step_interval = (self.N - 1) / (K - 1)
            self.trade_steps = [int(round(i * step_interval)) for i in range(K)]
        
        # Ensure we don't have duplicate steps and sort them
        self.trade_steps = sorted(list(set(self.trade_steps)))
        self.K = len(self.trade_steps)  # Update K in case of duplicates
        
        # Track trading state
        self.trades_completed = None
        self.initial_gap = None
        self.trade_amount_per_trade = None

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
            self.trades_completed = torch.zeros(batch_size, 1, dtype=torch.int32, device=self.device)
            self.initial_gap = G - X  # Store the initial G-X gap
            self.trade_amount_per_trade = self.initial_gap / self.K  # Amount to trade each time
        
        # Check if current step is a trading step
        if current_step_index in self.trade_steps:
            # Determine which trade this is (0-indexed)
            trade_index = self.trade_steps.index(current_step_index)
            
            # Check if we haven't completed this trade yet
            batch_size = X.shape[0]
            mask = (self.trades_completed.squeeze() <= trade_index).unsqueeze(1)
            
            if torch.any(mask):
                # For the last trade, trade exactly to the target to handle any rounding errors
                if trade_index == self.K - 1:
                    # Last trade: close the remaining gap exactly
                    remaining_gap = G - X
                    q = torch.where(mask, remaining_gap / self.dt, torch.zeros_like(X))
                else:
                    # Regular trade: trade 1/K of the initial gap
                    q = torch.where(mask, self.trade_amount_per_trade / self.dt, torch.zeros_like(X))
                
                # Update trades completed counter
                self.trades_completed = torch.where(mask, 
                                                  torch.full_like(self.trades_completed, trade_index + 1),
                                                  self.trades_completed)
                return q
        
        # No trading at this time step
        batch_size = X.shape[0]
        return torch.zeros(batch_size, 1, device=self.device)

    def reset(self):
        """Reset the agent state for a new simulation run."""
        self.trades_completed = None
        self.initial_gap = None
        self.trade_amount_per_trade = None

    def get_trade_schedule(self):
        """Return the trading schedule for debugging/visualization."""
        trade_times = [step * self.dt for step in self.trade_steps]
        return {
            'K': self.K,
            'trade_steps': self.trade_steps,
            'trade_times': trade_times,
            'trade_amount_fraction': 1.0 / self.K
        }
