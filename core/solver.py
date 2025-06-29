import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from scipy import stats

from utils.simulator import simulate_paths, compute_cost_objective


class Solver:
    def __init__(self, dynamics, seed, n_sim):
        self.dynamics = dynamics
        self.device = dynamics.device
        self.seed = seed
        self.n_sim = n_sim
        self.results = {}
        self.costs = {}
        self.risk_metrics = {}
        self.colors = {}
        self._color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self._color_idx = 0

    def evaluate_agent(self, agent, agent_name):
        # Assign a color if not already assigned
        if agent_name not in self.colors:
            self.colors[agent_name] = self._color_cycle[self._color_idx % len(self._color_cycle)]
            self._color_idx += 1
        
        # Run simulation
        timesteps, results = simulate_paths(
            dynamics=self.dynamics,
            agent=agent,
            n_sim=self.n_sim,
            seed=self.seed,
            analytical=False
        )
        # Compute cost objective
        cost_objective = compute_cost_objective(
            dynamics=self.dynamics,
            q_traj=torch.from_numpy(results["q_learned"]).to(self.device),
            y_traj=torch.from_numpy(results["y_learned"]).to(self.device),
            terminal_cost=True
        )
        
        costs_numpy = cost_objective.detach().cpu().numpy()
        
        self.results[agent_name] = {
            'timesteps': timesteps,
            'results': results
        }

        self.costs[agent_name] = costs_numpy
        
        # Calculate and store risk metrics
        self.risk_metrics[agent_name] = self.calculate_risk_metrics(costs_numpy)

    def plot_trajectories_expectation(self, plot=True, save_dir=None):
        """
        Plot expected trajectories (mean ± std) of q, y, Y for each agent.
        """
        plt.rcParams.update({'font.size': 14})
        if not self.results:
            print("No results to plot.")
            return
        
        # Plot q, y, Y as subplots in a single figure
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        var_ylabels = {'q': '$q(t)$', 'y': '$y(t)$', 'Y': '$Y(t)$'}
        line_styles = {'learned': '-'}
        
        agent_handles = []
        for agent_name, color in self.colors.items():
            agent_handles.append(plt.Line2D([0], [0], color=color, linestyle='-', label=agent_name))
            
        style_handles = [Patch(facecolor='gray', alpha=0.2, label='Agent ±1 std')]

        for idx, var in enumerate(['q', 'y', 'Y']):
            ax = axs[idx]
            for agent_name, data in self.results.items():
                timesteps = data['timesteps']
                results = data['results']
                color = self.colors[agent_name]
                # Plot learned
                key_learned = f'{var}_learned'
                arr_learned = results.get(key_learned, None)
                if arr_learned is not None:
                    arr_learned = np.asarray(arr_learned)
                    arr_mean = arr_learned.mean(axis=1)
                    arr_std = arr_learned.std(axis=1)

                    # Squeeze last dim if it's 1
                    if arr_mean.ndim > 1 and arr_mean.shape[-1] == 1:
                        arr_mean = arr_mean.squeeze(-1)
                        arr_std = arr_std.squeeze(-1)

                    # Handle plotting based on dimension
                    if arr_mean.ndim == 1:
                        ax.plot(
                            timesteps[:arr_mean.shape[0]],
                            arr_mean,
                            color=color,
                            linestyle=line_styles['learned'],
                            alpha=1.0
                        )
                        ax.fill_between(
                            timesteps[:arr_mean.shape[0]],
                            arr_mean - arr_std,
                            arr_mean + arr_std,
                            color=color,
                            alpha=0.2,
                            linewidth=0
                        )
                    else:  # arr_mean.ndim > 1
                        for i in range(arr_mean.shape[-1]):
                            # Plot each dimension separately
                            ax.plot(
                                timesteps[:arr_mean.shape[0]],
                                arr_mean[:, i],
                                color=color,
                                linestyle=line_styles['learned'],
                                alpha=1.0
                            )
                            ax.fill_between(
                                timesteps[:arr_mean.shape[0]],
                                arr_mean[:, i] - arr_std[:, i],
                                arr_mean[:, i] + arr_std[:, i],
                                color=color,
                                alpha=0.2,
                                linewidth=0
                            )

            ax.set_xlabel("Time")
            ax.set_ylabel(var_ylabels[var])
            ax.grid(True, linestyle='--', alpha=0.5)
            leg1 = ax.legend(handles=agent_handles, loc='upper right', frameon=True)
            ax.add_artist(leg1)
            ax.legend(handles=style_handles, loc='lower right', frameon=True)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/imgs/trajectories_expectation.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()

    def plot_trajectories_individual(self, n_traj=5, plot=True, save_dir=None):
        """
        Plot individual trajectories of q, y, Y for each agent.
        Shows n_traj individual realizations instead of expectation.
        
        Args:
            n_traj (int): Number of individual trajectories to plot per agent
            plot (bool): Whether to show the plots interactively
            save_dir (str): Directory to save the plots, if provided
        """
        plt.rcParams.update({'font.size': 14})
        if not self.results:
            print("No results to plot.")
            return
        
        # Check if n_traj is valid
        if n_traj > self.n_sim:
            print(f"Warning: n_traj ({n_traj}) is larger than n_simulations ({self.n_sim}). Using n_simulations instead.")
            n_traj = self.n_sim
        
        # Plot q, y, Y as subplots in a single figure
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        var_ylabels = {'q': '$q(t)$', 'y': '$y(t)$', 'Y': '$Y(t)$'}
        line_styles = {'learned': '-'}
        
        agent_handles = []
        for agent_name, color in self.colors.items():
            agent_handles.append(plt.Line2D([0], [0], color=color, linestyle='-', label=agent_name))

        for idx, var in enumerate(['q', 'y', 'Y']):
            ax = axs[idx]
            for agent_name, data in self.results.items():
                timesteps = data['timesteps']
                results = data['results']
                color = self.colors[agent_name]
                # Plot learned
                key_learned = f'{var}_learned'
                arr_learned = results.get(key_learned, None)
                if arr_learned is not None:
                    arr_learned = np.asarray(arr_learned)

                    # Plot individual trajectories
                    for traj_idx in range(min(n_traj, arr_learned.shape[1])):
                        traj_data = arr_learned[:, traj_idx, :]
                        
                        # Squeeze last dim if it's 1
                        if traj_data.ndim > 1 and traj_data.shape[-1] == 1:
                            traj_data = traj_data.squeeze(-1)

                        # Handle plotting based on dimension
                        if traj_data.ndim == 1:
                            ax.plot(
                                timesteps[:traj_data.shape[0]],
                                traj_data,
                                color=color,
                                linestyle=line_styles['learned'],
                                alpha=0.6,
                                linewidth=1
                            )
                        else:  # traj_data.ndim > 1
                            for i in range(traj_data.shape[-1]):
                                # Plot each dimension separately
                                ax.plot(
                                    timesteps[:traj_data.shape[0]],
                                    traj_data[:, i],
                                    color=color,
                                    linestyle=line_styles['learned'],
                                    alpha=0.6,
                                    linewidth=1
                                )

            ax.set_xlabel("Time")
            ax.set_ylabel(var_ylabels[var])
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(handles=agent_handles, loc='upper right', frameon=True)

        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/imgs/trajectories_individual.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()

    # Backward compatibility method
    def plot_traj(self, plot=True, save_dir=None):
        """
        Backward compatibility method. Calls plot_trajectories_expectation.
        """
        return self.plot_trajectories_expectation(plot=plot, save_dir=save_dir)

    def plot_cost_histograms(self, plot=True, save_dir=None):
        plt.rcParams.update({'font.size': 14})
        if not self.costs:
            print("No costs to plot.")
            return

        n_agents = len(self.costs)
        if n_agents == 0:
            return
            
        fig, axs = plt.subplots(n_agents, 1, figsize=(8, 6 * n_agents), squeeze=False)

        for i, (agent_name, costs) in enumerate(self.costs.items()):
            ax = axs[i, 0]
            color = self.colors.get(agent_name, 'gray')

            # Plot histogram
            ax.hist(costs, bins='auto', color=color, alpha=0.7, label='Cost Distribution')

            # Calculate stats
            mean_cost = np.mean(costs)
            std_cost = np.std(costs)
            mean_std_ratio = mean_cost / std_cost if std_cost > 0 else 0.0

            # Plot mean line
            ax.axvline(mean_cost, color='red', linestyle='dotted', linewidth=2, label=f'Mean: {mean_cost:.4f}')

            # Calculate and plot mode from histogram
            hist, bin_edges = np.histogram(costs, bins='auto')
            if len(hist) > 0:
                max_hist_index = np.argmax(hist)
                mode_cost = (bin_edges[max_hist_index] + bin_edges[max_hist_index+1]) / 2
                ax.axvline(mode_cost, color='green', linestyle='dotted', linewidth=2, label=f'Mode: {mode_cost:.4f}')

            ax.set_title(f'Cost Distribution for {agent_name}')
            ax.set_xlabel("Cost Objective")
            ax.set_ylabel("Frequency")
            
            # Create legend
            handles, labels = ax.get_legend_handles_labels()
            
            # Add stats to legend
            handles.append(Patch(color='none', label=f'Std Dev: {std_cost:.4f}'))
            handles.append(Patch(color='none', label=f'Mean/Std: {mean_std_ratio:.4f}'))
            
            ax.legend(handles=handles, loc="upper left")
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/imgs/cost_histograms.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()
            
    def calculate_risk_metrics(self, costs, confidence_levels=[0.95, 0.99]):
        """
        Calculate comprehensive risk metrics for a given cost distribution.
        
        Args:
            costs (np.array): Array of cost values
            confidence_levels (list): Confidence levels for VaR and CVaR calculation
            
        Returns:
            dict: Dictionary containing various risk metrics
        """
        costs = np.array(costs)
        risk_metrics = {}
        
        # Basic statistics
        risk_metrics['mean'] = float(np.mean(costs))
        risk_metrics['std'] = float(np.std(costs))
        risk_metrics['skewness'] = float(stats.skew(costs))
        risk_metrics['kurtosis'] = float(stats.kurtosis(costs))
        risk_metrics['min'] = float(np.min(costs))
        risk_metrics['max'] = float(np.max(costs))
        
        # Percentiles
        risk_metrics['percentile_10'] = float(np.percentile(costs, 10))
        risk_metrics['percentile_25'] = float(np.percentile(costs, 25))
        risk_metrics['percentile_50'] = float(np.percentile(costs, 50))  # Median
        risk_metrics['percentile_75'] = float(np.percentile(costs, 75))
        risk_metrics['percentile_90'] = float(np.percentile(costs, 90))
        
        # Calculate VaR and CVaR for different confidence levels
        for confidence_level in confidence_levels:
            alpha = 1 - confidence_level
            
            # Value at Risk (VaR) - quantile at confidence level
            var_value = float(np.percentile(costs, confidence_level * 100))
            risk_metrics[f'VaR_{int(confidence_level*100)}'] = var_value
            
            # Conditional Value at Risk (CVaR) - expected value of costs above VaR
            tail_costs = costs[costs >= var_value]
            if len(tail_costs) > 0:
                cvar_value = float(np.mean(tail_costs))
            else:
                cvar_value = var_value
            risk_metrics[f'CVaR_{int(confidence_level*100)}'] = cvar_value
            
            # Expected Shortfall (same as CVaR but calculated differently for verification)
            sorted_costs = np.sort(costs)
            n = len(sorted_costs)
            var_index = int(np.ceil(confidence_level * n)) - 1
            if var_index < n - 1:
                es_value = float(np.mean(sorted_costs[var_index:]))
            else:
                es_value = float(sorted_costs[-1])
            risk_metrics[f'ES_{int(confidence_level*100)}'] = es_value
        
        # Semi-deviation (downside risk)
        mean_cost = risk_metrics['mean']
        downside_deviations = costs[costs > mean_cost] - mean_cost
        if len(downside_deviations) > 0:
            risk_metrics['semi_deviation'] = float(np.sqrt(np.mean(downside_deviations**2)))
        else:
            risk_metrics['semi_deviation'] = 0.0
            
        # Maximum Drawdown (in context of cost, this is maximum increase from minimum)
        running_min = np.minimum.accumulate(costs)
        drawdowns = costs - running_min
        risk_metrics['max_drawdown'] = float(np.max(drawdowns))
        
        # Sharpe-like ratio (mean/std)
        if risk_metrics['std'] > 0:
            risk_metrics['sharpe_ratio'] = float(risk_metrics['mean'] / risk_metrics['std'])
        else:
            risk_metrics['sharpe_ratio'] = 0.0
            
        # Sortino ratio (mean/semi_deviation)
        if risk_metrics['semi_deviation'] > 0:
            risk_metrics['sortino_ratio'] = float(risk_metrics['mean'] / risk_metrics['semi_deviation'])
        else:
            risk_metrics['sortino_ratio'] = 0.0
            
        return risk_metrics
            
    def compare_agents(self, agent_name1, agent_name2, alpha=0.05, alternative='less'):
        """
        Perform a t-test to compare the cost objectives between two agents.
        
        Args:
            agent_name1 (str): First agent name (test if this agent performs better)
            agent_name2 (str): Second agent name
            alpha (float): Significance level for the test (default: 0.05)
            alternative (str): 'less' tests if agent1 has lower costs than agent2,
                              'greater' tests if agent1 has higher costs than agent2,
                              'two-sided' tests if the costs are different
                              
        Returns:
            dict: Dictionary containing test results with the following keys:
                  - p_value: p-value of the test
                  - significant: True if the difference is statistically significant
                  - mean_diff: Difference in means (agent1 - agent2)
                  - better: If alternative='less', True if agent1 has significantly lower costs
        """
        plt.rcParams.update({'font.size': 14})
        if agent_name1 not in self.costs or agent_name2 not in self.costs:
            missing = []
            if agent_name1 not in self.costs:
                missing.append(agent_name1)
            if agent_name2 not in self.costs:
                missing.append(agent_name2)
            raise ValueError(f"Agent(s) not found in evaluated costs: {', '.join(missing)}")
            
        # Get cost arrays
        costs1 = self.costs[agent_name1]
        costs2 = self.costs[agent_name2]
        
        # Calculate means
        mean1 = np.mean(costs1)
        mean2 = np.mean(costs2)
        mean_diff = mean1 - mean2
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(costs1, costs2, equal_var=False, alternative=alternative)
        
        # Check if result is statistically significant
        significant = p_value < alpha
        
        # Determine if agent1 is better (has lower costs)
        # Note: This interpretation depends on the alternative hypothesis
        if alternative == 'less':
            better = mean1 < mean2 and significant
        elif alternative == 'greater':
            better = mean1 > mean2 and significant
        else:  # two-sided
            better = False  # Cannot determine "better" with a two-sided test
        
        # Return results - ensure p_value is a float, not an array
        return {
            'p_value': float(p_value),  # Convert to float
            'significant': significant,
            'mean_diff': mean_diff,
            'better': better,
            'mean1': mean1,
            'mean2': mean2,
            't_statistic': float(t_stat),  # Convert to float
            'sample_size1': len(costs1),
            'sample_size2': len(costs2)
        }
    
    def plot_detailed_trading_trajectories(self, plot=True, save_dir=None):
        """
        Plot detailed trading trajectories showing:
        1. Individual trade sizes vs execution prices
        2. Cumulative position vs time with price overlay
        3. Trade-by-trade analysis for each agent
        """
        plt.rcParams.update({'font.size': 14})
        if not self.results:
            print("No results to plot.")
            return
        
        # Create a comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(20, 20))
        
        # Define subplot layout: 5 rows, 2 columns  
        gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1], hspace=0.3, wspace=0.25)
        
        # Subplot 1: Trade Size vs Execution Price (scatter plot)
        ax1 = fig.add_subplot(gs[0, 0])
        # Subplot 2: Cumulative Position vs Time
        ax2 = fig.add_subplot(gs[0, 1])
        # Subplot 3: Price Evolution vs Time
        ax3 = fig.add_subplot(gs[1, 0])
        # Subplot 4: Individual Trade Sizes vs Time
        ax4 = fig.add_subplot(gs[1, 1])
        # Subplot 5: Execution Price vs Time
        ax5 = fig.add_subplot(gs[2, 0])
        # Subplot 6: Trade Value (Trade Size × Price) vs Time
        ax6 = fig.add_subplot(gs[2, 1])
        # Subplot 7: Terminal Cost vs Time
        ax7 = fig.add_subplot(gs[3, 0])
        # Subplot 8: Cumulative Trading Cost vs Time
        ax8 = fig.add_subplot(gs[3, 1])
        # Subplot 9: Total Cost Components vs Time (spans both columns)
        ax9 = fig.add_subplot(gs[4, :])  # Spans both columns
        
        agent_handles = []
        
        for agent_name, data in self.results.items():
            timesteps = data['timesteps']
            results = data['results']
            color = self.colors[agent_name]
            
            # Extract data
            q_traj = results.get('q_learned', None)  # Trade sizes (N x n_sim x control_dim)
            y_traj = results.get('y_learned', None)  # State trajectories (N+1 x n_sim x state_dim)
            
            if q_traj is None or y_traj is None:
                continue
                
            q_traj = np.asarray(q_traj)
            y_traj = np.asarray(y_traj)
            
            # Extract components from state trajectory
            X_traj = y_traj[:, :, 0]  # Cumulative position
            P_traj = y_traj[:, :, 1]  # Price
            if y_traj.shape[-1] > 2:
                D_traj = y_traj[:, :, 2]  # Demand
                # Calculate position-demand gap (this is what the terminal cost penalizes)
                gap_traj = D_traj - X_traj  # Positive gap means undersupply, negative means oversupply
            else:
                D_traj = None
                gap_traj = None
            
            # Calculate execution prices using the dynamics generator
            # The generator computes q * execution_price, so we divide by q to get execution_price
            execution_prices = np.zeros_like(q_traj)
            for t in range(q_traj.shape[0]):
                y_t = torch.from_numpy(y_traj[t, :, :]).to(self.device)
                q_t = torch.from_numpy(q_traj[t, :, :]).to(self.device)
                
                # Use the dynamics generator to get accurate execution prices
                if hasattr(self.dynamics, 'generator'):
                    # For dynamics like FullDynamics: execution_price = P + sign(q)*psi + gamma*q
                    P_t = y_t[:, 1:2]  # Price
                    if hasattr(self.dynamics, 'psi') and hasattr(self.dynamics, 'gamma'):
                        temporary_impact = self.dynamics.gamma * q_t
                        bid_ask_spread = torch.sign(q_t) * self.dynamics.psi
                        exec_price_t = P_t + bid_ask_spread + temporary_impact
                
                execution_prices[t, :, :] = exec_price_t.detach().cpu().numpy()
            
            # Calculate means and stds for plotting
            q_mean = q_traj.mean(axis=1).squeeze()
            q_std = q_traj.std(axis=1).squeeze()
            X_mean = X_traj.mean(axis=1)
            X_std = X_traj.std(axis=1)
            P_mean = P_traj.mean(axis=1)
            P_std = P_traj.std(axis=1)
            exec_price_mean = execution_prices.mean(axis=1).squeeze()
            exec_price_std = execution_prices.std(axis=1).squeeze()
            
            # Calculate gap statistics if demand is available
            if gap_traj is not None:
                gap_mean = gap_traj.mean(axis=1)
                gap_std = gap_traj.std(axis=1)
            else:
                gap_mean = None
                gap_std = None
            
            # Calculate terminal cost at each time step
            terminal_costs = np.zeros((y_traj.shape[0], y_traj.shape[1]))
            for t in range(y_traj.shape[0]):
                y_t = torch.from_numpy(y_traj[t, :, :]).to(self.device)
                if hasattr(self.dynamics, 'terminal_cost'):
                    terminal_cost_t = self.dynamics.terminal_cost(y_t)
                    terminal_costs[t, :] = terminal_cost_t.detach().cpu().numpy().squeeze()
                else:
                    # Fallback: assume terminal cost is 0.5 * eta * (D - X)^2
                    X_t = y_t[:, 0:1]  # Position
                    if y_t.shape[-1] > 2:
                        D_t = y_t[:, 2:3]  # Demand
                        eta = getattr(self.dynamics, 'eta', 1.0)  # Default eta = 1.0
                        terminal_cost_t = 0.5 * eta * (D_t - X_t)**2
                        terminal_costs[t, :] = terminal_cost_t.detach().cpu().numpy().squeeze()
            
            terminal_cost_mean = terminal_costs.mean(axis=1)
            terminal_cost_std = terminal_costs.std(axis=1)
            
            # Trade times (excluding t=0 since no trades happen then)
            trade_times = timesteps[:-1]
            
            # 1. Trade Size vs Execution Price (scatter plot)
            ax1.scatter(exec_price_mean[q_mean != 0], q_mean[q_mean != 0], 
                       color=color, alpha=0.7, s=50, label=agent_name)
            
            # 2. Cumulative Position vs Time (with Demand overlay)
            ax2.plot(timesteps, X_mean, color=color, linewidth=2, 
                    linestyle='-', label=f'{agent_name} (Position)')
            ax2.fill_between(timesteps, X_mean - X_std, X_mean + X_std, 
                           color=color, alpha=0.2)
            
            # Also plot demand if available
            if D_traj is not None:
                D_mean = D_traj.mean(axis=1)
                D_std = D_traj.std(axis=1)
                ax2.plot(timesteps, D_mean, color=color, linewidth=2, 
                        linestyle='--', label=f'{agent_name} (Demand)')
                ax2.fill_between(timesteps, D_mean - D_std, D_mean + D_std, 
                               color=color, alpha=0.1)
            
            # 3. Price Evolution vs Time
            ax3.plot(timesteps, P_mean, color=color, linewidth=2, label=agent_name)
            ax3.fill_between(timesteps, P_mean - P_std, P_mean + P_std, 
                           color=color, alpha=0.2)
            
            # 4. Individual Trade Sizes vs Time
            ax4.plot(trade_times, q_mean, color=color, linewidth=2, 
                    marker='o', markersize=4, label=agent_name)
            ax4.fill_between(trade_times, q_mean - q_std, q_mean + q_std, 
                           color=color, alpha=0.2)
            
            # 5. Execution Price vs Time
            ax5.plot(trade_times, exec_price_mean, color=color, linewidth=2, 
                    marker='s', markersize=3, label=agent_name)
            ax5.fill_between(trade_times, exec_price_mean - exec_price_std, 
                           exec_price_mean + exec_price_std, color=color, alpha=0.2)
            
            # 6. Trade Value vs Time (instantaneous running cost)
            trade_values = q_mean * exec_price_mean
            ax6.plot(trade_times, trade_values, color=color, linewidth=2, 
                    marker='^', markersize=3, label=agent_name)
            
            # 7. Terminal Cost vs Time
            ax7.plot(timesteps, terminal_cost_mean, color=color, linewidth=2, 
                    marker='d', markersize=3, label=agent_name)
            ax7.fill_between(timesteps, terminal_cost_mean - terminal_cost_std, 
                           terminal_cost_mean + terminal_cost_std, color=color, alpha=0.2)
            
            # 8. Cumulative Trading Cost vs Time (properly scaled with dt)
            dt = self.dynamics.dt
            cumulative_trade_value = np.cumsum(np.concatenate([[0], trade_values * dt]))
            ax8.plot(timesteps, cumulative_trade_value, color=color, linewidth=3, 
                    label=agent_name)
            
            # 9. Total Cost Components vs Time (running cost + terminal cost)
            total_cost_at_time = cumulative_trade_value + terminal_cost_mean
            ax9.plot(timesteps, cumulative_trade_value, color=color, linewidth=2, 
                    linestyle='-', label=f'{agent_name} (Running Cost)')
            ax9.plot(timesteps, terminal_cost_mean, color=color, linewidth=2, 
                    linestyle='--', label=f'{agent_name} (Terminal Cost)')
            ax9.plot(timesteps, total_cost_at_time, color=color, linewidth=3, 
                    linestyle=':', label=f'{agent_name} (Total Cost)')
            
            # Store handle for legend
            if agent_name not in [h.get_label() for h in agent_handles]:
                agent_handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=agent_name))
        
        # Customize subplots
        ax1.set_title('Trade Size vs Execution Price', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Execution Price')
        ax1.set_ylabel('Trade Size')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_title('Position vs Demand vs Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Quantity')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3.set_title('Price Evolution vs Time', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Price')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        ax4.set_title('Trade Sizes vs Time', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Trade Size')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.legend()
        
        ax5.set_title('Execution Prices vs Time', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Execution Price')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        ax6.set_title('Instantaneous Trading Cost Rate vs Time', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Time')
        ax6.set_ylabel('q(t) × execution_price(t)')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax6.legend()
        
        ax7.set_title('Terminal Cost vs Time', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Time')
        ax7.set_ylabel('Terminal Cost: 0.5η(D-X)²')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        
        ax8.set_title('Cumulative Running Cost vs Time', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Time')
        ax8.set_ylabel('∫ q(s) × execution_price(s) ds')
        ax8.grid(True, alpha=0.3)
        ax8.legend()
        
        ax9.set_title('Total Cost Components Breakdown vs Time', fontsize=12, fontweight='bold')
        ax9.set_xlabel('Time')
        ax9.set_ylabel('Cost')
        ax9.grid(True, alpha=0.3)
        ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle('Detailed Trading Trajectories Comparison', fontsize=16, fontweight='bold', y=0.98)
        
        if save_dir:
            plt.savefig(f"{save_dir}/imgs/detailed_trading_trajectories.png", 
                       dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()

    def plot_trading_heatmap(self, plot=True, save_dir=None):
        """
        Create a heatmap showing trading intensity across time and price levels for each agent
        """
        plt.rcParams.update({'font.size': 14})
        if not self.results:
            print("No results to plot.")
            return
            
        fig, axes = plt.subplots(1, len(self.results), figsize=(6*len(self.results), 8))
        if len(self.results) == 1:
            axes = [axes]
        
        for idx, (agent_name, data) in enumerate(self.results.items()):
            timesteps = data['timesteps']
            results = data['results']
            
            q_traj = results.get('q_learned', None)
            y_traj = results.get('y_learned', None)
            
            if q_traj is None or y_traj is None:
                continue
                
            q_traj = np.asarray(q_traj)
            y_traj = np.asarray(y_traj)
            
            # Extract price and trade data
            P_traj = y_traj[:-1, :, 1]  # Price at trade times
            q_traj_flat = q_traj.reshape(-1)
            P_traj_flat = P_traj.reshape(-1)
            
            # Create 2D histogram for heatmap
            price_bins = np.linspace(P_traj_flat.min(), P_traj_flat.max(), 30)
            trade_bins = np.linspace(q_traj_flat.min(), q_traj_flat.max(), 30)
            
            hist, xedges, yedges = np.histogram2d(P_traj_flat, q_traj_flat, 
                                                 bins=[price_bins, trade_bins])
            
            # Plot heatmap
            im = axes[idx].imshow(hist.T, origin='lower', aspect='auto', 
                                 extent=[price_bins[0], price_bins[-1], 
                                        trade_bins[0], trade_bins[-1]],
                                 cmap='YlOrRd')
            
            axes[idx].set_title(f'{agent_name}\nTrading Intensity Heatmap', 
                              fontweight='bold')
            axes[idx].set_xlabel('Price')
            axes[idx].set_ylabel('Trade Size')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx], label='Frequency')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f"{save_dir}/imgs/trading_heatmap.png", 
                       dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()
        
    def generate_comparison_report(self, alpha=0.05, alternative='less', save_dir=None):
        """
        Generate a comprehensive comparison report between all pairs of agents.
        
        Args:
            alpha (float): Significance level for the tests
            alternative (str): Direction for the t-test ('less', 'greater', or 'two-sided')
            save_dir (str): Directory to save the report table figure, if provided
            
        Returns:
            dict: Dictionary of comparison results between all agent pairs
        """
        plt.rcParams.update({'font.size': 14})
        if len(self.costs) < 2:
            print("Need at least 2 agents to generate a comparison report.")
            return {}
            
        agent_names = list(self.costs.keys())
        n_agents = len(agent_names)
        comparison_results = {}
        
        # Perform all pairwise comparisons
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                agent1 = agent_names[i]
                agent2 = agent_names[j]
                
                # Perform comparison both ways
                result_1_vs_2 = self.compare_agents(agent1, agent2, alpha, alternative)
                result_2_vs_1 = self.compare_agents(agent2, agent1, alpha, alternative)
                
                comparison_results[f"{agent1} vs {agent2}"] = result_1_vs_2
                comparison_results[f"{agent2} vs {agent1}"] = result_2_vs_1
        
        # Create a visual table of the results
        if save_dir:
            self._plot_comparison_table(comparison_results, alpha, save_dir)
            
        return comparison_results
        
    def _plot_comparison_table(self, comparison_results, alpha, save_dir):
        """Create a visual table of the comparison results"""
        agent_names = list(set([name.split(" vs ")[0] for name in comparison_results.keys()]))
        n_agents = len(agent_names)
        
        # Create figure and axis
        fig = plt.figure(figsize=(n_agents*2 + 2, n_agents*1.5 + 1))
        ax = fig.gca()
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        header = ["Agent"] + agent_names
        table_data.append(header)
        
        for i, row_agent in enumerate(agent_names):
            row = [row_agent]
            for j, col_agent in enumerate(agent_names):
                if i == j:  # Same agent
                    cell = "—"
                else:
                    key = f"{row_agent} vs {col_agent}"
                    if key in comparison_results:
                        result = comparison_results[key]
                        p_value = result['p_value']
                        mean_diff = result['mean_diff']
                        
                        # Format cell content - ensure p_value is a float
                        p_val = float(p_value)  # Convert to float explicitly
                        if result['significant']:
                            if result['better']:
                                cell = f"Better\np={p_val:.4f}*"
                            else:
                                cell = f"Worse\np={p_val:.4f}*"
                        else:
                            cell = f"No diff\np={p_val:.4f}"
                    else:
                        cell = "N/A"
                row.append(cell)
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data[1:], 
                        colLabels=table_data[0],
                        loc='center',
                        cellLoc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.0)
        
        # Add title and notes
        plt.title(f'Pairwise Agent Comparison (α={alpha})', pad=20)
        plt.figtext(0.5, 0.01, 
                   f"*Statistically significant at α={alpha}. 'Better' means lower cost objective.", 
                   ha='center', fontsize=8)
        
        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/imgs/agent_comparison_table.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def display_risk_metrics(self, agent_name=None):
        """
        Display risk metrics for one or all agents in a formatted table.
        
        Args:
            agent_name (str, optional): Specific agent to display. If None, displays all agents.
        """
        plt.rcParams.update({'font.size': 14})
        if not self.risk_metrics:
            print("No risk metrics calculated. Run evaluate_agent() first.")
            return
            
        if agent_name and agent_name not in self.risk_metrics:
            print(f"Agent '{agent_name}' not found in risk metrics.")
            return
            
        agents_to_display = [agent_name] if agent_name else list(self.risk_metrics.keys())
        
        print("\n" + "="*80)
        print("RISK METRICS SUMMARY")
        print("="*80)
        
        for agent in agents_to_display:
            metrics = self.risk_metrics[agent]
            print(f"\nAgent: {agent}")
            print("-" * 50)
            
            # Basic statistics
            print("BASIC STATISTICS:")
            print(f"  Mean:                {metrics['mean']:.6f}")
            print(f"  Std Deviation:       {metrics['std']:.6f}")
            print(f"  Minimum:             {metrics['min']:.6f}")
            print(f"  Maximum:             {metrics['max']:.6f}")
            print(f"  Skewness:            {metrics['skewness']:.6f}")
            print(f"  Kurtosis:            {metrics['kurtosis']:.6f}")
            
            # Percentiles
            print("\nPERCENTILES:")
            print(f"  10th Percentile:     {metrics['percentile_10']:.6f}")
            print(f"  25th Percentile:     {metrics['percentile_25']:.6f}")
            print(f"  50th Percentile:     {metrics['percentile_50']:.6f}")
            print(f"  75th Percentile:     {metrics['percentile_75']:.6f}")
            print(f"  90th Percentile:     {metrics['percentile_90']:.6f}")
            
            # Risk measures
            print("\nRISK MEASURES:")
            print(f"  VaR 95%:             {metrics['VaR_95']:.6f}")
            print(f"  CVaR 95%:            {metrics['CVaR_95']:.6f}")
            print(f"  VaR 99%:             {metrics['VaR_99']:.6f}")
            print(f"  CVaR 99%:            {metrics['CVaR_99']:.6f}")
            print(f"  Semi-deviation:      {metrics['semi_deviation']:.6f}")
            print(f"  Max Drawdown:        {metrics['max_drawdown']:.6f}")
            
            # Risk-adjusted ratios
            print("\nRISK-ADJUSTED RATIOS:")
            print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.6f}")
            print(f"  Sortino Ratio:       {metrics['sortino_ratio']:.6f}")
            
        print("\n" + "="*80)
    
    def plot_risk_metrics(self, plot=True, save_dir=None):
        """
        Create visualizations for risk metrics comparison across agents.
        """
        if not self.risk_metrics:
            print("No risk metrics to plot.")
            return
            
        n_agents = len(self.risk_metrics)
        if n_agents == 0:
            return
            
        agent_names = list(self.risk_metrics.keys())
        
        # Create figure with subplots for different risk metric categories
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Define metrics to plot
        metrics_to_plot = [
            ('VaR_95', 'Value at Risk (95%)', 'VaR 95%'),
            ('CVaR_95', 'Conditional Value at Risk (95%)', 'CVaR 95%'),
            ('VaR_99', 'Value at Risk (99%)', 'VaR 99%'),
            ('CVaR_99', 'Conditional Value at Risk (99%)', 'CVaR 99%'),
            ('semi_deviation', 'Semi-Deviation (Downside Risk)', 'Semi-Deviation'),
            ('max_drawdown', 'Maximum Drawdown', 'Max Drawdown')
        ]
        
        for i, (metric_key, title, ylabel) in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Extract metric values for all agents
            values = [self.risk_metrics[agent][metric_key] for agent in agent_names]
            colors = [self.colors.get(agent, 'gray') for agent in agent_names]
            
            # Create line plot with points
            x_positions = range(len(agent_names))
            for j, (agent, value, color) in enumerate(zip(agent_names, values, colors)):
                ax.plot(j, value, 'o-', color=color, markersize=8, linewidth=2, 
                       label=agent if i == 0 else "")  # Only add legend for first subplot
            
            # Add value labels next to each point
            for j, (agent, value) in enumerate(zip(agent_names, values)):
                ax.text(j, value, f'{value:.4f}', ha='center', va='bottom', 
                       fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(agent_names)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Add legend only to the first subplot
            if i == 0:
                ax.legend(loc='best')
            
            # Rotate x-axis labels if many agents
            if len(agent_names) > 3:
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/imgs/risk_metrics.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()
            
    def plot_risk_comparison_radar(self, plot=True, save_dir=None):
        """
        Create a radar chart comparing risk metrics across agents.
        """
        plt.rcParams.update({'font.size': 14})
        if not self.risk_metrics:
            print("No risk metrics to plot.")
            return
            
        agent_names = list(self.risk_metrics.keys())
        if len(agent_names) == 0:
            return
            
        # Select key risk metrics for radar chart (normalize them)
        risk_metrics_keys = ['CVaR_95', 'CVaR_99', 'semi_deviation', 'max_drawdown', 'std']
        risk_labels = ['CVaR 95%', 'CVaR 99%', 'Semi-Dev', 'Max DD', 'Std Dev']
        
        # Prepare data - normalize each metric to 0-1 scale
        normalized_data = {}
        for metric in risk_metrics_keys:
            all_values = [self.risk_metrics[agent][metric] for agent in agent_names]
            min_val, max_val = min(all_values), max(all_values)
            
            # Normalize to 0-1 (higher values = higher risk)
            if max_val > min_val:
                normalized_data[metric] = [(val - min_val) / (max_val - min_val) for val in all_values]
            else:
                normalized_data[metric] = [0.5] * len(all_values)  # All equal
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(risk_metrics_keys), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for i, agent in enumerate(agent_names):
            values = [normalized_data[metric][i] for metric in risk_metrics_keys]
            values += values[:1]  # Complete the circle
            
            color = self.colors.get(agent, 'gray')
            ax.plot(angles, values, 'o-', linewidth=2, label=agent, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # Customize the radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(risk_labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Risk Metrics Comparison (Normalized)\nHigher values = Higher risk', 
                 size=14, pad=20)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/imgs/risk_radar_chart.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()
            
    def plot_control_histograms(self, plot=True, save_dir=None, timestep_idx=None, exclude_zeros=True):
        """
        Plot histograms of control values (q) for each agent.
        
        Args:
            plot (bool): Whether to show the plots interactively
            save_dir (str): Directory to save the plots, if provided
            timestep_idx (int or list, optional): Specific timestep index(es) to plot.
                          If None, plots the average control across all timesteps.
                          If int, plots the control at that specific timestep.
                          If list, plots the control for each timestep in the list.
            exclude_zeros (bool): If True, excludes zero values from histogram to better show
                                 the distribution of actual trades. Default is True.
        """
        plt.rcParams.update({'font.size': 14})
        if not self.results:
            print("No results to plot.")
            return
            
        n_agents = len(self.results)
        if n_agents == 0:
            return
            
        # Handle different timestep_idx options
        if isinstance(timestep_idx, (list, tuple)):
            # Multiple specific timesteps
            timestep_indices = timestep_idx
            title_suffix = f"at timesteps {timestep_indices}"
        elif isinstance(timestep_idx, int):
            # Single specific timestep
            timestep_indices = [timestep_idx]
            title_suffix = f"at timestep {timestep_idx}"
        else:
            # Average across all timesteps
            timestep_indices = None
            title_suffix = "(averaged across all timesteps)"
            
        fig, axs = plt.subplots(n_agents, 1, figsize=(10, 6 * n_agents), squeeze=False)
        
        for i, (agent_name, data) in enumerate(self.results.items()):
            ax = axs[i, 0]
            color = self.colors.get(agent_name, 'gray')
            
            # Get the control values for this agent
            results = data['results']
            q_values = results.get('q_learned', None)
            
            if q_values is None:
                ax.text(0.5, 0.5, f"No control data available for {agent_name}", 
                        ha='center', va='center', transform=ax.transAxes)
                continue
                
            # Convert to numpy array if needed
            q_values = np.asarray(q_values)
            
            # Extract data based on timestep_idx
            if timestep_indices is not None:
                # Only use specific timesteps
                q_data = []
                for idx in timestep_indices:
                    if idx < q_values.shape[0]:
                        q_data.append(q_values[idx].flatten())
                    else:
                        print(f"Warning: Timestep {idx} is out of range for {agent_name}")
                        
                if not q_data:
                    ax.text(0.5, 0.5, f"No valid timesteps for {agent_name}", 
                            ha='center', va='center', transform=ax.transAxes)
                    continue
                    
                q_data = np.concatenate(q_data)
            else:
                # Average across all timesteps
                q_data = q_values.flatten()
            
            # Store original data for statistics
            q_data_original = q_data.copy()
            
            # Exclude zeros if requested
            if exclude_zeros:
                q_data_nonzero = q_data[q_data != 0]
                if len(q_data_nonzero) == 0:
                    ax.text(0.5, 0.5, f"All control values are zero for {agent_name}", 
                            ha='center', va='center', transform=ax.transAxes)
                    continue
                q_data_plot = q_data_nonzero
                zero_count = len(q_data) - len(q_data_nonzero)
                zero_percentage = (zero_count / len(q_data)) * 100
                title_suffix_full = f"{title_suffix} (excl. {zero_count} zeros: {zero_percentage:.1f}%)"
            else:
                q_data_plot = q_data
                title_suffix_full = title_suffix
            
            # Plot histogram
            ax.hist(q_data_plot, bins='auto', color=color, alpha=0.7, label='Control Distribution')
            
            # Calculate statistics (always use original data for complete picture)
            mean_q = np.mean(q_data_original)
            std_q = np.std(q_data_original)
            min_q = np.min(q_data_original)
            max_q = np.max(q_data_original)
            
            # Calculate statistics for non-zero data if excluding zeros
            if exclude_zeros and len(q_data_plot) > 0:
                mean_q_nonzero = np.mean(q_data_plot)
                std_q_nonzero = np.std(q_data_plot)
                min_q_nonzero = np.min(q_data_plot)
                max_q_nonzero = np.max(q_data_plot)
            
            # Plot mean line (use appropriate mean based on what's displayed)
            if exclude_zeros and len(q_data_plot) > 0:
                ax.axvline(mean_q_nonzero, color='red', linestyle='dotted', linewidth=2, 
                          label=f'Mean (non-zero): {mean_q_nonzero:.4f}')
            else:
                ax.axvline(mean_q, color='red', linestyle='dotted', linewidth=2, 
                          label=f'Mean: {mean_q:.4f}')
            
            # Calculate and plot mode from histogram
            hist, bin_edges = np.histogram(q_data_plot, bins='auto')
            if len(hist) > 0:
                max_hist_index = np.argmax(hist)
                mode_q = (bin_edges[max_hist_index] + bin_edges[max_hist_index+1]) / 2
                ax.axvline(mode_q, color='green', linestyle='dotted', linewidth=2, label=f'Mode: {mode_q:.4f}')
            
            ax.set_title(f'Control (q) Distribution for {agent_name} {title_suffix_full}')
            ax.set_xlabel("Control Value (q)")
            ax.set_ylabel("Frequency")
            
            # Create legend
            handles, labels = ax.get_legend_handles_labels()
            
            # Add stats to legend (include both original and non-zero stats if relevant)
            if exclude_zeros and len(q_data_plot) > 0:
                handles.append(Patch(color='none', label=f'All data - Mean: {mean_q:.4f}, Std: {std_q:.4f}'))
                handles.append(Patch(color='none', label=f'Non-zero - Std: {std_q_nonzero:.4f}'))
                handles.append(Patch(color='none', label=f'Non-zero - Min: {min_q_nonzero:.4f}, Max: {max_q_nonzero:.4f}'))
            else:
                handles.append(Patch(color='none', label=f'Std Dev: {std_q:.4f}'))
                handles.append(Patch(color='none', label=f'Min: {min_q:.4f}'))
                handles.append(Patch(color='none', label=f'Max: {max_q:.4f}'))
            
            ax.legend(handles=handles, loc="upper left")
            ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        if save_dir:
            if timestep_indices is not None:
                # Create a string representation of the timestep indices for the filename
                ts_str = "_".join(map(str, timestep_indices)) if isinstance(timestep_indices, (list, tuple)) else str(timestep_indices)
                filename = f"control_histograms_t{ts_str}"
            else:
                filename = "control_histograms"
            
            # Add suffix for zero exclusion
            if exclude_zeros:
                filename += "_no_zeros"
            
            plt.savefig(f"{save_dir}/imgs/{filename}.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()
    
    def plot_terminal_cost_analysis(self, plot=True, save_dir=None):
        """
        Create a detailed analysis of terminal costs showing:
        1. Final terminal cost distribution across simulations
        2. Position-demand gap evolution over time
        3. Terminal cost evolution over time
        """
        if not self.results:
            print("No results to plot.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Subplot 1: Final terminal cost distribution (histogram)
        ax1 = axes[0]
        # Subplot 2: Position-demand gap evolution
        ax2 = axes[1] 
        # Subplot 3: Terminal cost evolution over time
        ax3 = axes[2]
        # Subplot 4: Summary statistics table
        ax4 = axes[3]
        ax4.axis('off')  # Turn off axis for table
        
        # Collect summary data for table
        summary_data = []
        summary_headers = ['Agent', 'Final Terminal Cost\n(Mean ± Std)', 'Final Gap\n(Mean ± Std)', 'Max Terminal Cost']
        
        for agent_name, data in self.results.items():
            timesteps = data['timesteps']
            results = data['results']
            color = self.colors[agent_name]
            
            y_traj = results.get('y_learned', None)
            if y_traj is None:
                continue
                
            y_traj = np.asarray(y_traj)
            
            # Extract final states for terminal cost calculation
            final_states = y_traj[-1, :, :]  # Last timestep, all simulations
            
            # Calculate final terminal costs
            final_terminal_costs = []
            for sim in range(final_states.shape[0]):
                y_final = torch.from_numpy(final_states[sim:sim+1, :]).to(self.device)
                if hasattr(self.dynamics, 'terminal_cost'):
                    tc = self.dynamics.terminal_cost(y_final)
                    final_terminal_costs.append(tc.detach().cpu().numpy().item())
                else:
                    # Fallback calculation
                    X_final = y_final[:, 0:1]
                    if y_final.shape[-1] > 2:
                        D_final = y_final[:, 2:3]
                        eta = getattr(self.dynamics, 'eta', 1.0)
                        tc = 0.5 * eta * (D_final - X_final)**2
                        final_terminal_costs.append(tc.detach().cpu().numpy().item())
            
            final_terminal_costs = np.array(final_terminal_costs)
            
            # Calculate position-demand gap if demand exists
            if y_traj.shape[-1] > 2:
                X_traj = y_traj[:, :, 0]
                D_traj = y_traj[:, :, 2]
                gap_traj = D_traj - X_traj
                gap_mean = gap_traj.mean(axis=1)
                gap_std = gap_traj.std(axis=1)
                final_gap_mean = gap_traj[-1, :].mean()
                final_gap_std = gap_traj[-1, :].std()
                
                # Plot position-demand gap evolution
                ax2.plot(timesteps, gap_mean, color=color, linewidth=2, label=agent_name)
                ax2.fill_between(timesteps, gap_mean - gap_std, gap_mean + gap_std, 
                               color=color, alpha=0.2)
            else:
                final_gap_mean = 0.0
                final_gap_std = 0.0
            
            # Calculate terminal cost evolution
            terminal_costs_time = np.zeros((y_traj.shape[0], y_traj.shape[1]))
            for t in range(y_traj.shape[0]):
                for sim in range(y_traj.shape[1]):
                    y_t = torch.from_numpy(y_traj[t:t+1, sim:sim+1, :]).to(self.device)
                    if hasattr(self.dynamics, 'terminal_cost'):
                        tc = self.dynamics.terminal_cost(y_t.squeeze(1))
                        terminal_costs_time[t, sim] = tc.detach().cpu().numpy().item()
                    else:
                        X_t = y_t[:, :, 0:1]
                        if y_t.shape[-1] > 2:
                            D_t = y_t[:, :, 2:3]
                            eta = getattr(self.dynamics, 'eta', 1.0)
                            tc = 0.5 * eta * (D_t - X_t)**2
                            terminal_costs_time[t, sim] = tc.detach().cpu().numpy().item()
            
            tc_time_mean = terminal_costs_time.mean(axis=1)
            tc_time_std = terminal_costs_time.std(axis=1)
            
            # Plot 1: Final terminal cost histogram
            ax1.hist(final_terminal_costs, bins=20, alpha=0.7, color=color, 
                    label=f'{agent_name} (μ={np.mean(final_terminal_costs):.4f})')
            
            # Plot 3: Terminal cost evolution
            ax3.plot(timesteps, tc_time_mean, color=color, linewidth=2, label=agent_name)
            ax3.fill_between(timesteps, tc_time_mean - tc_time_std, tc_time_mean + tc_time_std,
                           color=color, alpha=0.2)
            
            # Add to summary table
            summary_data.append([
                agent_name,
                f'{np.mean(final_terminal_costs):.4f} ± {np.std(final_terminal_costs):.4f}',
                f'{final_gap_mean:.4f} ± {final_gap_std:.4f}',
                f'{np.max(final_terminal_costs):.4f}'
            ])
        
        # Customize subplots
        ax1.set_title('Final Terminal Cost Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Terminal Cost')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Position-Demand Gap Evolution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Gap: D(t) - X(t)')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Perfect Execution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_title('Terminal Cost Evolution Over Time', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Terminal Cost: 0.5η(D-X)²')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Create summary table
        if summary_data:
            table = ax4.table(cellText=summary_data, colLabels=summary_headers,
                             loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2.0)
            ax4.set_title('Terminal Cost Summary Statistics', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f"{save_dir}/imgs/terminal_cost_analysis.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()
        
    def _get_trajectory_colors_and_styles(self, n_traj):
        """
        Helper method to ensure consistent colors and line styles across all trajectory plots.
        Returns colors and line styles that should be used consistently.
        """
        # Define line styles for different agents
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
        
        # Get colormap for trajectories - use consistent colormap
        total_trajectories = len(self.results) * n_traj
        colors = cm.get_cmap("tab20", max(total_trajectories, 20))
        
        return colors, line_styles

    def plot_state_trajectories(self, n_traj=5, plot=True, save_dir=None):
        """
        Plot individual trajectories of state variables (X and D) for each agent.
        X and D are plotted in the same subplot with different markers.
        Each trajectory gets a unique color, different agents use different line styles.
        
        Args:
            n_traj (int): Number of individual trajectories to plot per agent
            plot (bool): Whether to show the plots interactively
            save_dir (str): Directory to save the plots, if provided
        """
        plt.rcParams.update({'font.size': 14})
        if not self.results:
            print("No results to plot.")
            return
            
        # Check if n_traj is valid
        if n_traj > self.n_sim:
            raise ValueError(f"n_traj ({n_traj}) cannot be larger than n_simulations ({self.n_sim})")
            
        # Get consistent colors and line styles
        colors, line_styles = self._get_trajectory_colors_and_styles(n_traj)
            
        # Create figure with single subplot for both X and D
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Track agent handles for legend
        agent_handles = []
        trajectory_idx = 0
        
        for agent_idx, (agent_name, data) in enumerate(self.results.items()):
            timesteps = data['timesteps']
            results = data['results']
            linestyle = line_styles[agent_idx % len(line_styles)]
            
            # Get state trajectories
            y_traj = results.get('y_learned', None)
            if y_traj is None:
                continue
                
            y_traj = np.asarray(y_traj)  # Shape: (N+1, n_sim, state_dim)
            
            # Extract X (cumulative position) and D (demand) if available
            X_traj = y_traj[:, :, 0]  # Cumulative position
            if y_traj.shape[-1] > 2:
                D_traj = y_traj[:, :, 2]  # Demand (if available)
            else:
                D_traj = None
            
            # Plot individual X and D trajectories - each with unique color but consistent across plots
            for i in range(min(n_traj, X_traj.shape[1])):
                color = colors(trajectory_idx)
                alpha = 0.8  # Make trajectories clearly visible
                
                # Plot X trajectory (cumulative position)
                line_label_X = f"{agent_name} X (traj {i+1})"
                ax.plot(timesteps, X_traj[:, i], color=color, alpha=alpha, 
                       linestyle=linestyle, linewidth=1.5, label=line_label_X,
                       marker='o', markersize=3, markevery=len(timesteps)//10)
                
                # Plot D trajectory (demand) if available
                if D_traj is not None:
                    line_label_D = f"{agent_name} D (traj {i+1})"
                    ax.plot(timesteps, D_traj[:, i], color=color, alpha=alpha*0.7, 
                           linestyle=linestyle, linewidth=1.2, label=line_label_D,
                           marker='s', markersize=2, markevery=len(timesteps)//10)
                
                trajectory_idx += 1
            
            # Store handle for agent legend (showing line style)
            agent_handles.append(plt.Line2D([0], [0], color='black', linestyle=linestyle, 
                                          linewidth=2, label=f"{agent_name} (style)"))
        
        # Customize subplot with DeepAgent style
        ax.set_title(f"State Trajectories $x(t)$ and $d(t)$: {n_traj} individual paths per agent")
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("State Values")
        ax.grid(True)
        
        # Create custom legend handles for X and D distinction
        x_handle = plt.Line2D([0], [0], color='gray', marker='o', markersize=6, 
                             label='X (position)', linestyle='-', linewidth=2)
        d_handle = plt.Line2D([0], [0], color='gray', marker='s', markersize=4, 
                             label='D (demand)', linestyle='-', linewidth=1.5, alpha=0.7)
        
        # Three legends: agents, X/D distinction, and individual trajectories
        leg1 = ax.legend(handles=agent_handles, title="Agents", loc='upper right', fontsize=9)
        ax.add_artist(leg1)
        
        if any('D_traj' in locals() and D_traj is not None for _, data in self.results.items()):
            leg2 = ax.legend(handles=[x_handle, d_handle], title="Variables", loc='upper left', fontsize=9)
            ax.add_artist(leg2)
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7, title="Trajectories")
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/imgs/state_trajectories_n{n_traj}.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()
    
    def plot_control_trajectories(self, n_traj=5, plot=True, save_dir=None):
        """
        Plot individual trajectories of control variables (q) for each agent.
        Each trajectory gets a unique color (consistent with other plots), different agents use different line styles.
        
        Args:
            n_traj (int): Number of individual trajectories to plot per agent
            plot (bool): Whether to show the plots interactively
            save_dir (str): Directory to save the plots, if provided
        """
        plt.rcParams.update({'font.size': 14})
        if not self.results:
            print("No results to plot.")
            return
            
        # Check if n_traj is valid
        if n_traj > self.n_sim:
            raise ValueError(f"n_traj ({n_traj}) cannot be larger than n_simulations ({self.n_sim})")
            
        # Get consistent colors and line styles
        colors, line_styles = self._get_trajectory_colors_and_styles(n_traj)
            
        # Create figure with DeepAgent style
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Track agent handles for legend
        agent_handles = []
        trajectory_idx = 0
        
        for agent_idx, (agent_name, data) in enumerate(self.results.items()):
            timesteps = data['timesteps']
            results = data['results']
            linestyle = line_styles[agent_idx % len(line_styles)]
            
            # Get control trajectories
            q_traj = results.get('q_learned', None)
            if q_traj is None:
                continue
                
            q_traj = np.asarray(q_traj)  # Shape: (N, n_sim, control_dim)
            
            # Use timesteps for controls (exclude last timestep since no control at terminal time)
            control_times = timesteps[:-1]
            
            # Plot individual control trajectories - each with consistent color from other plots
            for i in range(min(n_traj, q_traj.shape[1])):
                # Squeeze if control is 1-dimensional
                if q_traj.shape[-1] == 1:
                    q_i = q_traj[:, i, 0]
                else:
                    q_i = q_traj[:, i, 0]  # Plot first control dimension if multi-dimensional
                
                color = colors(trajectory_idx)  # Same color as corresponding trajectory in other plots
                alpha = 0.8  # Make trajectories clearly visible
                line_label = f"{agent_name} (traj {i+1})"
                ax.plot(control_times, q_i, color=color, alpha=alpha, 
                       linestyle=linestyle, linewidth=1.5, label=line_label)
                trajectory_idx += 1
            
            # Store handle for agent legend (showing line style)
            agent_handles.append(plt.Line2D([0], [0], color='black', linestyle=linestyle, 
                                          linewidth=2, label=f"{agent_name} (style)"))
        
        # Customize subplot with DeepAgent style
        ax.set_title(f"Control $q(t)$: {n_traj} individual paths per agent")
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("$q(t)$")
        ax.grid(True)
        # Two legends: one for agents (line styles), one for individual trajectories
        leg1 = ax.legend(handles=agent_handles, title="Agents", loc='upper right', fontsize=9)
        ax.add_artist(leg1)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, title="Trajectories")
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/imgs/control_trajectories_n{n_traj}.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()
