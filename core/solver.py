import numpy as np
import torch
import matplotlib.pyplot as plt
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

    def plot_traj(self, plot=True, save_dir=None):
        if not self.results:
            print("No results to plot.")
            return
        
        # Plot q, y, Y as subplots in a single figure
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        var_titles = {'q': 'Control $q(t)$', 'y': 'State $y(t)$', 'Y': 'Cost-to-Go $Y(t)$'}
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

            ax.set_title(var_titles[var])
            ax.set_xlabel("Time")
            ax.set_ylabel(var_ylabels[var])
            ax.grid(True, linestyle='--', alpha=0.5)
            # Add two legends: one for agent color, one for line style/area
            leg1 = ax.legend(handles=agent_handles, title="Agent (mean)", loc='upper right', frameon=True)
            ax.add_artist(leg1)
            ax.legend(handles=style_handles, title="Area", loc='lower right', frameon=True)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/imgs/trajectories_all.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()

    def plot_cost_histograms(self, plot=True, save_dir=None):
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
        if not self.results:
            print("No results to plot.")
            return
        
        # Create a comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Define subplot layout: 3 rows, 2 columns
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.25)
        
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
        # Subplot 7: Cumulative Trade Value vs Time
        ax7 = fig.add_subplot(gs[3, :])  # Spans both columns
        
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
            
            # Trade times (excluding t=0 since no trades happen then)
            trade_times = timesteps[:-1]
            
            # 1. Trade Size vs Execution Price (scatter plot)
            ax1.scatter(exec_price_mean[q_mean != 0], q_mean[q_mean != 0], 
                       color=color, alpha=0.7, s=50, label=agent_name)
            
            # 2. Cumulative Position vs Time
            ax2.plot(timesteps, X_mean, color=color, linewidth=2, label=agent_name)
            ax2.fill_between(timesteps, X_mean - X_std, X_mean + X_std, 
                           color=color, alpha=0.2)
            
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
            
            # 6. Trade Value vs Time
            trade_values = q_mean * exec_price_mean
            ax6.plot(trade_times, trade_values, color=color, linewidth=2, 
                    marker='^', markersize=3, label=agent_name)
            
            # 7. Cumulative Trade Value vs Time
            cumulative_trade_value = np.cumsum(np.concatenate([[0], trade_values]))
            ax7.plot(timesteps, cumulative_trade_value, color=color, linewidth=3, 
                    label=agent_name)
            
            # Store handle for legend
            if agent_name not in [h.get_label() for h in agent_handles]:
                agent_handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=agent_name))
        
        # Customize subplots
        ax1.set_title('Trade Size vs Execution Price', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Execution Price')
        ax1.set_ylabel('Trade Size')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_title('Cumulative Position vs Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Cumulative Position')
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
        
        ax6.set_title('Trade Values vs Time', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Time')
        ax6.set_ylabel('Trade Value (Size × Price)')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax6.legend()
        
        ax7.set_title('Cumulative Trade Value vs Time', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Time')
        ax7.set_ylabel('Cumulative Trade Value')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        
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
            
    def plot_control_histograms(self, plot=True, save_dir=None, timestep_idx=None):
        """
        Plot histograms of control values (q) for each agent.
        
        Args:
            plot (bool): Whether to show the plots interactively
            save_dir (str): Directory to save the plots, if provided
            timestep_idx (int or list, optional): Specific timestep index(es) to plot.
                          If None, plots the average control across all timesteps.
                          If int, plots the control at that specific timestep.
                          If list, plots the control for each timestep in the list.
        """
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
            
            # Plot histogram
            ax.hist(q_data, bins='auto', color=color, alpha=0.7, label='Control Distribution')
            
            # Calculate statistics
            mean_q = np.mean(q_data)
            std_q = np.std(q_data)
            min_q = np.min(q_data)
            max_q = np.max(q_data)
            
            # Plot mean line
            ax.axvline(mean_q, color='red', linestyle='dotted', linewidth=2, label=f'Mean: {mean_q:.4f}')
            
            # Calculate and plot mode from histogram
            hist, bin_edges = np.histogram(q_data, bins='auto')
            if len(hist) > 0:
                max_hist_index = np.argmax(hist)
                mode_q = (bin_edges[max_hist_index] + bin_edges[max_hist_index+1]) / 2
                ax.axvline(mode_q, color='green', linestyle='dotted', linewidth=2, label=f'Mode: {mode_q:.4f}')
            
            ax.set_title(f'Control (q) Distribution for {agent_name} {title_suffix}')
            ax.set_xlabel("Control Value (q)")
            ax.set_ylabel("Frequency")
            
            # Create legend
            handles, labels = ax.get_legend_handles_labels()
            
            # Add stats to legend
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
                plt.savefig(f"{save_dir}/imgs/control_histograms_t{ts_str}.png", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f"{save_dir}/imgs/control_histograms.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()
    