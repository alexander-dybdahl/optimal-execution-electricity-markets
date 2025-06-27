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
        self.results[agent_name] = {
            'timesteps': timesteps,
            'results': results
        }

        self.costs[agent_name] = cost_objective.detach().cpu().numpy()

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
            
            ax.legend(handles=handles)
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/imgs/cost_histograms.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()
            
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
