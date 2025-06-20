import numpy as np
import torch
import matplotlib.pyplot as plt

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

    def evaluate_agent(self, agent, agent_name, analytical=True):
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
            analytical=analytical
        )
        # Compute cost objective
        cost_objective = compute_cost_objective(
            dynamics=self.dynamics,
            q_traj=torch.from_numpy(results["q_learned"]).to(self.device),
            y_traj=torch.from_numpy(results["y_learned"]).to(self.device)
        )
        self.results[agent_name] = {
            'timesteps': timesteps,
            'results': results
        }

        self.costs[agent_name] = cost_objective.mean().item()

    def plot_traj(self, plot=True, save_dir=None):
        if not self.results:
            print("No results to plot.")
            return
        
        # Plot q, y, Y as subplots in a single figure, with learned/analytic as different line styles
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        var_titles = {'q': 'Control $q(t)$', 'y': 'State $y(t)$', 'Y': 'Cost-to-Go $Y(t)$'}
        var_ylabels = {'q': '$q(t)$', 'y': '$y(t)$', 'Y': '$Y(t)$'}
        line_styles = {'learned': '-', 'analytic': '--'}
        style_labels = {'-': 'Learned', '--': 'Analytic'}
        agent_handles = []
        for agent_name, color in self.colors.items():
            agent_handles.append(plt.Line2D([0], [0], color=color, linestyle='-', label=agent_name))
        style_handles = [plt.Line2D([0], [0], color='k', linestyle=ls, label=style_labels[ls]) for ls in line_styles.values()]
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
                    if arr_mean.ndim > 1 and arr_mean.shape[-1] == 1:
                        arr_mean = arr_mean.squeeze(-1)
                    ax.plot(
                        timesteps[:arr_mean.shape[0]],
                        arr_mean,
                        color=color,
                        linestyle=line_styles['learned'],
                        alpha=1.0
                    )
                # Plot analytical if available
                key_analytical = f'{var}_analytical'
                arr_analytical = results.get(key_analytical, None)
                if arr_analytical is not None:
                    arr_analytical = np.asarray(arr_analytical)
                    arr_mean = arr_analytical.mean(axis=1)
                    if arr_mean.ndim > 1 and arr_mean.shape[-1] == 1:
                        arr_mean = arr_mean.squeeze(-1)
                    ax.plot(
                        timesteps[:arr_mean.shape[0]],
                        arr_mean,
                        color=color,
                        linestyle=line_styles['analytic'],
                        linewidth=2,
                        alpha=0.7
                    )
            ax.set_title(var_titles[var])
            ax.set_xlabel("Time")
            ax.set_ylabel(var_ylabels[var])
            ax.grid(True, linestyle='--', alpha=0.5)
            # Add two legends: one for agent color, one for line style
            leg1 = ax.legend(handles=agent_handles, title="Agent (Color)", loc='upper right', frameon=True)
            ax.add_artist(leg1)
            ax.legend(handles=style_handles, title="Line Style", loc='lower right', frameon=True)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/imgs/trajectories_all.png", dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()
