import numpy as np
import matplotlib.pyplot as plt

def plot_all_diagnostics(results, timesteps):

    q_vals = results["q"]  # shape: (T, N)
    Y_vals = results["Y"]
    X_T, D_T, B_T, I_T = results["X_T"], results["D_T"], results["B_T"], results["I_T"]

    # Retrieve the state trajectories (assumed shape: (N*T, 4))
    y_all = results["final_y"]  # shape: (T * n_paths, 4)
    n_paths = q_vals.shape[1]
    T = q_vals.shape[0]
    state_trajectories = y_all.reshape(T, -1, 4)

    mean_states = state_trajectories.mean(axis=1)
    std_states = state_trajectories.std(axis=1)
    ci_states = 1.96 * std_states / np.sqrt(n_paths)

    fig, axs = plt.subplots(3, 2, figsize=(16, 14))

    # Subplot 1: Control over time
    mean_q = q_vals.mean(axis=1)
    std_q = q_vals.std(axis=1)
    ci_q = 1.96 * std_q / np.sqrt(n_paths)
    axs[0, 0].plot(timesteps, mean_q, label="Mean $q(t)$")
    axs[0, 0].fill_between(timesteps, mean_q - ci_q, mean_q + ci_q, alpha=0.3, label="95% CI")
    axs[0, 0].set_title("Optimal Trading Rate $q(t)$")
    axs[0, 0].set_xlabel("Time $t$")
    axs[0, 0].set_ylabel("$q(t)$")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Subplot 2: Value function over time
    for i in range(Y_vals.shape[1]):  # Iterate over all trajectories
        axs[0, 1].plot(timesteps, Y_vals[:, i, 0], alpha=0.1, color="blue")
    axs[0, 1].set_title("Cost-to-Go $Y(t)$ (Value Function)")
    axs[0, 1].set_xlabel("Time $t$")
    axs[0, 1].set_ylabel("$Y(t)$")
    axs[0, 1].grid(True)
    # mean_Y = Y_vals.mean(axis=1).squeeze()
    # std_Y = Y_vals.std(axis=1).squeeze()
    # ci_Y = 1.96 * std_Y / np.sqrt(Y_vals.shape[1])
    # axs[0, 1].plot(timesteps, mean_Y, label="Mean $Y(t)$")
    # axs[0, 1].fill_between(timesteps, mean_Y - ci_Y, mean_Y + ci_Y, alpha=0.3, label="95% CI")
    # axs[0, 1].set_title("Cost-to-Go $Y(t)$ (Value Function)")
    # axs[0, 1].set_xlabel("Time $t$")
    # axs[0, 1].set_ylabel("$Y(t)$")
    # axs[0, 1].grid(True)
    # axs[0, 1].legend()

    # Subplot 3: Scatter of imbalance vs B(T)
    axs[1, 0].scatter(B_T, I_T, alpha=0.3, s=10)
    axs[1, 0].set_xlabel("Terminal Imbalance Price $B(T)$")
    axs[1, 0].set_ylabel("Imbalance $I(T)$")
    axs[1, 0].set_title("Imbalance $I(T)$ vs. Imbalance Price $B(T)$")
    axs[1, 0].grid(True)

    # Subplot 4: X(T) vs D(T)
    axs[1, 1].scatter(D_T, X_T, alpha=0.3, s=10)
    corr = np.corrcoef(X_T, D_T)[0, 1]
    axs[1, 1].set_title(f"$X(T)$ vs. $D(T)$ (Corr = {corr:.3f})")
    axs[1, 1].set_xlabel("$D(T)$ (Residual Demand)")
    axs[1, 1].set_ylabel("$X(T)$ (Cumulative Position)")
    axs[1, 1].grid(True)

    # Subplot 5: All states with confidence bands
    labels = ["$X(t)$ (Cumulative)", "$P(t)$ (Mid Price)", "$D(t)$ (Demand)", "$B(t)$ (Imbalance Price)"]
    colors = ["blue", "orange", "green", "red"]
    for i in range(4):
        alpha = 0.5 if n_paths < 100 else 0.1
        for j in range(n_paths):
            if j == 0:
                axs[2, i % 2].plot(timesteps, state_trajectories[:, j, i], label=labels[i], alpha=alpha, color=colors[i])
            else:
                axs[2, i % 2].plot(timesteps, state_trajectories[:, j, i], alpha=alpha, color=colors[i])

        # axs[2, i % 2].plot(timesteps, mean_states[:, i], label=f"Mean {labels[i]}")
        # axs[2, i % 2].fill_between(timesteps,
        #                            mean_states[:, i] - ci_states[:, i],
        #                            mean_states[:, i] + ci_states[:, i],
        #                            alpha=0.3, label="95% CI")
        axs[2, i % 2].set_title(labels[i])
        axs[2, i % 2].set_xlabel("Time $t$")
        axs[2, i % 2].set_ylabel(labels[i])
        axs[2, i % 2].grid(True)
        axs[2, i % 2].legend()

    plt.tight_layout()
    plt.show()