import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

# Plot training losses from a CSV file in the model directory
def plot_training_losses(model_dir, save_dir, plot=True):
    """
    Plot training losses from CSV file in the model directory.
    
    Args:
        model_dir (str): Directory containing the training CSV file
        save_dir (str): Directory to save the plot
        plot (bool): Whether to show the plot interactively
    """
    # Look for CSV files in the model directory
    csv_files = [f for f in os.listdir(model_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {model_dir}")
        return
    
    # Use the first CSV file found (assuming there's one training log)
    csv_path = os.path.join(model_dir, csv_files[0])
    print(f"Loading training losses from: {csv_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        required_cols = ['Epoch', 'Y_val_loss', 'q_val_loss']
        if not all(col in df.columns for col in required_cols):
            print(f"Required columns not found in CSV. Available columns: {list(df.columns)}")
            return
        
        # Calculate other losses sum (excluding Total_loss and cost_loss)
        other_loss_cols = []
        for col in df.columns:
            if col not in ['Epoch', 'Y_val_loss', 'q_val_loss', 'Total_loss', 'cost_loss'] and 'loss' in col.lower():
                other_loss_cols.append(col)
        
        if other_loss_cols:
            # Sum other losses, handling potential negative values
            df['Train_losses_sum'] = df[other_loss_cols].sum(axis=1)
        else:
            # If no other loss columns, create a zero column
            df['Train_losses_sum'] = 0
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot validation losses
        plt.plot(df['Epoch'], df['Y_val_loss'], color='#1f77b4', linewidth=2, label='Y Validation Loss')  # Blue
        plt.plot(df['Epoch'], df['q_val_loss'], color='#ff7f0e', linewidth=2, label='q Validation Loss')  # Orange
        
        # Plot other losses sum
        plt.plot(df['Epoch'], df['Train_losses_sum'], color='#2ca02c', linewidth=2, label='Train Losses')  # Green
        
        # If cost_loss exists, plot it separately (can be negative)
        # if 'cost_loss' in df.columns:
        #     plt.plot(df['Epoch'], df['cost_loss'], color='#d62728', linewidth=2, label='Cost Loss')  # Red
        
        plt.xlabel('Epoch', fontsize=28)
        plt.ylabel('Loss', fontsize=28)
        plt.legend(fontsize=24)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=24)
        
        # Use log scale if values span multiple orders of magnitude
        y_values = np.concatenate([df['Y_val_loss'].values, df['q_val_loss'].values, df['Train_losses_sum'].values])
        if np.max(y_values) / np.min(y_values[y_values > 0]) > 100:
            plt.yscale('log')
            plt.ylabel('Loss (log scale)', fontsize=28)
        
        plt.tight_layout()
        
        # Save the plot
        if save_dir:
            os.makedirs(os.path.join(save_dir, "imgs"), exist_ok=True)
            plt.savefig(os.path.join(save_dir, "imgs", "training_losses.png"), dpi=300, bbox_inches='tight')
            print(f"Training losses plot saved to: {os.path.join(save_dir, 'imgs', 'training_losses.png')}")
        
        if plot:
            plt.show()
        else:
            plt.close()
            
        # Print summary statistics
        print("\nTraining Loss Summary:")
        print("-" * 40)
        print(f"Final Y validation loss: {df['Y_val_loss'].iloc[-1]:.6f}")
        print(f"Final q validation loss: {df['q_val_loss'].iloc[-1]:.6f}")
        print(f"Final train losses sum: {df['Train_losses_sum'].iloc[-1]:.6f}")
        if 'cost_loss' in df.columns:
            print(f"Final cost loss: {df['cost_loss'].iloc[-1]:.6f}")
        print(f"Epochs trained: {len(df)}")
        
        if other_loss_cols:
            print(f"Other loss components: {', '.join(other_loss_cols)}")
        
    except Exception as e:
        print(f"Error plotting training losses: {e}")
        print(f"CSV file path: {csv_path}")

# All these plotting functions assume that the results include both learned and analytical results
def plot_approx_vs_analytic(results, timesteps, validation=None, plot=True, save_dir=None, num=None, same_color=False):

    approx_q = results["q_learned"]
    y_vals = results["y_learned"]
    Y_vals = results["Y_learned"]
    true_q = results["q_analytical"]
    true_y = results["y_analytical"]
    true_Y = results["Y_analytical"]
    
    if validation is not None:
        val_q_loss = validation["q_loss"]
        val_Y_loss = validation["Y_loss"]

    fig, axs = plt.subplots(3, 2, figsize=(18, 12))
    # Use standard matplotlib color cycle instead of tab10
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if same_color:
        # All learned trajectories use the first color, analytical uses red
        learned_color = color_cycle[0]  # First color in cycle
        analytical_color = 'red'
        colors = lambda i: learned_color
        analytical_colors = lambda i: analytical_color
    else:
        # Each trajectory uses its own color from the color cycle
        colors = lambda i: color_cycle[i % len(color_cycle)]
        analytical_colors = lambda i: color_cycle[i % len(color_cycle)]

    for i in range(approx_q.shape[1]):
        if true_q is not None:
            axs[0, 0].plot(timesteps[:-1], true_q[:, i], linestyle="--", color=analytical_colors(i), alpha=0.7, linewidth=1.5, label=f"Analytical $\\bar{{q}}(t)$" if i == 0 else None)
        axs[0, 0].plot(timesteps[:-1], approx_q[:, i], color=colors(i), linewidth=2, label=f"Learned $q^\\theta(t)$" if i == 0 else None)
    axs[0, 0].set_title("Control $q(t)$: Learned vs Analytical", fontsize=20)
    axs[0, 0].set_xlabel("Time $t$", fontsize=18)
    axs[0, 0].set_ylabel("$q(t)$", fontsize=18)
    axs[0, 0].grid(True, linestyle='--')
    axs[0, 0].legend(loc='upper left', fontsize=16)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=14)

    if validation is not None and true_q is not None:
        for i in range(approx_q.shape[1]):
            # Calculate relative error: |q - q_true| / |q_true|, with small epsilon to avoid division by zero
            eps = 1e-8
            rel_error = np.abs(approx_q[:, i] - true_q[:, i]) / (np.abs(true_q[:, i]) + eps)
            axs[0, 1].plot(timesteps[:-1], rel_error, color=colors(i), label=f"Rel. Error $q(t)$ ({val_q_loss[-1]:.2f})" if i == 0 else None)
        axs[0, 1].axhline(0, color='red', linestyle='--', linewidth=0.8)
        axs[0, 1].set_title("Relative Error in Control $q(t)$", fontsize=20)
        axs[0, 1].set_xlabel("Time $t$", fontsize=18)
        axs[0, 1].set_ylabel("$|q(t) - \\bar{{q}}(t)| / |\\bar{{q}}(t)|$", fontsize=18)
        axs[0, 1].grid(True, linestyle='--')
        axs[0, 1].legend(loc='upper left', fontsize=16)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=14)

    for i in range(Y_vals.shape[1]):
        if true_Y is not None:
            axs[1, 0].plot(timesteps, true_Y[:, i, 0], linestyle="--", color=analytical_colors(i), alpha=0.7, linewidth=1.5, label=f"Analytical $\\bar{{Y}}(t)$" if i == 0 else None)
        axs[1, 0].plot(timesteps, Y_vals[:, i, 0], color=colors(i), linewidth=2, label=f"Learned $Y^\\theta(t)$" if i == 0 else None)
    axs[1, 0].set_title("Cost-to-Go $Y(t)$", fontsize=20)
    axs[1, 0].set_xlabel("Time $t$", fontsize=18)
    axs[1, 0].set_ylabel("Y(t)", fontsize=18)
    axs[1, 0].grid(True, linestyle='--')
    axs[1, 0].legend(loc='upper left', fontsize=16)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=14)

    if validation is not None and true_Y is not None:
        for i in range(Y_vals.shape[1]):
            # Calculate relative error: |Y - Y_true| / |Y_true|, with small epsilon to avoid division by zero
            eps = 1e-8
            rel_error_Y = np.abs(Y_vals[:, i, 0] - true_Y[:, i, 0]) / (np.abs(true_Y[:, i, 0]) + eps)
            axs[1, 1].plot(timesteps, rel_error_Y, color=colors(i), label=f"Rel. Error $Y(t)$ ({val_Y_loss[-1]:.2f})" if i == 0 else None)
        axs[1, 1].axhline(0, color='red', linestyle='--', linewidth=0.8)
        axs[1, 1].set_title("Relative Error in Cost-to-Go $Y(t)$", fontsize=20)
        axs[1, 1].set_xlabel("Time $t$", fontsize=18)
        axs[1, 1].set_ylabel("$|Y(t) - \\bar{{Y}}(t)| / |\\bar{{Y}}(t)|$", fontsize=18)
        axs[1, 1].grid(True, linestyle='--')
        axs[1, 1].legend(loc='upper left', fontsize=16)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=14)

    for i in range(y_vals.shape[1]):
        if true_y is not None:
            axs[2, 0].plot(timesteps, true_y[:, i, 0], linestyle="--", color=analytical_colors(i), alpha=0.7, linewidth=1.5, label=f"$\\bar{{X}}(t)$" if i == 0 else None)
        axs[2, 0].plot(timesteps, y_vals[:, i, 0], color=colors(i), linewidth=2, label=f"$X(t)$" if i == 0 else None)
        if y_vals.shape[2] > 1:
            axs[2, 0].plot(timesteps, y_vals[:, i, 2], linestyle="-.", color=colors(i), linewidth=2, label=f"$G(t)$" if i == 0 else None)
    axs[2, 0].set_title("States: $X(t)$ and $G(t)$", fontsize=20)
    axs[2, 0].set_xlabel("Time $t$", fontsize=18)
    axs[2, 0].set_ylabel("X(t), G(t)", fontsize=18)
    axs[2, 0].grid(True, linestyle='--')
    axs[2, 0].legend(loc='upper left', fontsize=16)
    axs[2, 0].tick_params(axis='both', which='major', labelsize=14)

    if y_vals.shape[2] > 1:
        for i in range(y_vals.shape[1]):
            axs[2, 1].plot(timesteps, y_vals[:, i, 1], color=colors(i), linewidth=2, label=f"$P(t)$" if i == 0 else None)
            if true_y is not None:
                axs[2, 1].plot(timesteps, true_y[:, i, 1], linestyle="--", color=analytical_colors(i), alpha=0.7, linewidth=1.5, label=f"$\\bar{{P}}(t)$" if i == 0 else None)
        axs[2, 1].set_title("State: $P(t)$", fontsize=20)
        axs[2, 1].set_xlabel("Time $t$", fontsize=18)
        axs[2, 1].set_ylabel("P(t)", fontsize=18)
        axs[2, 1].grid(True, linestyle='--')
        axs[2, 1].legend(loc='upper left', fontsize=16)
        axs[2, 1].tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    if save_dir:
        if num:
            plt.savefig(f"{save_dir}/approx_vs_analytic_{num}.png", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{save_dir}/approx_vs_analytic.png", dpi=300, bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()

def plot_approx_vs_analytic_expectation(results, timesteps, plot=True, save_dir=None, num=None):
    approx_q = results["q_learned"]
    Y_vals = results["Y_learned"]
    true_q = results["q_analytical"]
    true_Y = results["Y_analytical"]

    # Learned results
    mean_q = approx_q.mean(axis=1).squeeze()
    std_q = approx_q.std(axis=1).squeeze()
    mean_Y = Y_vals[:, :, 0].mean(axis=1).squeeze()
    std_Y = Y_vals[:, :, 0].std(axis=1).squeeze()

    # Analytic results
    mean_q_analytical = true_q.mean(axis=1).squeeze()
    std_q_analytical = true_q.std(axis=1).squeeze()
    mean_true_Y = true_Y.mean(axis=1).squeeze()
    std_true_Y = true_Y.std(axis=1).squeeze()

    # Calculate relative errors
    diff = np.abs(approx_q - true_q) / (np.abs(true_q) + 1e-8)  # Relative error with epsilon
    mean_diff = np.mean(diff, axis=1).squeeze()
    std_diff = np.std(diff, axis=1).squeeze()
    
    diff_Y = np.abs(Y_vals - true_Y) / (np.abs(true_Y) + 1e-8)  # Relative error with epsilon
    mean_diff_Y = np.mean(diff_Y, axis=1).squeeze()
    std_diff_Y = np.std(diff_Y, axis=1).squeeze()

    # # Create individual wider plots
    # if save_dir:
    #     # Plot 1: Control q(t) - Learned vs Analytical
    #     fig1, ax1 = plt.subplots(1, 1, figsize=(16, 6))
    #     ax1.plot(timesteps[:-1], mean_q, label='Learned $q^\\theta$ Mean', color='blue', linewidth=2)
    #     ax1.fill_between(timesteps[:-1], mean_q - std_q, mean_q + std_q, color='blue', alpha=0.3, label='Learned $q^\\theta$ ±1 Std')
    #     ax1.plot(timesteps[:-1], mean_q_analytical, label='Analytical Mean', color='black', linestyle='--', linewidth=2)
    #     ax1.fill_between(timesteps[:-1], mean_q_analytical - std_q_analytical, mean_q_analytical + std_q_analytical, color='black', alpha=0.2, label='Analytical ±1 Std')
    #     ax1.set_title("Control $q(t)$: Learned vs Analytical", fontsize=18)
    #     ax1.set_xlabel("Time $t$", fontsize=16)
    #     ax1.set_ylabel("$q(t)$", fontsize=16)
    #     ax1.grid(True, linestyle='--')
    #     ax1.legend(loc='upper left', fontsize=14)
    #     ax1.tick_params(axis='both', which='major', labelsize=12)
    #     plt.tight_layout()
    #     suffix = f"_{num}" if num else ""
    #     plt.savefig(f"{save_dir}/control_q_expectation{suffix}.png", dpi=300, bbox_inches='tight')
    #     plt.close()

    #     # Plot 2: Relative Error in Control q(t)
    #     fig2, ax2 = plt.subplots(1, 1, figsize=(16, 6))
    #     ax2.fill_between(timesteps[:-1], mean_diff - std_diff, mean_diff + std_diff, color='red', alpha=0.4, label='±1 Std Dev')
    #     ax2.plot(timesteps[:-1], mean_diff, color='red', label='Mean Rel. Error', linewidth=2)
    #     ax2.set_title("Relative Error in Control $q(t)$", fontsize=18)
    #     ax2.set_xlabel("Time $t$", fontsize=16)
    #     ax2.set_ylabel("$|q(t) - \\bar{{q}}(t)| / |\\bar{{q}}(t)|$", fontsize=16)
    #     ax2.grid(True, linestyle='--')
    #     ax2.legend(loc='upper left', fontsize=14)
    #     ax2.tick_params(axis='both', which='major', labelsize=12)
    #     plt.tight_layout()
    #     plt.savefig(f"{save_dir}/control_q_error{suffix}.png", dpi=300, bbox_inches='tight')
    #     plt.close()

    #     # Plot 3: Cost-to-Go Y(t)
    #     fig3, ax3 = plt.subplots(1, 1, figsize=(16, 6))
    #     ax3.plot(timesteps, mean_Y, color='blue', label='Learned $Y^\\theta$ Mean', linewidth=2)
    #     ax3.fill_between(timesteps, mean_Y - std_Y, mean_Y + std_Y, color='blue', alpha=0.3, label='Learned $Y^\\theta$ ±1 Std')
    #     ax3.plot(timesteps, mean_true_Y, color='black', linestyle='--', label='Analytical Mean', linewidth=2)
    #     ax3.fill_between(timesteps, mean_true_Y - std_true_Y, mean_true_Y + std_true_Y, color='black', alpha=0.2, label='Analytical ±1 Std')
    #     ax3.set_title("Cost-to-Go $Y(t)$", fontsize=18)
    #     ax3.set_xlabel("Time $t$", fontsize=16)
    #     ax3.set_ylabel("Y(t)", fontsize=16)
    #     ax3.grid(True, linestyle='--')
    #     ax3.legend(loc='upper left', fontsize=14)
    #     ax3.tick_params(axis='both', which='major', labelsize=12)
    #     plt.tight_layout()
    #     plt.savefig(f"{save_dir}/cost_to_go_Y_expectation{suffix}.png", dpi=300, bbox_inches='tight')
    #     plt.close()

    #     # Plot 4: Relative Error in Cost-to-Go Y(t)
    #     fig4, ax4 = plt.subplots(1, 1, figsize=(16, 6))
    #     ax4.fill_between(timesteps, mean_diff_Y - std_diff_Y, mean_diff_Y + std_diff_Y, color='red', alpha=0.4, label='±1 Std Dev')
    #     ax4.plot(timesteps, mean_diff_Y, color='red', label='Mean Rel. Error', linewidth=2)
    #     ax4.set_title("Relative Error in Cost-to-Go $Y(t)$", fontsize=18)
    #     ax4.set_xlabel("Time $t$", fontsize=16)
    #     ax4.set_ylabel("$|Y(t) - \\bar{{Y}}(t)| / |\\bar{{Y}}(t)|$", fontsize=16)
    #     ax4.grid(True, linestyle='--')
    #     ax4.legend(loc='upper left', fontsize=14)
    #     ax4.tick_params(axis='both', which='major', labelsize=12)
    #     plt.tight_layout()
    #     plt.savefig(f"{save_dir}/cost_to_go_Y_error{suffix}.png", dpi=300, bbox_inches='tight')
    #     plt.close()

    # Create combined plot
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    axs[0, 0].plot(timesteps[:-1], mean_q, label='Learned $q^\\theta$ Mean', color='blue', linewidth=2)
    axs[0, 0].fill_between(timesteps[:-1], mean_q - std_q, mean_q + std_q, color='blue', alpha=0.3, label='Learned $q^\\theta$ ±1 Std')
    axs[0, 0].plot(timesteps[:-1], mean_q_analytical, label='Analytical Mean', color='black', linestyle='--', linewidth=2)
    axs[0, 0].fill_between(timesteps[:-1], mean_q_analytical - std_q_analytical, mean_q_analytical + std_q_analytical, color='black', alpha=0.2, label='Analytical ±1 Std')
    axs[0, 0].set_title("Control $q(t)$: Learned vs Analytical", fontsize=16)
    axs[0, 0].set_xlabel("Time $t$", fontsize=14)
    axs[0, 0].set_ylabel("$q(t)$", fontsize=14)
    axs[0, 0].grid(True, linestyle='--')
    axs[0, 0].legend(loc='upper left', fontsize=12)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=10)

    axs[0, 1].fill_between(timesteps[:-1], mean_diff - std_diff, mean_diff + std_diff, color='red', alpha=0.4, label='±1 Std Dev')
    axs[0, 1].plot(timesteps[:-1], mean_diff, color='red', label='Mean Rel. Error', linewidth=2)
    axs[0, 1].set_title("Relative Error in Control $q(t)$", fontsize=16)
    axs[0, 1].set_xlabel("Time $t$", fontsize=14)
    axs[0, 1].set_ylabel("$|q(t) - \\bar{{q}}(t)| / |\\bar{{q}}(t)|$", fontsize=14)
    axs[0, 1].grid(True, linestyle='--')
    axs[0, 1].legend(loc='upper left', fontsize=12)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=10)

    axs[1, 0].plot(timesteps, mean_Y, color='blue', label='Learned $Y^\\theta$ Mean', linewidth=2)
    axs[1, 0].fill_between(timesteps, mean_Y - std_Y, mean_Y + std_Y, color='blue', alpha=0.3, label='Learned $Y^\\theta$ ±1 Std')
    axs[1, 0].plot(timesteps, mean_true_Y, color='black', linestyle='--', label='Analytical Mean', linewidth=2)
    axs[1, 0].fill_between(timesteps, mean_true_Y - std_true_Y, mean_true_Y + std_true_Y, color='black', alpha=0.2, label='Analytical ±1 Std')
    axs[1, 0].set_title("Cost-to-Go $Y(t)$", fontsize=16)
    axs[1, 0].set_xlabel("Time $t$", fontsize=14)
    axs[1, 0].set_ylabel("Y(t)", fontsize=14)
    axs[1, 0].grid(True, linestyle='--')
    axs[1, 0].legend(loc='upper left', fontsize=12)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=10)

    axs[1, 1].fill_between(timesteps, mean_diff_Y - std_diff_Y, mean_diff_Y + std_diff_Y, color='red', alpha=0.4, label='±1 Std Dev')
    axs[1, 1].plot(timesteps, mean_diff_Y, color='red', label='Mean Rel. Error', linewidth=2)
    axs[1, 1].set_title("Relative Error in Cost-to-Go $Y(t)$", fontsize=16)
    axs[1, 1].set_xlabel("Time $t$", fontsize=14)
    axs[1, 1].set_ylabel("$|Y(t) - \\bar{{Y}}(t)| / |\\bar{{Y}}(t)|$", fontsize=14)
    axs[1, 1].grid(True, linestyle='--')
    axs[1, 1].legend(loc='upper left', fontsize=12)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    if save_dir:
        if num:
            plt.savefig(f"{save_dir}/approx_vs_analytic_expectation_{num}.png", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{save_dir}/approx_vs_analytic_expectation.png", dpi=300, bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()
    
def plot_terminal_histogram(results, dynamics, plot=True, save_dir=None, num=None):
    y_vals = results["y_learned"]
    q_vals = results["q_learned"]
    Y_vals = results["Y_learned"]

    y_T = y_vals[-1, :, :]
    Y_T_approx = Y_vals[-1, :, 0]
    q_T_approx = q_vals[-1, :, 0]
    y_T_tensor = torch.tensor(y_T, dtype=torch.float32, device=dynamics.device)
    Y_T_true = dynamics.terminal_cost(y_T_tensor).detach().cpu().numpy()
    if dynamics.analytical_known:
        q_T_true = dynamics.optimal_control_analytic(dynamics.T - dynamics.dt, y_T_tensor).detach().cpu().numpy()

    # Ensure arrays are at least 1D for consistent indexing
    Y_T_true = np.atleast_1d(Y_T_true.squeeze())
    if dynamics.analytical_known:
        q_T_true = np.atleast_1d(q_T_true.squeeze())

    # Filter out NaN or Inf
    mask = np.isfinite(Y_T_approx) & np.isfinite(Y_T_true)
    Y_T_approx = Y_T_approx[mask]
    Y_T_true = Y_T_true[mask]

    if dynamics.analytical_known:
        mask_q = np.isfinite(q_T_approx) & np.isfinite(q_T_true)
        q_T_approx = q_T_approx[mask_q]
        q_T_true = q_T_true[mask_q]

    if len(Y_T_approx) == 0 or len(Y_T_true) == 0:
        print("Warning: No valid terminal values to plot.")
        return

    range_approx = np.ptp(Y_T_approx)  # Peak-to-peak (max - min)
    range_true = np.ptp(Y_T_true)
    range_combined = max(range_approx, range_true)

    if range_combined == 0:
        print("Warning: No variation in terminal values. Skipping histogram.")
        return

    # Choose bins depending on data spread
    bins = min(30, max(1, int(range_combined / 1e-2)))

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.hist(Y_T_approx, bins=bins, alpha=0.6, label="Approx. $Y^\\theta_T$", color="blue", density=True)
    plt.hist(Y_T_true, bins=bins, alpha=0.6, label="Analytical $g(y_T)$", color="green", density=True)
    plt.axvline(np.mean(Y_T_approx), color='blue', linestyle='--', label=f"Mean approx: {np.mean(Y_T_approx):.3f}")
    plt.axvline(np.mean(Y_T_true), color='green', linestyle='--', label=f"Mean true: {np.mean(Y_T_true):.3f}")
    plt.title("Distribution of Terminal Values")
    plt.xlabel("$Y(T)$ / $g(y_T)$")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    plt.hist(q_T_approx, bins=bins, alpha=0.6, label="Approx. $q^\\theta_T$", color="blue", density=True)
    plt.axvline(np.mean(q_T_approx), color='blue', linestyle='--', label=f"Mean approx: {np.mean(q_T_approx):.3f}")
    if dynamics.analytical_known:
        plt.hist(q_T_true, bins=bins, alpha=0.6, label="Analytical $\\bar{{q}}(y_T)$", color="green", density=True)
        plt.axvline(np.mean(q_T_true), color='green', linestyle='--', label=f"Mean true: {np.mean(q_T_true):.3f}")
    plt.title("Distribution of Terminal Controls")
    plt.xlabel("$q(T)$ / $\\bar{{q}}(y_T)$")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()

    if save_dir:
        if num:
            plt.savefig(f"{save_dir}/terminal_histogram_{num}.png", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{save_dir}/terminal_histogram.png", dpi=300, bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()
