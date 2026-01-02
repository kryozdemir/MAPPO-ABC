"""
Visualize and Compare Training Results

This script loads training logs and creates comparison plots.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize training results')
    
    parser.add_argument('--log-dirs', nargs='+', required=True,
                      help='Directories containing training logs')
    parser.add_argument('--labels', nargs='+', required=True,
                      help='Labels for each log directory')
    parser.add_argument('--output-dir', type=str, default='plots',
                      help='Directory to save plots')
    parser.add_argument('--smooth', type=int, default=10,
                      help='Smoothing window size')
    
    return parser.parse_args()


def smooth_data(data, window=10):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values


def plot_training_curves(log_dirs, labels, output_dir, smooth_window=10):
    """Plot training curves comparing different algorithms."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Metrics to plot
    metrics = ['return', 'battle_won', 'policy_loss', 'value_loss', 'entropy']
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for log_dir, label in zip(log_dirs, labels):
            csv_path = os.path.join(log_dir, 'metrics.csv')
            
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found, skipping...")
                continue
            
            # Load data
            df = pd.read_csv(csv_path)
            
            if metric not in df.columns:
                continue
            
            episodes = df['episode'].values
            values = df[metric].values
            
            # Smooth
            values_smooth = smooth_data(values, window=smooth_window)
            
            # Plot
            ax.plot(episodes, values_smooth, label=label, linewidth=2)
            
            # Add confidence interval (std)
            if len(values) > smooth_window:
                std = pd.Series(values).rolling(window=smooth_window, min_periods=1).std().values
                ax.fill_between(
                    episodes,
                    values_smooth - std,
                    values_smooth + std,
                    alpha=0.2
                )
        
        # Formatting
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save
        save_path = os.path.join(output_dir, f'{metric}_comparison.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_path}")


def plot_head_selection(log_dir, output_dir):
    """Plot policy head selection frequency for MAPPO-ABC."""
    
    csv_path = os.path.join(log_dir, 'metrics.csv')
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    
    # Check if head_chosen column exists
    if 'head_chosen' not in df.columns:
        print("No head selection data found (not MAPPO-ABC?)")
        return
    
    # Count head selections
    head_counts = df['head_chosen'].value_counts().sort_index()
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot of selection frequency
    head_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Policy Head', fontsize=12)
    ax1.set_ylabel('Selection Count', fontsize=12)
    ax1.set_title('Policy Head Selection Frequency', fontsize=14, fontweight='bold')
    ax1.set_xticklabels([f'H{i}' for i in head_counts.index], rotation=0)
    
    # Over time
    window = 100
    for head_idx in range(4):
        head_mask = (df['head_chosen'] == head_idx).astype(int)
        head_freq = pd.Series(head_mask).rolling(window=window, min_periods=1).mean()
        ax2.plot(df['episode'], head_freq, label=f'H{head_idx}', linewidth=2)
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Selection Probability', fontsize=12)
    ax2.set_title(f'Head Selection Over Time (window={window})', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'head_selection.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def plot_fitness_evolution(log_dir, output_dir):
    """Plot ABC fitness evolution for each head."""
    
    csv_path = os.path.join(log_dir, 'metrics.csv')
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    
    # Check if fitness columns exist
    fitness_cols = [f'fit_H{i}' for i in range(4)]
    if not all(col in df.columns for col in fitness_cols):
        print("No fitness data found")
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in range(4):
        ax.plot(df['episode'], df[f'fit_H{i}'], label=f'Head {i}', linewidth=2)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    ax.set_title('ABC Fitness Evolution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fitness_evolution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def main():
    args = parse_args()
    
    if len(args.log_dirs) != len(args.labels):
        print("Error: Number of log directories must match number of labels")
        return
    
    print("Creating comparison plots...")
    plot_training_curves(
        args.log_dirs,
        args.labels,
        args.output_dir,
        smooth_window=args.smooth
    )
    
    # If first log is MAPPO-ABC, plot additional metrics
    print("\nCreating MAPPO-ABC specific plots...")
    plot_head_selection(args.log_dirs[0], args.output_dir)
    plot_fitness_evolution(args.log_dirs[0], args.output_dir)
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

