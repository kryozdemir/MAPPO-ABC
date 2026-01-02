"""
Evaluation Script for MAPPO-ABC

Evaluates a trained MAPPO-ABC model on SMAC environments.
"""

import os
import argparse
import numpy as np
import torch
from smac.env import StarCraft2Env

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.algorithms.mappo_abc import MAPPOABC
from src.utils.utils import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate MAPPO-ABC on SMAC')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--map', type=str, default='3m',
                      help='SMAC map name')
    parser.add_argument('--episodes', type=int, default=100,
                      help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--render', action='store_true',
                      help='Render episodes')
    parser.add_argument('--deterministic', action='store_true',
                      help='Use deterministic policy')
    
    return parser.parse_args()


def evaluate(env, agent, n_episodes, deterministic=True, render=False):
    """
    Evaluate agent on environment.
    
    Args:
        env: SMAC environment
        agent: MAPPO-ABC agent
        n_episodes: Number of episodes
        deterministic: Use deterministic actions
        render: Render episodes
        
    Returns:
        Dictionary with evaluation statistics
    """
    episode_returns = []
    episode_steps = []
    wins = 0
    
    for ep in range(n_episodes):
        env.reset()
        episode_return = 0
        steps = 0
        done = False
        
        while not done:
            # Get observations
            observations = env.get_obs()
            
            # Select actions
            actions, _, head_used = agent.select_actions(
                np.array(observations),
                deterministic=deterministic
            )
            
            # Step environment
            reward, done, info = env.step(actions.tolist())
            
            episode_return += reward
            steps += 1
            
            if render:
                env.render()
        
        # Store episode stats
        episode_returns.append(episode_return)
        episode_steps.append(steps)
        
        if info.get('battle_won', False):
            wins += 1
        
        # Print progress
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{n_episodes} | "
                  f"Return: {episode_return:.2f} | "
                  f"Steps: {steps} | "
                  f"Win: {info.get('battle_won', False)}")
    
    # Compute statistics
    stats = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'min_return': np.min(episode_returns),
        'max_return': np.max(episode_returns),
        'mean_steps': np.mean(episode_steps),
        'win_rate': wins / n_episodes,
        'total_wins': wins
    }
    
    return stats


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create environment
    env = StarCraft2Env(map_name=args.map)
    env_info = env.get_env_info()
    
    n_agents = env_info['n_agents']
    obs_dim = env_info['obs_shape']
    n_actions = env_info['n_actions']
    
    print(f"Evaluating on: {args.map}")
    print(f"Agents: {n_agents}, Obs dim: {obs_dim}, Actions: {n_actions}")
    
    # Create agent (with default config)
    agent = MAPPOABC(
        n_agents=n_agents,
        obs_dim=obs_dim,
        act_dim=n_actions,
        device=args.device
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    agent.load(args.checkpoint)
    
    # Evaluate
    print(f"\nRunning {args.episodes} evaluation episodes...")
    stats = evaluate(
        env,
        agent,
        n_episodes=args.episodes,
        deterministic=args.deterministic,
        render=args.render
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Mean Return: {stats['mean_return']:.3f} ± {stats['std_return']:.3f}")
    print(f"Min/Max Return: {stats['min_return']:.3f} / {stats['max_return']:.3f}")
    print(f"Mean Steps: {stats['mean_steps']:.1f}")
    print(f"Win Rate: {stats['win_rate']:.2%} ({stats['total_wins']}/{args.episodes})")
    print("="*50)
    
    # Close environment
    env.close()


if __name__ == '__main__':
    main()
