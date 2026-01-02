"""
Training Script for MAPPO-ABC on SMAC

This script trains the MAPPO-ABC algorithm on StarCraft II multi-agent scenarios.
"""

import os
import argparse
import yaml
import numpy as np
import torch
from datetime import datetime
from smac.env import StarCraft2Env

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.algorithms.mappo_abc import MAPPOABC
from src.utils.logger import Logger
from src.utils.utils import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MAPPO-ABC on SMAC')
    
    parser.add_argument('--map', type=str, default='3m',
                      help='SMAC map name')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--episodes', type=int, default=5000,
                      help='Number of training episodes')
    parser.add_argument('--config', type=str, default='configs/mappo_abc_default.yaml',
                      help='Path to config file')
    parser.add_argument('--save-dir', type=str, default='models',
                      help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='results',
                      help='Directory for logs')
    parser.add_argument('--save-interval', type=int, default=100,
                      help='Save model every N episodes')
    parser.add_argument('--eval-interval', type=int, default=50,
                      help='Evaluate every N episodes')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Return default config if file doesn't exist
        return {
            'hidden_dim': 256,
            'n_heads': 4,
            'lr': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_param': 0.2,
            'value_loss_coef': 1.0,
            'entropy_coef': 0.01,
            'max_grad_norm': 10.0,
            'batch_size': 32,
            'n_epochs': 4,
            'abc': {
                'food_sources': 20,
                'iterations': 50,
                'limit': 10
            }
        }


def evaluate(env, agent, n_episodes=10):
    """
    Evaluate the agent.
    
    Args:
        env: SMAC environment
        agent: MAPPO-ABC agent
        n_episodes: Number of evaluation episodes
        
    Returns:
        Average episode return and win rate
    """
    episode_returns = []
    wins = 0
    
    for _ in range(n_episodes):
        env.reset()
        episode_return = 0
        done = False
        
        while not done:
            # Get observations
            observations = env.get_obs()
            state = env.get_state()
            
            # Select actions (deterministic for evaluation)
            actions, _, _ = agent.select_actions(
                np.array(observations),
                deterministic=True
            )
            
            # Step environment
            reward, done, info = env.step(actions.tolist())
            episode_return += reward
        
        episode_returns.append(episode_return)
        if info.get('battle_won', False):
            wins += 1
    
    avg_return = np.mean(episode_returns)
    win_rate = wins / n_episodes
    
    return avg_return, win_rate


def train(args):
    """Main training loop."""
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Create environment
    env = StarCraft2Env(map_name=args.map)
    env_info = env.get_env_info()
    
    n_agents = env_info['n_agents']
    obs_dim = env_info['obs_shape']
    n_actions = env_info['n_actions']
    
    print(f"Environment: {args.map}")
    print(f"Agents: {n_agents}, Obs dim: {obs_dim}, Actions: {n_actions}")
    
    # Create agent
    agent = MAPPOABC(
        n_agents=n_agents,
        obs_dim=obs_dim,
        act_dim=n_actions,
        hidden_dim=config['hidden_dim'],
        n_heads=config['n_heads'],
        lr=config['lr'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_param=config['clip_param'],
        value_loss_coef=config['value_loss_coef'],
        entropy_coef=config['entropy_coef'],
        max_grad_norm=config['max_grad_norm'],
        abc_config=config['abc'],
        device=args.device
    )
    
    # Create logger
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f"{args.map}_{timestamp}")
    logger = Logger(log_dir)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Starting training for {args.episodes} episodes...")
    print(f"Logs will be saved to: {log_dir}")
    
    # Training loop
    for episode in range(1, args.episodes + 1):
        env.reset()
        episode_return = 0
        episode_steps = 0
        done = False
        
        # Select head for this episode using ABC
        head_idx = agent.abc_optimizer.select_best_head()
        
        while not done:
            # Get observations and state
            observations = env.get_obs()
            state = env.get_state()
            
            # Select actions
            actions, log_probs, _ = agent.select_actions(
                np.array(observations),
                head_idx=head_idx
            )
            
            # Compute values
            centralized_state = np.concatenate(observations)
            values = agent.compute_values(centralized_state.reshape(1, -1))
            
            # Step environment
            reward, done, info = env.step(actions.tolist())
            
            # Store transition
            agent.store_transition(
                observations=np.array(observations),
                actions=actions,
                rewards=np.array([reward] * n_agents),
                dones=np.array([done] * n_agents),
                values=values[0],
                log_probs=log_probs,
                head_used=head_idx
            )
            
            episode_return += reward
            episode_steps += 1
        
        # Update ABC fitness
        agent.update_abc_fitness(head_idx, episode_return)
        
        # Compute advantages and update policy
        agent.buffer.compute_advantages(
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda']
        )
        metrics = agent.update(
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs']
        )
        
        # Get ABC statistics
        abc_stats = agent.abc_optimizer.get_statistics()
        
        # Log metrics
        logger.log({
            'episode': episode,
            'return': episode_return,
            'steps': episode_steps,
            'battle_won': int(info.get('battle_won', False)),
            'policy_loss': metrics['policy_loss'],
            'value_loss': metrics['value_loss'],
            'entropy': metrics['entropy'],
            'head_chosen': head_idx,
            **{f'fit_H{i}': abc_stats['fitness'][i] for i in range(4)},
            **{f'p_H{i}': abc_stats['probabilities'][i] for i in range(4)}
        })
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}/{args.episodes} | "
                  f"Return: {episode_return:.2f} | "
                  f"Steps: {episode_steps} | "
                  f"Head: H{head_idx} | "
                  f"Win: {info.get('battle_won', False)}")
        
        # Evaluate
        if episode % args.eval_interval == 0:
            avg_return, win_rate = evaluate(env, agent, n_episodes=10)
            logger.log({
                'episode': episode,
                'eval_return': avg_return,
                'eval_win_rate': win_rate
            })
            print(f"Evaluation | Avg Return: {avg_return:.2f} | Win Rate: {win_rate:.2%}")
        
        # Save model
        if episode % args.save_interval == 0:
            save_path = os.path.join(
                args.save_dir,
                f"mappo_abc_{args.map}_ep{episode}.pth"
            )
            agent.save(save_path)
            print(f"Model saved to {save_path}")
    
    # Final save
    final_path = os.path.join(args.save_dir, f"mappo_abc_{args.map}_final.pth")
    agent.save(final_path)
    print(f"Training complete! Final model saved to {final_path}")
    
    # Close environment
    env.close()


if __name__ == '__main__':
    args = parse_args()
    train(args)
