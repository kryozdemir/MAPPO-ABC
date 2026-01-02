"""
Training Script for Baseline Algorithms (IPPO, MAPPO, QMIX, VDN)

This script trains baseline MARL algorithms on SMAC environments.
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

from src.algorithms.ippo import IPPO
from src.algorithms.mappo import BaselineMAPPO
from src.algorithms.qmix import QMIX
from src.algorithms.vdn import VDN
from src.utils.logger import Logger
from src.utils.utils import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train baseline MARL algorithms')
    
    parser.add_argument('--algorithm', type=str, required=True,
                      choices=['ippo', 'mappo', 'qmix', 'vdn'],
                      help='Algorithm to train')
    parser.add_argument('--map', type=str, default='3m',
                      help='SMAC map name')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--episodes', type=int, default=5000,
                      help='Number of training episodes')
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
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    
    return parser.parse_args()


def create_agent(algorithm, env_info, device):
    """Create agent based on algorithm type."""
    n_agents = env_info['n_agents']
    obs_dim = env_info['obs_shape']
    n_actions = env_info['n_actions']
    state_dim = env_info['state_shape']
    
    if algorithm == 'ippo':
        return IPPO(
            n_agents=n_agents,
            obs_dim=obs_dim,
            act_dim=n_actions,
            device=device
        )
    elif algorithm == 'mappo':
        return BaselineMAPPO(
            n_agents=n_agents,
            obs_dim=obs_dim,
            act_dim=n_actions,
            device=device
        )
    elif algorithm == 'qmix':
        return QMIX(
            n_agents=n_agents,
            obs_dim=obs_dim,
            act_dim=n_actions,
            state_dim=state_dim,
            device=device
        )
    elif algorithm == 'vdn':
        return VDN(
            n_agents=n_agents,
            obs_dim=obs_dim,
            act_dim=n_actions,
            device=device
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def evaluate(env, agent, algorithm, n_episodes=10):
    """Evaluate the agent."""
    episode_returns = []
    wins = 0
    
    for _ in range(n_episodes):
        env.reset()
        episode_return = 0
        done = False
        
        while not done:
            observations = env.get_obs()
            
            if algorithm in ['ippo', 'mappo']:
                actions, _ = agent.select_actions(
                    np.array(observations),
                    deterministic=True
                )
            else:  # qmix, vdn
                actions = agent.select_actions(
                    np.array(observations),
                    deterministic=True
                )
            
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
    # Set seed
    set_seed(args.seed)
    
    # Create environment
    env = StarCraft2Env(map_name=args.map)
    env_info = env.get_env_info()
    
    n_agents = env_info['n_agents']
    obs_dim = env_info['obs_shape']
    n_actions = env_info['n_actions']
    
    print(f"Training {args.algorithm.upper()} on {args.map}")
    print(f"Agents: {n_agents}, Obs dim: {obs_dim}, Actions: {n_actions}")
    
    # Create agent
    agent = create_agent(args.algorithm, env_info, args.device)
    
    # Create logger
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f"{args.algorithm}_{args.map}_{timestamp}")
    logger = Logger(log_dir)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Starting training for {args.episodes} episodes...")
    
    # Training loop
    for episode in range(1, args.episodes + 1):
        env.reset()
        episode_return = 0
        episode_steps = 0
        done = False
        
        while not done:
            observations = env.get_obs()
            state = env.get_state() if args.algorithm in ['qmix'] else None
            
            # Select actions
            if args.algorithm in ['ippo', 'mappo']:
                actions, log_probs = agent.select_actions(np.array(observations))
                values = agent.compute_values(np.array(observations))
            else:  # qmix, vdn
                actions = agent.select_actions(np.array(observations))
            
            # Step environment
            reward, done, info = env.step(actions.tolist())
            next_observations = env.get_obs()
            next_state = env.get_state() if args.algorithm in ['qmix'] else None
            
            # Store transition
            if args.algorithm in ['ippo', 'mappo']:
                agent.store_transitions(
                    observations=np.array(observations),
                    actions=actions,
                    rewards=np.array([reward] * n_agents),
                    dones=np.array([done] * n_agents),
                    values=values,
                    log_probs=log_probs
                )
            elif args.algorithm == 'qmix':
                agent.buffer.push(
                    observations=np.array(observations),
                    actions=actions,
                    rewards=reward,
                    next_observations=np.array(next_observations),
                    done=done,
                    state=state,
                    next_state=next_state
                )
            else:  # vdn
                agent.buffer.push(
                    observations=np.array(observations),
                    actions=actions,
                    rewards=reward,
                    next_observations=np.array(next_observations),
                    done=done
                )
            
            episode_return += reward
            episode_steps += 1
        
        # Update agent
        if args.algorithm in ['ippo', 'mappo']:
            metrics = agent.update(batch_size=args.batch_size, n_epochs=4)
        else:  # qmix, vdn
            metrics = agent.update(batch_size=args.batch_size)
        
        # Log metrics
        log_data = {
            'episode': episode,
            'return': episode_return,
            'steps': episode_steps,
            'win': int(info.get('battle_won', False))
        }
        
        if args.algorithm in ['ippo', 'mappo']:
            log_data.update({
                'policy_loss': metrics.get('policy_loss', 0),
                'value_loss': metrics.get('value_loss', 0),
                'entropy': metrics.get('entropy', 0)
            })
        else:
            log_data.update({
                'loss': metrics.get('loss', 0),
                'epsilon': metrics.get('epsilon', 0)
            })
        
        logger.log(log_data)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}/{args.episodes} | "
                  f"Return: {episode_return:.2f} | "
                  f"Steps: {episode_steps} | "
                  f"Win: {info.get('battle_won', False)}")
        
        # Evaluate
        if episode % args.eval_interval == 0:
            avg_return, win_rate = evaluate(env, agent, args.algorithm, n_episodes=10)
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
                f"{args.algorithm}_{args.map}_ep{episode}.pth"
            )
            agent.save(save_path)
    
    # Final save
    final_path = os.path.join(args.save_dir, f"{args.algorithm}_{args.map}_final.pth")
    agent.save(final_path)
    print(f"Training complete! Final model saved to {final_path}")
    
    # Close environment
    env.close()


if __name__ == '__main__':
    args = parse_args()
    train(args)

