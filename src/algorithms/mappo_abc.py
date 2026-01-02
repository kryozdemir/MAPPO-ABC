"""
MAPPO-ABC: Multi-Agent Proximal Policy Optimization with Artificial Bee Colony

This module implements the core MAPPO-ABC algorithm, combining multi-agent PPO
with ABC-based policy head selection for improved exploration-exploitation balance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional

from src.networks.policy_network import MultiHeadPolicyNetwork
from src.networks.value_network import ValueNetwork
from src.algorithms.abc_optimizer import ABCOptimizer
from src.algorithms.buffer import RolloutBuffer


class MAPPOABC:
    """
    Multi-Agent Proximal Policy Optimization with Artificial Bee Colony optimization.
    
    This algorithm uses 4 specialized policy heads with different exploration strategies,
    and ABC optimization to dynamically select the best-performing head during training.
    
    Args:
        n_agents: Number of agents in the environment
        obs_dim: Observation dimension per agent
        act_dim: Action dimension per agent
        hidden_dim: Hidden layer dimension for networks
        n_heads: Number of policy heads (default: 4)
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_param: PPO clipping parameter
        value_loss_coef: Value loss coefficient
        entropy_coef: Entropy coefficient
        max_grad_norm: Maximum gradient norm for clipping
        abc_config: Configuration dict for ABC optimizer
        device: Device to run on (cuda/cpu)
    """
    
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        n_heads: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_param: float = 0.2,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 10.0,
        abc_config: Optional[Dict] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Initialize multi-head policy network
        self.policy = MultiHeadPolicyNetwork(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads
        ).to(device)
        
        # Initialize centralized critic
        self.critic = ValueNetwork(
            state_dim=obs_dim * n_agents,  # Centralized state
            hidden_dim=hidden_dim
        ).to(device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # ABC optimizer for head selection
        if abc_config is None:
            abc_config = {
                'food_sources': 20,
                'iterations': 50,
                'limit': 10
            }
        self.abc_optimizer = ABCOptimizer(n_heads=n_heads, **abc_config)
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Training statistics
        self.training_step = 0
        
    def select_actions(
        self,
        observations: np.ndarray,
        head_idx: Optional[int] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select actions for all agents using specified policy head.
        
        Args:
            observations: Agent observations, shape (n_agents, obs_dim)
            head_idx: Policy head index to use (if None, ABC selects)
            deterministic: Whether to use deterministic policy
            
        Returns:
            actions: Selected actions, shape (n_agents,)
            log_probs: Log probabilities, shape (n_agents,)
            head_used: Index of head that was used
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            
            # ABC selects head if not specified
            if head_idx is None:
                head_idx = self.abc_optimizer.select_best_head()
            
            # Get action distribution from selected head
            dist, _ = self.policy(obs_tensor, head_idx)
            
            # Sample or take mean
            if deterministic:
                actions = dist.mode()
            else:
                actions = dist.sample()
            
            log_probs = dist.log_prob(actions)
            
        return actions.cpu().numpy(), log_probs.cpu().numpy(), head_idx
    
    def compute_values(self, states: np.ndarray) -> np.ndarray:
        """
        Compute state values using centralized critic.
        
        Args:
            states: Centralized states, shape (batch_size, state_dim)
            
        Returns:
            values: State values, shape (batch_size,)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(states).to(self.device)
            values = self.critic(state_tensor)
        return values.cpu().numpy()
    
    def store_transition(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        head_used: int
    ):
        """Store a transition in the rollout buffer."""
        self.buffer.store(
            observations, actions, rewards, dones,
            values, log_probs, head_used
        )
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Reward sequence
            values: Value sequence
            dones: Done flags
            next_value: Value of next state
            
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        # Backward pass to compute advantages
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            
            # GAE
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        # Returns are advantages + values
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, batch_size: int = 32, n_epochs: int = 4) -> Dict[str, float]:
        """
        Update policy and critic networks using collected rollouts.
        
        Args:
            batch_size: Mini-batch size for updates
            n_epochs: Number of epochs to train
            
        Returns:
            Dictionary of training metrics
        """
        # Get data from buffer
        data = self.buffer.get()
        
        observations = torch.FloatTensor(data['observations']).to(self.device)
        actions = torch.LongTensor(data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        advantages = torch.FloatTensor(data['advantages']).to(self.device)
        returns = torch.FloatTensor(data['returns']).to(self.device)
        head_indices = data['head_indices']
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Storage for metrics
        policy_losses = []
        value_losses = []
        entropies = []
        
        # Multiple epochs of updates
        for _ in range(n_epochs):
            # Mini-batch updates
            indices = np.arange(len(observations))
            np.random.shuffle(indices)
            
            for start in range(0, len(observations), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                # Get batch data
                obs_batch = observations[batch_idx]
                act_batch = actions[batch_idx]
                old_lp_batch = old_log_probs[batch_idx]
                adv_batch = advantages[batch_idx]
                ret_batch = returns[batch_idx]
                
                # We'll use the most frequently selected head for this batch
                head_idx = max(set(head_indices), key=head_indices.count)
                
                # Forward pass through policy
                dist, _ = self.policy(obs_batch, head_idx)
                new_log_probs = dist.log_prob(act_batch)
                entropy = dist.entropy().mean()
                
                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - old_lp_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                # For centralized critic, we need to reshape observations
                batch_size_actual = obs_batch.shape[0] // self.n_agents
                centralized_state = obs_batch.reshape(batch_size_actual, -1)
                values = self.critic(centralized_state).squeeze()
                
                # Reshape returns to match
                ret_batch_reshaped = ret_batch.reshape(batch_size_actual, -1).mean(dim=1)
                value_loss = nn.MSELoss()(values, ret_batch_reshaped)
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.policy_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.policy_optimizer.step()
                self.critic_optimizer.step()
                
                # Record metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
        
        # Clear buffer
        self.buffer.clear()
        
        # Increment training step
        self.training_step += 1
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies)
        }
    
    def update_abc_fitness(self, head_idx: int, episode_return: float):
        """
        Update ABC optimizer with episode results.
        
        Args:
            head_idx: Index of head that was used
            episode_return: Total episode return achieved
        """
        self.abc_optimizer.update_fitness(head_idx, episode_return)
    
    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'abc_fitness': self.abc_optimizer.get_fitness(),
            'training_step': self.training_step
        }, filepath)
    
    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.abc_optimizer.set_fitness(checkpoint['abc_fitness'])
        self.training_step = checkpoint['training_step']
