"""
Baseline MAPPO Algorithm (without ABC)

Standard Multi-Agent PPO implementation for comparison with MAPPO-ABC.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple

from src.networks.value_network import ValueNetwork
from src.algorithms.buffer import RolloutBuffer


class BaselineMAPPO:
    """
    Standard MAPPO algorithm without ABC optimization.
    
    Uses a single policy head for all agents.
    
    Args:
        n_agents: Number of agents
        obs_dim: Observation dimension
        act_dim: Action dimension
        hidden_dim: Hidden layer size
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_param: PPO clip parameter
        value_loss_coef: Value loss coefficient
        entropy_coef: Entropy coefficient
        max_grad_norm: Max gradient norm
        device: Device (cuda/cpu)
    """
    
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_param: float = 0.2,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 10.0,
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
        
        # Single policy network (no multi-head)
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        ).to(device)
        
        # Centralized critic
        self.critic = ValueNetwork(
            state_dim=obs_dim * n_agents,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Buffer
        self.buffer = RolloutBuffer()
    
    def select_actions(
        self,
        observations: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select actions using policy."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            logits = self.policy(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            
            if deterministic:
                actions = dist.mode()
            else:
                actions = dist.sample()
            
            log_probs = dist.log_prob(actions)
        
        return actions.cpu().numpy(), log_probs.cpu().numpy()
    
    def compute_values(self, states: np.ndarray) -> np.ndarray:
        """Compute state values."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(states).to(self.device)
            values = self.critic(state_tensor)
        return values.cpu().numpy()
    
    def update(self, batch_size: int = 32, n_epochs: int = 4) -> Dict[str, float]:
        """Update policy and critic."""
        data = self.buffer.get()
        
        observations = torch.FloatTensor(data['observations']).to(self.device)
        actions = torch.LongTensor(data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        advantages = torch.FloatTensor(data['advantages']).to(self.device)
        returns = torch.FloatTensor(data['returns']).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        policy_losses = []
        value_losses = []
        entropies = []
        
        for _ in range(n_epochs):
            indices = np.arange(len(observations))
            np.random.shuffle(indices)
            
            for start in range(0, len(observations), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                obs_batch = observations[batch_idx]
                act_batch = actions[batch_idx]
                old_lp_batch = old_log_probs[batch_idx]
                adv_batch = advantages[batch_idx]
                ret_batch = returns[batch_idx]
                
                # Policy forward
                logits = self.policy(obs_batch)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(act_batch)
                entropy = dist.entropy().mean()
                
                # Policy loss
                ratio = torch.exp(new_log_probs - old_lp_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                batch_size_actual = obs_batch.shape[0] // self.n_agents
                centralized_state = obs_batch.reshape(batch_size_actual, -1)
                values = self.critic(centralized_state).squeeze()
                ret_batch_reshaped = ret_batch.reshape(batch_size_actual, -1).mean(dim=1)
                value_loss = nn.MSELoss()(values, ret_batch_reshaped)
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Backward
                self.policy_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.policy_optimizer.step()
                self.critic_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
        
        self.buffer.clear()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies)
        }
    
    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
