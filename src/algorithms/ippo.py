"""
Independent Proximal Policy Optimization (IPPO)

Independent PPO applies PPO to each agent separately, treating other agents
as part of the environment. This is a decentralized baseline for MARL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple

from src.algorithms.buffer import RolloutBuffer


class IPPOAgent:
    """
    Single agent using PPO independently.
    
    Each agent maintains its own policy and value network, and treats
    other agents as part of the environment dynamics.
    
    Args:
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
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Independent policy network for this agent
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        ).to(device)
        
        # Independent value network for this agent
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        # Separate optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """Select action using policy."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            logits = self.policy(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def compute_value(self, observation: np.ndarray) -> float:
        """Compute state value."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            value = self.value_net(obs_tensor)
        return value.item()
    
    def update(self, batch_size: int = 32, n_epochs: int = 4) -> Dict[str, float]:
        """Update policy and value networks."""
        data = self.buffer.get()
        
        observations = torch.FloatTensor(data['observations']).to(self.device)
        actions = torch.LongTensor(data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        advantages = torch.FloatTensor(data['advantages']).to(self.device)
        returns = torch.FloatTensor(data['returns']).to(self.device)
        
        # Normalize advantages
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
                
                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - old_lp_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = self.value_net(obs_batch).squeeze()
                value_loss = nn.MSELoss()(values, ret_batch)
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
        
        self.buffer.clear()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies)
        }


class IPPO:
    """
    Independent PPO for multi-agent systems.
    
    Maintains separate PPO agents for each agent in the environment.
    
    Args:
        n_agents: Number of agents
        obs_dim: Observation dimension
        act_dim: Action dimension
        **kwargs: Additional arguments passed to IPPOAgent
    """
    
    def __init__(self, n_agents: int, obs_dim: int, act_dim: int, **kwargs):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Create independent agents
        self.agents = [
            IPPOAgent(obs_dim, act_dim, **kwargs)
            for _ in range(n_agents)
        ]
    
    def select_actions(self, observations: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Select actions for all agents."""
        actions = []
        log_probs = []
        
        for i, agent in enumerate(self.agents):
            action, log_prob = agent.select_action(observations[i], deterministic)
            actions.append(action)
            log_probs.append(log_prob)
        
        return np.array(actions), np.array(log_probs)
    
    def compute_values(self, observations: np.ndarray) -> np.ndarray:
        """Compute values for all agents."""
        values = []
        for i, agent in enumerate(self.agents):
            value = agent.compute_value(observations[i])
            values.append(value)
        return np.array(values)
    
    def store_transitions(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray
    ):
        """Store transitions for all agents."""
        for i, agent in enumerate(self.agents):
            agent.buffer.store(
                observations[i],
                actions[i],
                rewards[i],
                dones[i],
                values[i],
                log_probs[i],
                head_idx=0  # Not used in IPPO
            )
    
    def update(self, batch_size: int = 32, n_epochs: int = 4) -> Dict[str, float]:
        """Update all agents."""
        all_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': []
        }
        
        for agent in self.agents:
            # Compute advantages for this agent
            agent.buffer.compute_advantages()
            
            # Update this agent
            metrics = agent.update(batch_size, n_epochs)
            
            for key, value in metrics.items():
                all_metrics[key].append(value)
        
        # Average metrics across agents
        return {key: np.mean(values) for key, values in all_metrics.items()}
    
    def save(self, filepath: str):
        """Save all agents."""
        checkpoint = {
            f'agent_{i}_policy': agent.policy.state_dict()
            for i, agent in enumerate(self.agents)
        }
        checkpoint.update({
            f'agent_{i}_value': agent.value_net.state_dict()
            for i, agent in enumerate(self.agents)
        })
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str):
        """Load all agents."""
        checkpoint = torch.load(filepath)
        for i, agent in enumerate(self.agents):
            agent.policy.load_state_dict(checkpoint[f'agent_{i}_policy'])
            agent.value_net.load_state_dict(checkpoint[f'agent_{i}_value'])
