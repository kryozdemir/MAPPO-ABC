"""
Value Decomposition Networks (VDN)

VDN decomposes the joint Q-value as a simple sum of individual Q-values,
enabling decentralized execution while maintaining centralized training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
from collections import deque
import random


class QNetwork(nn.Module):
    """
    Q-network for individual agent.
    
    Args:
        obs_dim: Observation dimension
        act_dim: Action dimension
        hidden_dim: Hidden layer size
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass to get Q-values."""
        return self.network(obs)


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: float,
        next_observations: np.ndarray,
        done: bool
    ):
        """Store a transition."""
        self.buffer.append({
            'observations': observations,
            'actions': actions,
            'reward': rewards,
            'next_observations': next_observations,
            'done': done
        })
    
    def sample(self, batch_size: int) -> Dict:
        """Sample a batch."""
        batch = random.sample(self.buffer, batch_size)
        
        return {
            'observations': np.array([t['observations'] for t in batch]),
            'actions': np.array([t['actions'] for t in batch]),
            'rewards': np.array([t['reward'] for t in batch]),
            'next_observations': np.array([t['next_observations'] for t in batch]),
            'dones': np.array([t['done'] for t in batch])
        }
    
    def __len__(self) -> int:
        return len(self.buffer)


class VDN:
    """
    Value Decomposition Networks for cooperative MARL.
    
    VDN assumes that the joint Q-value can be decomposed as:
    Q_tot(s, a) = sum_i Q_i(o_i, a_i)
    
    This simple additive decomposition enables fully decentralized execution
    while allowing centralized training with the global reward.
    
    Args:
        n_agents: Number of agents
        obs_dim: Observation dimension per agent
        act_dim: Action dimension per agent
        hidden_dim: Hidden dimension for Q-networks
        lr: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Exploration decay steps
        target_update_interval: Target network update frequency
        buffer_capacity: Replay buffer size
        device: Device to use
    """
    
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 50000,
        target_update_interval: int = 200,
        buffer_capacity: int = 10000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval
        self.device = device
        
        # Create Q-networks for each agent
        self.q_networks = nn.ModuleList([
            QNetwork(obs_dim, act_dim, hidden_dim)
            for _ in range(n_agents)
        ]).to(device)
        
        # Create target Q-networks
        self.target_q_networks = nn.ModuleList([
            QNetwork(obs_dim, act_dim, hidden_dim)
            for _ in range(n_agents)
        ]).to(device)
        
        # Initialize target networks
        for q_net, target_q_net in zip(self.q_networks, self.target_q_networks):
            target_q_net.load_state_dict(q_net.state_dict())
        
        # Optimizer for all Q-networks
        self.optimizer = optim.Adam(self.q_networks.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)
        
        # Training step counter
        self.steps = 0
    
    def select_actions(
        self,
        observations: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select actions for all agents using epsilon-greedy.
        
        Args:
            observations: Observations for all agents
            deterministic: If True, use greedy action selection
            
        Returns:
            Actions for all agents
        """
        actions = []
        
        with torch.no_grad():
            for i in range(self.n_agents):
                obs = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
                q_values = self.q_networks[i](obs)
                
                # Epsilon-greedy
                if deterministic or np.random.random() > self.epsilon:
                    action = q_values.argmax(dim=1).item()
                else:
                    action = np.random.randint(self.act_dim)
                
                actions.append(action)
        
        return np.array(actions)
    
    def update(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Update Q-networks using VDN.
        
        Args:
            batch_size: Mini-batch size
            
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) < batch_size:
            return {'loss': 0.0}
        
        # Sample batch
        batch = self.buffer.sample(batch_size)
        
        observations = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_observations = torch.FloatTensor(batch['next_observations']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        # Compute current Q-values for chosen actions
        chosen_action_qvals = []
        for i in range(self.n_agents):
            q_vals = self.q_networks[i](observations[:, i])
            chosen_q = q_vals.gather(1, actions[:, i].unsqueeze(1))
            chosen_action_qvals.append(chosen_q)
        
        # VDN: Sum individual Q-values
        q_tot = torch.stack(chosen_action_qvals, dim=1).sum(dim=1)
        
        # Compute target Q-values
        with torch.no_grad():
            target_qvals = []
            for i in range(self.n_agents):
                target_q = self.target_q_networks[i](next_observations[:, i])
                max_q = target_q.max(dim=1, keepdim=True)[0]
                target_qvals.append(max_q)
            
            # VDN: Sum target Q-values
            target_q_tot = torch.stack(target_qvals, dim=1).sum(dim=1)
            
            # TD target
            targets = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * target_q_tot
        
        # Compute loss
        loss = nn.MSELoss()(q_tot, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_networks.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target networks periodically
        self.steps += 1
        if self.steps % self.target_update_interval == 0:
            for q_net, target_q_net in zip(self.q_networks, self.target_q_networks):
                target_q_net.load_state_dict(q_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay
        )
        
        return {'loss': loss.item(), 'epsilon': self.epsilon}
    
    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'q_networks': [q.state_dict() for q in self.q_networks],
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        for i, q_state in enumerate(checkpoint['q_networks']):
            self.q_networks[i].load_state_dict(q_state)
            self.target_q_networks[i].load_state_dict(q_state)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']
