"""
QMIX: Monotonic Value Function Factorization for Deep Multi-Agent RL

QMIX uses a mixing network to combine individual agent Q-values into a 
joint Q-value while maintaining monotonicity for decentralized execution.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
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
        """Forward pass to get Q-values for all actions."""
        return self.network(obs)


class QMixingNetwork(nn.Module):
    """
    Mixing network that combines individual Q-values monotonically.
    
    The mixing network ensures that the joint Q-value is monotonic in
    each agent's Q-value, enabling decentralized execution.
    
    Args:
        n_agents: Number of agents
        state_dim: Global state dimension
        hidden_dim: Hidden dimension for mixing network
    """
    
    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 64):
        super(QMixingNetwork, self).__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Hypernetworks that generate weights for mixing network
        # Weights must be positive to ensure monotonicity
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Hypernetworks for biases
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Mix agent Q-values into joint Q-value.
        
        Args:
            agent_qs: Individual agent Q-values, shape (batch, n_agents)
            states: Global states, shape (batch, state_dim)
            
        Returns:
            Mixed Q-value, shape (batch, 1)
        """
        batch_size = agent_qs.shape[0]
        
        # First layer weights (must be positive)
        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)
        
        # First layer bias
        b1 = self.hyper_b1(states)
        b1 = b1.view(batch_size, 1, self.hidden_dim)
        
        # First layer
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)
        hidden = torch.relu(torch.bmm(agent_qs, w1) + b1)
        
        # Second layer weights (must be positive)
        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        
        # Second layer bias
        b2 = self.hyper_b2(states)
        b2 = b2.view(batch_size, 1, 1)
        
        # Final Q-value
        q_tot = torch.bmm(hidden, w2) + b2
        
        return q_tot.view(batch_size, 1)


class ReplayBuffer:
    """
    Experience replay buffer for QMIX.
    
    Args:
        capacity: Maximum buffer size
    """
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: float,
        next_observations: np.ndarray,
        done: bool,
        state: np.ndarray,
        next_state: np.ndarray
    ):
        """Store a transition."""
        self.buffer.append({
            'observations': observations,
            'actions': actions,
            'reward': rewards,
            'next_observations': next_observations,
            'done': done,
            'state': state,
            'next_state': next_state
        })
    
    def sample(self, batch_size: int) -> Dict:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        
        return {
            'observations': np.array([t['observations'] for t in batch]),
            'actions': np.array([t['actions'] for t in batch]),
            'rewards': np.array([t['reward'] for t in batch]),
            'next_observations': np.array([t['next_observations'] for t in batch]),
            'dones': np.array([t['done'] for t in batch]),
            'states': np.array([t['state'] for t in batch]),
            'next_states': np.array([t['next_state'] for t in batch])
        }
    
    def __len__(self) -> int:
        return len(self.buffer)


class QMIX:
    """
    QMIX algorithm for cooperative multi-agent RL.
    
    Args:
        n_agents: Number of agents
        obs_dim: Observation dimension per agent
        act_dim: Action dimension per agent
        state_dim: Global state dimension
        hidden_dim: Hidden dimension for Q-networks
        lr: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Exploration decay rate
        target_update_interval: Target network update frequency
        buffer_capacity: Replay buffer capacity
        device: Device to use
    """
    
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        state_dim: int,
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
        self.state_dim = state_dim
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
        
        # Copy initial weights to target networks
        for q_net, target_q_net in zip(self.q_networks, self.target_q_networks):
            target_q_net.load_state_dict(q_net.state_dict())
        
        # Mixing networks
        self.mixer = QMixingNetwork(n_agents, state_dim).to(device)
        self.target_mixer = QMixingNetwork(n_agents, state_dim).to(device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        # Optimizer
        self.params = list(self.q_networks.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.Adam(self.params, lr=lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)
        
        # Training step counter
        self.steps = 0
    
    def select_actions(
        self,
        observations: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """Select actions for all agents using epsilon-greedy."""
        actions = []
        
        with torch.no_grad():
            for i in range(self.n_agents):
                obs = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
                q_values = self.q_networks[i](obs)
                
                # Epsilon-greedy action selection
                if deterministic or np.random.random() > self.epsilon:
                    action = q_values.argmax(dim=1).item()
                else:
                    action = np.random.randint(self.act_dim)
                
                actions.append(action)
        
        return np.array(actions)
    
    def update(self, batch_size: int = 32) -> Dict[str, float]:
        """Update Q-networks and mixer using sampled batch."""
        if len(self.buffer) < batch_size:
            return {'loss': 0.0}
        
        # Sample batch
        batch = self.buffer.sample(batch_size)
        
        observations = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_observations = torch.FloatTensor(batch['next_observations']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        states = torch.FloatTensor(batch['states']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        
        # Compute current Q-values
        chosen_action_qvals = []
        for i in range(self.n_agents):
            q_vals = self.q_networks[i](observations[:, i])
            chosen_q = q_vals.gather(1, actions[:, i].unsqueeze(1))
            chosen_action_qvals.append(chosen_q)
        
        chosen_action_qvals = torch.stack(chosen_action_qvals, dim=1).squeeze(-1)
        
        # Mix Q-values
        q_tot = self.mixer(chosen_action_qvals, states)
        
        # Compute target Q-values
        with torch.no_grad():
            target_qvals = []
            for i in range(self.n_agents):
                target_q = self.target_q_networks[i](next_observations[:, i])
                target_qvals.append(target_q.max(dim=1)[0])
            
            target_qvals = torch.stack(target_qvals, dim=1)
            target_q_tot = self.target_mixer(target_qvals, next_states)
            
            # Compute TD target
            targets = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * target_q_tot
        
        # Compute loss
        loss = nn.MSELoss()(q_tot, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, 10.0)
        self.optimizer.step()
        
        # Update target networks
        self.steps += 1
        if self.steps % self.target_update_interval == 0:
            for q_net, target_q_net in zip(self.q_networks, self.target_q_networks):
                target_q_net.load_state_dict(q_net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay
        )
        
        return {'loss': loss.item(), 'epsilon': self.epsilon}
    
    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'q_networks': [q.state_dict() for q in self.q_networks],
            'mixer': self.mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        for i, q_state in enumerate(checkpoint['q_networks']):
            self.q_networks[i].load_state_dict(q_state)
            self.target_q_networks[i].load_state_dict(q_state)
        self.mixer.load_state_dict(checkpoint['mixer'])
        self.target_mixer.load_state_dict(checkpoint['mixer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']
