"""
Rollout Buffer for storing and processing episode trajectories.

This buffer stores transitions during episode rollouts and computes
advantages using Generalized Advantage Estimation (GAE).
"""

import numpy as np
from typing import Dict, List


class RolloutBuffer:
    """
    Buffer for storing trajectories during policy rollouts.
    
    Stores:
    - Observations
    - Actions
    - Rewards
    - Done flags
    - Value estimates
    - Log probabilities
    - Policy head indices
    
    Also computes GAE advantages and returns for policy updates.
    """
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.head_indices = []
        
        # Processed data
        self.advantages = None
        self.returns = None
    
    def store(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
        head_idx: int
    ):
        """
        Store a single transition.
        
        Args:
            observation: Agent observation
            action: Action taken
            reward: Reward received
            done: Episode termination flag
            value: Estimated state value
            log_prob: Log probability of action
            head_idx: Policy head used for action selection
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.head_indices.append(head_idx)
    
    def compute_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        next_value: float = 0.0
    ):
        """
        Compute GAE advantages and returns.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            next_value: Bootstrap value for final state
        """
        # Convert lists to arrays
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            # TD error
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            
            # GAE
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        
        # Returns are advantages + values
        self.advantages = advantages
        self.returns = advantages + values
    
    def get(self) -> Dict[str, np.ndarray]:
        """
        Get all stored data as a dictionary.
        
        Returns:
            Dictionary containing all buffer data
        """
        # Make sure advantages are computed
        if self.advantages is None:
            self.compute_advantages()
        
        return {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs),
            'head_indices': self.head_indices,
            'advantages': self.advantages,
            'returns': self.returns
        }
    
    def clear(self):
        """Clear all stored data."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.head_indices = []
        self.advantages = None
        self.returns = None
    
    def __len__(self) -> int:
        """Return number of stored transitions."""
        return len(self.observations)
