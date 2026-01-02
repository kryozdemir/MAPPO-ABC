"""
Centralized Value Network (Critic) for MAPPO-ABC

This module implements the centralized critic that estimates state values
using global state information from all agents.
"""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    Centralized critic network for value estimation.
    
    In MAPPO, the critic has access to global state information (observations
    from all agents) while each agent's policy only sees local observations.
    This asymmetry helps with credit assignment in multi-agent settings.
    
    Args:
        state_dim: Centralized state dimension (typically obs_dim * n_agents)
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(ValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Three-layer network for value estimation
        # We use a deeper network for the critic since it has more information
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Orthogonal initialization helps with training stability
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to estimate state value.
        
        Args:
            state: Centralized state, shape (batch_size, state_dim)
            
        Returns:
            value: Estimated state value, shape (batch_size, 1)
        """
        return self.network(state)
