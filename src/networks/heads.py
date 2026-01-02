"""
Individual Policy Heads for Multi-Head Policy Network

Each policy head represents a potential exploration strategy that can be
selected by the ABC optimizer during training.
"""

import torch
import torch.nn as nn


class PolicyHead(nn.Module):
    """
    Individual policy head with specific characteristics.
    
    In MAPPO-ABC, heads are initialized with identical architectures but
    different random seeds. Behavioral specialization emerges naturally
    during training through ABC-based fitness-driven selection, rather
    than being manually assigned.
    
    Args:
        input_dim: Input feature dimension from shared encoder
        hidden_dim: Hidden layer dimension
        output_dim: Output action dimension
        temperature: Temperature for action distribution (default: 1.0)
        exploration_noise: Noise level for exploration (default: 0.0)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        temperature: float = 1.0,
        exploration_noise: float = 0.0
    ):
        super(PolicyHead, self).__init__()
        
        self.temperature = temperature
        self.exploration_noise = exploration_noise
        
        # Two-layer network for this head
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through head.
        
        Args:
            features: Shared features from backbone encoder
            
        Returns:
            Action logits scaled by temperature
        """
        logits = self.fc(features)
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Add exploration noise during training (if configured)
        if self.training and self.exploration_noise > 0:
            noise = torch.randn_like(logits) * self.exploration_noise
            logits = logits + noise
        
        return logits
