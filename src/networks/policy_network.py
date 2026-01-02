"""
Multi-Head Policy Network for MAPPO-ABC

This module implements a policy network with multiple heads, each configured
with different exploration-exploitation trade-offs.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple, Optional


class PolicyHead(nn.Module):
    """
    Individual policy head with specific exploration characteristics.
    
    Each head has:
    - Different action selection temperature
    - Different entropy regularization
    - Different exploration noise levels
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output action dimension
        temperature: Temperature for action distribution
        exploration_noise: Noise level for exploration
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        temperature: float = 1.0,
        exploration_noise: float = 0.1
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
            features: Shared features from backbone
            
        Returns:
            Action logits scaled by temperature
        """
        logits = self.fc(features)
        
        # Apply temperature scaling
        logits = logits / self.temperature
        
        # Add exploration noise during training
        if self.training and self.exploration_noise > 0:
            noise = torch.randn_like(logits) * self.exploration_noise
            logits = logits + noise
        
        return logits


class MultiHeadPolicyNetwork(nn.Module):
    """
    Multi-head policy network with 4 specialized heads.
    
    Architecture:
    - Shared backbone network
    - 4 policy heads with different exploration strategies:
        H0: Conservative (low exploration)
        H1: Balanced (medium exploration)
        H2: Aggressive (high exploration)
        H3: Exploratory (very high exploration)
    
    Args:
        obs_dim: Observation dimension
        act_dim: Action space dimension
        hidden_dim: Hidden layer dimension
        n_heads: Number of policy heads (default: 4)
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        n_heads: int = 4
    ):
        super(MultiHeadPolicyNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # Shared feature extraction backbone
        # This processes raw observations into useful features
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Initialize 4 policy heads with different random seeds
        # Heads are NOT manually assigned roles - behavioral specialization
        # emerges naturally during training through ABC-based selection
        # As stated in the paper: "Policy heads are initialized using different 
        # random seeds, introducing stochastic variation in their initial action preferences"
        head_configs = [
            {'temperature': 1.0, 'exploration_noise': 0.0},   # Head 0
            {'temperature': 1.0, 'exploration_noise': 0.0},   # Head 1
            {'temperature': 1.0, 'exploration_noise': 0.0},   # Head 2
            {'temperature': 1.0, 'exploration_noise': 0.0}    # Head 3
        ]
        
        self.heads = nn.ModuleList([
            PolicyHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                output_dim=act_dim,
                **config
            )
            for config in head_configs[:n_heads]
        ])
        
        # Initialize weights with different random seeds for each head
        # This creates initial behavioral diversity without manual role assignment
        for i, head in enumerate(self.heads):
            torch.manual_seed(42 + i)  # Different seed for each head
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=0.01)
                    nn.init.constant_(module.bias, 0)
        
        # Reset to original seed after initialization
        torch.manual_seed(42)
    
    def forward(
        self,
        observations: torch.Tensor,
        head_idx: int,
        return_features: bool = False
    ) -> Tuple[Categorical, Optional[torch.Tensor]]:
        """
        Forward pass through network.
        
        Args:
            observations: Agent observations, shape (batch_size, obs_dim)
            head_idx: Index of policy head to use (0-3)
            return_features: Whether to return backbone features
            
        Returns:
            distribution: Categorical action distribution
            features: Backbone features (if return_features=True)
        """
        # Extract shared features
        features = self.backbone(observations)
        
        # Get action logits from selected head
        logits = self.heads[head_idx](features)
        
        # Create categorical distribution
        distribution = Categorical(logits=logits)
        
        if return_features:
            return distribution, features
        else:
            return distribution, None
    
    def get_head_outputs(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Get outputs from all heads (useful for analysis).
        
        Args:
            observations: Agent observations
            
        Returns:
            Tensor of shape (n_heads, batch_size, act_dim)
        """
        features = self.backbone(observations)
        
        outputs = []
        for head in self.heads:
            logits = head(features)
            outputs.append(logits)
        
        return torch.stack(outputs, dim=0)
    
    def get_action_probabilities(
        self,
        observations: torch.Tensor,
        head_idx: int
    ) -> torch.Tensor:
        """
        Get action probabilities from specific head.
        
        Args:
            observations: Agent observations
            head_idx: Policy head index
            
        Returns:
            Action probabilities, shape (batch_size, act_dim)
        """
        with torch.no_grad():
            dist, _ = self.forward(observations, head_idx)
            probs = dist.probs
        return probs
    
    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        head_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions taken by a specific head.
        
        Used during policy updates to compute log probabilities and entropy.
        
        Args:
            observations: Agent observations
            actions: Actions that were taken
            head_idx: Policy head that was used
            
        Returns:
            log_probs: Log probabilities of actions
            entropy: Entropy of action distribution
        """
        dist, _ = self.forward(observations, head_idx)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy
