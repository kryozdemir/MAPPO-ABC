"""
Artificial Bee Colony (ABC) Optimizer for Policy Head Selection

This module implements the ABC algorithm for selecting the most suitable
policy head based on fitness evaluation from episode returns.
"""

import numpy as np
from typing import List, Dict, Optional


class ABCOptimizer:
    """
    Artificial Bee Colony optimizer for multi-head policy selection.
    
    The ABC algorithm maintains a population of food sources (policy heads),
    where each source has an associated fitness value. The algorithm uses
    three types of bees:
    - Employed bees: Explore current food sources
    - Onlooker bees: Exploit promising food sources
    - Scout bees: Explore new solutions when sources are exhausted
    
    Args:
        n_heads: Number of policy heads (food sources)
        food_sources: Number of food sources (default: same as n_heads)
        iterations: Number of ABC iterations per update
        limit: Abandonment limit for food sources
    """
    
    def __init__(
        self,
        n_heads: int = 4,
        food_sources: int = 20,
        iterations: int = 50,
        limit: int = 10
    ):
        self.n_heads = n_heads
        self.food_sources = min(food_sources, n_heads)  # Can't have more sources than heads
        self.iterations = iterations
        self.limit = limit
        
        # Initialize fitness for each head
        # Higher fitness = better performance
        self.fitness = np.ones(n_heads) * 0.25  # Start with equal probability
        
        # Track how many times each head failed to improve
        self.trial_counter = np.zeros(n_heads, dtype=int)
        
        # Probability of selecting each head
        self.probabilities = np.ones(n_heads) / n_heads
        
        # Historical performance tracking
        self.performance_history = [[] for _ in range(n_heads)]
        
        # Episode counter for each head
        self.head_usage_count = np.zeros(n_heads, dtype=int)
        
    def update_fitness(self, head_idx: int, episode_return: float):
        """
        Update fitness value for a specific head based on episode performance.
        
        Args:
            head_idx: Index of the policy head
            episode_return: Episode return achieved by this head
        """
        # Store performance history
        self.performance_history[head_idx].append(episode_return)
        self.head_usage_count[head_idx] += 1
        
        # Compute fitness as exponential moving average of returns
        # We use negative returns since lower is better in many SMAC scenarios
        # Normalize to [0, 1] range with sigmoid
        alpha = 0.3  # EMA coefficient
        normalized_return = 1.0 / (1.0 + np.exp(-episode_return / 10.0))
        
        if self.head_usage_count[head_idx] == 1:
            # First update
            self.fitness[head_idx] = normalized_return
        else:
            # Exponential moving average
            self.fitness[head_idx] = alpha * normalized_return + (1 - alpha) * self.fitness[head_idx]
        
        # Update selection probabilities using softmax
        self._update_probabilities()
        
    def _update_probabilities(self):
        """Update selection probabilities based on current fitness values."""
        # Apply softmax with temperature for exploration
        temperature = 1.0
        exp_fitness = np.exp(self.fitness / temperature)
        self.probabilities = exp_fitness / np.sum(exp_fitness)
        
        # Ensure minimum exploration (epsilon-greedy component)
        epsilon = 0.1
        self.probabilities = (1 - epsilon) * self.probabilities + epsilon / self.n_heads
        
    def select_best_head(self) -> int:
        """
        Select the best policy head using ABC algorithm.
        
        The selection is based on:
        1. Current fitness values
        2. Exploration needs (less-tried heads get bonus)
        3. Probabilistic selection (not always greedy)
        
        Returns:
            Index of selected policy head
        """
        # ABC employed bee phase: explore current solutions
        for head_idx in range(self.n_heads):
            # Try to improve current solution
            if np.random.rand() < 0.5:
                # Exploration: try neighbor
                neighbor_idx = (head_idx + np.random.randint(1, self.n_heads)) % self.n_heads
                
                # Compare fitness
                if self.fitness[neighbor_idx] > self.fitness[head_idx]:
                    # Neighbor is better, increment trial counter
                    self.trial_counter[head_idx] += 1
                else:
                    # Current is better, reset trial counter
                    self.trial_counter[head_idx] = 0
        
        # ABC onlooker bee phase: exploit good solutions
        # Select based on fitness probability
        selected_idx = np.random.choice(self.n_heads, p=self.probabilities)
        
        # ABC scout bee phase: abandon exhausted sources
        for head_idx in range(self.n_heads):
            if self.trial_counter[head_idx] > self.limit:
                # Reset this head's fitness to encourage re-exploration
                self.fitness[head_idx] = np.random.uniform(0.2, 0.8)
                self.trial_counter[head_idx] = 0
        
        # Encourage exploration of under-utilized heads
        min_usage = np.min(self.head_usage_count)
        if min_usage < 10:  # First 10 episodes: ensure all heads are tried
            # Find least-used head
            least_used_idx = np.argmin(self.head_usage_count)
            if np.random.rand() < 0.3:  # 30% chance to select least-used
                selected_idx = least_used_idx
        
        return selected_idx
    
    def get_fitness(self) -> np.ndarray:
        """Get current fitness values for all heads."""
        return self.fitness.copy()
    
    def set_fitness(self, fitness: np.ndarray):
        """Set fitness values (used when loading checkpoints)."""
        self.fitness = fitness.copy()
        self._update_probabilities()
    
    def get_probabilities(self) -> np.ndarray:
        """Get current selection probabilities for all heads."""
        return self.probabilities.copy()
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about ABC optimizer state.
        
        Returns:
            Dictionary containing fitness, probabilities, and usage stats
        """
        stats = {
            'fitness': self.fitness.tolist(),
            'probabilities': self.probabilities.tolist(),
            'usage_count': self.head_usage_count.tolist(),
            'trial_counter': self.trial_counter.tolist()
        }
        
        # Add performance statistics if available
        for head_idx in range(self.n_heads):
            if len(self.performance_history[head_idx]) > 0:
                stats[f'head_{head_idx}_mean_return'] = np.mean(self.performance_history[head_idx])
                stats[f'head_{head_idx}_std_return'] = np.std(self.performance_history[head_idx])
        
        return stats
    
    def reset(self):
        """Reset ABC optimizer to initial state."""
        self.fitness = np.ones(self.n_heads) * 0.25
        self.trial_counter = np.zeros(self.n_heads, dtype=int)
        self.probabilities = np.ones(self.n_heads) / self.n_heads
        self.performance_history = [[] for _ in range(self.n_heads)]
        self.head_usage_count = np.zeros(self.n_heads, dtype=int)
