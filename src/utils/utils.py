"""
Utility functions for MAPPO-ABC.
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_grad_norm(model: torch.nn.Module) -> float:
    """
    Calculate the gradient norm of a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def linear_schedule(initial_value: float, final_value: float, current_step: int, total_steps: int) -> float:
    """
    Linear annealing schedule.
    
    Args:
        initial_value: Starting value
        final_value: Ending value
        current_step: Current timestep
        total_steps: Total timesteps
        
    Returns:
        Annealed value
    """
    fraction = min(current_step / total_steps, 1.0)
    return initial_value + (final_value - initial_value) * fraction


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute explained variance.
    
    Measures how well predictions explain the variance in true values.
    
    Args:
        y_pred: Predicted values
        y_true: True values
        
    Returns:
        Explained variance ratio
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
