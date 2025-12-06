"""
Exponential Moving Average (EMA) helper for model weights.

EMA maintains a shadow copy of model weights that is updated as:
    ema_weights = mu * ema_weights + (1 - mu) * model_weights

This often leads to better generalization during evaluation.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn


class EMAHelper:
    """
    Exponential Moving Average helper for model weights.
    
    Usage:
        ema = EMAHelper(mu=0.999)
        ema.register(model)
        
        # During training, after each step:
        ema.update(model)
        
        # For evaluation:
        ema_model = ema.ema_copy(model)
        
        # Or apply EMA weights directly:
        ema.apply_ema(model)
        # ... evaluate ...
        ema.restore(model)
    """
    
    def __init__(self, mu: float = 0.999, device: Optional[torch.device] = None):
        """
        Initialize EMA helper.
        
        Args:
            mu: EMA decay rate (0.999 means 99.9% of old weights kept)
            device: Device to store EMA weights (None = same as model)
        """
        self.mu = mu
        self.device = device
        self.shadow = {}
        self.backup = {}
    
    def register(self, model: nn.Module):
        """
        Register model parameters for EMA tracking.
        
        Args:
            model: PyTorch model to track
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                if self.device is not None:
                    self.shadow[name] = param.data.clone().to(self.device)
                else:
                    self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """
        Update EMA weights with current model weights.
        
        Args:
            model: Model with updated weights
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    # Register new parameters dynamically
                    if self.device is not None:
                        self.shadow[name] = param.data.clone().to(self.device)
                    else:
                        self.shadow[name] = param.data.clone()
                else:
                    # EMA update: shadow = mu * shadow + (1 - mu) * param
                    new_average = self.mu * self.shadow[name] + (1.0 - self.mu) * param.data
                    self.shadow[name] = new_average.clone()
    
    def apply_ema(self, model: nn.Module):
        """
        Apply EMA weights to model (backs up current weights).
        
        Args:
            model: Model to apply EMA weights to
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model: nn.Module):
        """
        Restore original weights (undo apply_ema).
        
        Args:
            model: Model to restore weights to
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def ema_copy(self, model: nn.Module) -> nn.Module:
        """
        Create a copy of the model with EMA weights.
        
        Args:
            model: Original model
            
        Returns:
            New model instance with EMA weights
        """
        ema_model = copy.deepcopy(model)
        for name, param in ema_model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
        return ema_model
    
    def state_dict(self) -> dict:
        """Get EMA state for checkpointing."""
        return {
            "mu": self.mu,
            "shadow": self.shadow.copy(),
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load EMA state from checkpoint."""
        self.mu = state_dict["mu"]
        self.shadow = state_dict["shadow"].copy()

