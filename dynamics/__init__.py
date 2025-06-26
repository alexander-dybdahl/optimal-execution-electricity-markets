"""
Dynamics module for optimal execution in electricity markets.

This module provides a factory function to dynamically create dynamics instances
based on configuration.
"""

from .aid_dynamics import AidDynamics
from .full_dynamics import FullDynamics
from .simple_dynamics import SimpleDynamics

# Registry of available dynamics classes
DYNAMICS_REGISTRY = {
    "SimpleDynamics": SimpleDynamics,
    "AidDynamics": AidDynamics,
    "FullDynamics": FullDynamics
}


def create_dynamics(dynamics_cfg, device):
    """
    Factory function to create dynamics instances based on configuration.
    
    Args:
        dynamics_cfg (dict): Configuration dictionary containing dynamics type and parameters
        device: Device to use for computations (cpu or cuda)
        
    Returns:
        Dynamics instance of the specified type
        
    Raises:
        ValueError: If the dynamics type is not recognized
    """
    dynamics_class_name = dynamics_cfg["dynamics"]
    
    if dynamics_class_name not in DYNAMICS_REGISTRY:
        available_classes = list(DYNAMICS_REGISTRY.keys())
        raise ValueError(f"Unknown dynamics class: {dynamics_class_name}. Available classes: {available_classes}")
    
    dynamics_class = DYNAMICS_REGISTRY[dynamics_class_name]
    return dynamics_class(dynamics_cfg=dynamics_cfg, device=device)


# Make classes available for direct import if needed
__all__ = ["AidDynamics", "FullDynamics", "SimpleDynamics", "create_dynamics", "DYNAMICS_REGISTRY"]
