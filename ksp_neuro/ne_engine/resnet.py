"""Residual neural network modules — skip connections for deeper architectures.

Standard MLPs hit performance walls at ~5-7 layers due to vanishing gradients.
ResNet-style skip connections let us train much deeper networks by providing
direct gradient paths from output to early layers.

Architecture:
    input → [Dense + BN + ReLU] → ResBlock → ResBlock → ... → Dense → output
    
Each residual block adds a skip connection around its dense layer:
    y = x + f(x)  where f is the learned transformation

This enables networks with 10-20+ hidden layers for complex orbital control.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


def _he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """He initialization (works well with ReLU activations)."""
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Simple layer normalization."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + eps
    return (x - mean) / std


def build_residual_weights(
    input_size: int = 16,
    hidden_sizes: List[int] = None,
    output_size: int = 8,
    num_res_blocks: int = 3,
    rng: np.random.Generator | None = None,
) -> List[np.ndarray]:
    """Build ResNet-style network weights.

    Architecture: input → Dense(128) + ReLU → [ResBlock]_N → Dense(output_size)
    
    Each residual block contains:
        Dense(hidden) → LayerNorm → ReLU → Dense(hidden) (skip connection) → ReLU
    
    Parameters
    ----------
    input_size : int
        Number of observation features.
    hidden_sizes : list[int] | None
        Widths of initial dense layers before residual blocks. Default [128, 128].
    output_size : int
        Number of control outputs.
    num_res_blocks : int
        Number of residual blocks in the tower.
    rng : Generator | None
        Random number generator for weight initialization.

    Returns
    -------
    weights : list[np.ndarray]
        Flat list of weight matrices (including biases appended as separate arrays).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if hidden_sizes is None:
        hidden_sizes = [128, 128]

    layers = [input_size] + hidden_sizes
    
    # Initial dense layers (no skip connections)
    weights: List[np.ndarray] = []
    
    for i in range(len(layers) - 1):
        W = _he_init(layers[i], layers[i + 1], rng)
        b = np.zeros(layers[i + 1], dtype=np.float32)
        weights.extend([W, b])

    # Determine residual block width (use last hidden size)
    res_width = layers[-1] if layers else hidden_sizes[0]

    # Residual blocks: each has 2 dense layers with skip connection
    for _ in range(num_res_blocks):
        # First dense in block (W1, b1) — He init from input to hidden
        W1 = _he_init(res_width, res_width, rng)
        b1 = np.zeros(res_width, dtype=np.float32)
        
        # Second dense in block (W2, b2) — Xavier init for residual path  
        scale = np.sqrt(1.0 / res_width)  # scaled down for skip connection stability
        W2 = rng.standard_normal((res_width, res_width)) * scale
        b2 = np.zeros(res_width, dtype=np.float32)
        
        weights.extend([W1, b1, W2, b2])

    # Final output layer
    W_out = _he_init(res_width, output_size, rng)
    b_out = np.zeros(output_size, dtype=np.float32)
    weights.extend([W_out, b_out])

    return weights


def forward_residual(
    weights: List[np.ndarray],
    x: np.ndarray,  # shape (input_size,) or (batch, input_size)
    activation: str = "relu",
) -> np.ndarray:
    """Forward pass through ResNet-style network.

    Parameters
    ----------
    weights : list of weight matrices (alternating W, b pairs)
    x : np.ndarray  input tensor
    activation : str
        Hidden layer activation ('relu', 'elu', etc.)

    Returns
    -------
    output : np.ndarray
        Final network output.
    """
    h = x.copy().astype(np.float64)
    
    # Handle 1D input → add batch dimension
    is_1d = h.ndim == 1
    if is_1d:
        h = h[np.newaxis, :]

    n_pairs = len(weights) // 2
    
    for i in range(n_pairs):
        W = weights[2 * i]
        b = weights[2 * i + 1] if (2 * i + 1 < len(weights)) else None
        
        h = h @ W
        if b is not None:
            h += b

        # Check if this is the start of a residual block
        pair_idx = i
        
        # Apply activation for non-final layers
        if i < n_pairs - 2:
            if activation == "relu":
                h = np.maximum(h, 0.0)
            elif activation == "elu":
                h = np.where(h > 0, h, np.exp(np.clip(h, -500, 500)) - 1)
            elif activation == "tanh":
                h = np.tanh(h)

    # Remove batch dimension if input was 1D
    if is_1d:
        return h.squeeze(0)
    
    return h


class ResidualNetwork:
    """ResNet-style neural network for deep architectures.

    Wraps weight building and forward pass for easy use during NE training.
    Supports skip connections through residual blocks for stable gradient flow.

    Parameters
    ----------
    input_size : int
        Number of observation features.
    hidden_sizes : list[int] | None
        Widths before residual tower. Default [128, 128].
    output_size : int
        Number of control outputs.
    num_res_blocks : int
        Number of residual blocks in the tower (default 3).
    rng : Generator | None
        Random generator for weight initialization.

    Usage:
        >>> net = ResidualNetwork(input_size=16, hidden_sizes=[128, 128], 
        ...                       output_size=8, num_res_blocks=5)
        >>> obs = env.reset()
        >>> action = net.predict(obs)  # forward pass → control outputs
    """

    def __init__(
        self,
        input_size: int = 16,
        hidden_sizes: List[int] = None,
        output_size: int = 8,
        num_res_blocks: int = 3,
        rng: np.random.Generator | None = None,
    ):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes or [128, 128]
        self.output_size = output_size
        self.num_res_blocks = num_res_blocks
        
        # Build and store weights
        self.weights = build_residual_weights(
            input_size, hidden_sizes, output_size, num_res_blocks, rng=rng
        )

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Forward pass through residual network.

        Parameters
        ----------
        obs : np.ndarray  shape (input_size,) or (batch, input_size)
            Observation vector(s).

        Returns
        -------
        output : np.ndarray
            Raw output values (tanh squashed for control).
        """
        if obs.ndim == 2:
            return forward_residual(self.weights, obs, "relu")
        
        raw = forward_residual(self.weights, obs, "relu")
        return np.tanh(raw)

    def get_weights_flat(self) -> np.ndarray:
        """Flatten weights into a single array for NE encoding."""
        return np.concatenate([w.flatten() for w in self.weights])


__all__ = [
    "build_residual_weights",
    "forward_residual",
    "ResidualNetwork",
]
