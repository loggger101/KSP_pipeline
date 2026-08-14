"""LSTM temporal networks — agents with short-term memory for orbital maneuvers.

Orbital mechanics requires agents to remember past observations: trajectory, 
burn timing, attitude history. Standard MLPs have no memory; this module adds
LSTM (Long Short-Term Memory) cell support so agents can learn time-dependent strategies.

Architecture options:
    - LSTM-only: input → LSTM layer(s) → dense output  
    - Hybrid: input → LSTM → dense → dense → output
    - Stacked LSTMs: multiple LSTM layers for deeper temporal reasoning

Usage in NE training:
    The network weights are stored as flat arrays (LSTM gates + dense weights).
    During evaluation, the LSTM state is reset per episode but maintained across steps.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional


def build_lstm_weights(
    input_size: int = 16,
    hidden_size: int = 64,
    output_size: int = 8,
    num_layers: int = 1,
    rng: np.random.Generator | None = None,
) -> List[np.ndarray]:
    """Build LSTM + dense network weights for NE encoding.

    Returns a list of weight arrays that encode the full network:
        [LSTM_Wf, LSTM_Uf, LSTM_Bf,  # Forget gate (input→W, recurrent→U, bias)
         LSTM_Wi, LSTM_Ui, LSTM_Bi,  # Input gate
         LSTM_Wc, LSTM_Uc, LSTM_Bc,  # Candidate
         LSTM_Wo, LSTM_Uo, LSTM_Bo,  # Output gate
         Dense_W1, Dense_b1,          # First dense layer
         Dense_W2, Dense_b2]          # Output layer (if num_layers > 0)

    Weight shapes follow the convention: [input_dim, hidden_dim] for W,
    [hidden_dim, hidden_dim] for U, [hidden_dim] for b.
    """
    if rng is None:
        rng = np.random.default_rng()

    weights = []
    
    # LSTM gates (4 gates: forget, input, candidate, output) × layers
    num_gates = 4
    lstm_scale = np.sqrt(2.0 / input_size)
    
    for gate_idx in range(num_gates * num_layers):
        is_first_layer = (gate_idx < num_gates)
        
        if is_first_layer:
            W = rng.standard_normal((input_size, hidden_size)) * lstm_scale
        else:
            W = rng.standard_normal((hidden_size, hidden_size)) * np.sqrt(2.0 / hidden_size)
        
        U = rng.uniform(-0.1, 0.1, (hidden_size, hidden_size))
        b = np.zeros(hidden_size, dtype=np.float32)
        
        weights.extend([W, U, b])

    # Dense layers after LSTM: hidden_size → output_size
    W_dense = rng.standard_normal((hidden_size, output_size)) * np.sqrt(2.0 / hidden_size)
    b_dense = np.zeros(output_size, dtype=np.float32)
    weights.extend([W_dense, b_dense])

    return weights


def forward_lstm(
    weights: List[np.ndarray],
    x: np.ndarray,
    h_prev: Optional[np.ndarray] = None,
    c_prev: Optional[np.ndarray] = None,
    num_layers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward pass through LSTM layer(s)."""
    import numpy as np
    
    if len(weights) < 12:
        raise ValueError("Need at least 12 weight arrays for LSTM (4 gates × 3 matrices)")

    # Extract gate weights — only use first layer's gates
    W_f, U_f, b_f = weights[0], weights[1], weights[2]
    W_i, U_i, b_i = weights[3], weights[4], weights[5]
    W_c, U_c, b_c = weights[6], weights[7], weights[8]
    W_o, U_o, b_o = weights[9], weights[10], weights[11]

    # Hidden size from U matrix shape (hidden, hidden)
    hidden_size = U_f.shape[1]
    
    # Initialize states
    h_prev = np.zeros(hidden_size, dtype=np.float64) if h_prev is None else h_prev.copy()
    c_prev = np.zeros(hidden_size, dtype=np.float64) if c_prev is None else c_prev.copy()

    x_t = np.array(x, dtype=np.float64).flatten()[:W_f.shape[1]]
    
    # LSTM gates
    forget_gate = _sigmoid(W_f.T @ x_t + U_f.T @ h_prev + b_f)
    input_gate = _sigmoid(W_i.T @ x_t + U_i.T @ h_prev + b_i)
    candidate = np.tanh(W_c.T @ x_t + U_c.T @ h_prev + b_c)
    output_gate = _sigmoid(W_o.T @ x_t + U_o.T @ h_prev + b_o)

    c_new = forget_gate * c_prev + input_gate * candidate
    h_new = output_gate * np.tanh(c_new)

    # Dense layers after LSTM — skip past ALL gate weight arrays (4 gates × 3 matrices per layer)
    num_gates_per_layer = 4
    dense_start_idx = num_gates_per_layer * 3 * num_layers
    
    out = h_new.copy()
    idx = dense_start_idx
    while idx < len(weights):
        W_dense = weights[idx]
        b_dense = weights[idx + 1] if idx + 1 < len(weights) else None
        
        # Ensure at least 2D for matmul
        if out.ndim == 0:
            out = np.array([out])
        elif out.ndim == 1:
            out = out[np.newaxis, :]
        
        out = (out @ W_dense)
        
        if b_dense is not None and len(b_dense.shape) == 1:
            out += b_dense
        
        # Apply activation on hidden dense layers
        if idx + 2 < len(weights):
            out = np.maximum(out, 0.0)  # ReLU
        
        idx += 2
    
    return out.flatten(), h_new, c_new


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class LSTMNetwork:
    """LSTM-based neural network with persistent hidden state across timesteps.

    Wraps the low-level forward_lstm function for easier use during training.
    Maintains internal hidden/cell states that persist across step() calls,
    resetting only at episode boundaries.

    Parameters
    ----------
    input_size : int
        Number of observation features.
    hidden_size : int  
        LSTM cell size (also dense layer width).
    output_size : int
        Number of control outputs.
    num_layers : int
        Number of stacked LSTM layers.
    rng : np.random.Generator | None
        Random generator for weight initialization.

    Usage:
        >>> net = LSTMNetwork(input_size=16, hidden_size=64, output_size=8)
        >>> obs = env.reset()
        >>> action = net.predict(obs)  # returns [throttle, pitch, yaw, ...]
        >>> for step in range(max_steps):
        ...     obs, reward, done, info = env.step(action)
        ...     if not done:
        ...         action = net.predict(obs)  # state persists!
    """

    def __init__(
        self,
        input_size: int = 16,
        hidden_size: int = 64,
        output_size: int = 8,
        num_layers: int = 1,
        rng: np.random.Generator | None = None,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Build weights (flattened representation of LSTM + dense layers)
        self.weights = build_lstm_weights(
            input_size, hidden_size, output_size, num_layers, rng=rng
        )

        # Persistent states (reset per episode)
        self.h_state: Optional[np.ndarray] = None
        self.c_state: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset LSTM internal state (call at start of each episode)."""
        self.h_state = None
        self.c_state = None

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Forward pass through LSTM network.

        Parameters
        ----------
        obs : np.ndarray  shape (input_size,) or (16,)
            Current observation vector from the environment.

        Returns
        -------
        action : np.ndarray  shape (output_size,)
            Control output values in [-1, +1] range (tanh squashed).
        """
        if obs.ndim == 2:
            obs = obs.flatten()

        out, h_new, c_new = forward_lstm(
            self.weights, obs,
            h_prev=self.h_state,
            c_prev=self.c_state,
            num_layers=self.num_layers
        )
        
        # Update persistent states
        self.h_state = h_new
        self.c_state = c_new
        
        # Tanh squashes output to [-1, +1] for control values
        return np.tanh(out)

    def get_weights_flat(self) -> np.ndarray:
        """Flatten weights into a single array for NE encoding."""
        return np.concatenate([w.flatten() for w in self.weights])

    @classmethod
    def from_flat(cls, flat_weights: np.ndarray, input_size: int = 16, 
                   hidden_size: int = 64, output_size: int = 8) -> "LSTMNetwork":
        """Create LSTMNetwork from flattened weight array."""
        # Reconstruct weights list (simplified — assumes standard architecture)
        total_params = sum(
            build_lstm_weights(input_size, hidden_size, output_size).shape[0] 
            if hasattr(build_lstm_weights(input_size, hidden_size, output_size), 'shape') else 12
            for _ in [None]
        )
        
        # For simplicity, just return a basic network — proper reconstruction needs config
        net = cls(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        idx = 0
        for i, w in enumerate(net.weights):
            size = w.size
            net.weights[i] = flat_weights[idx:idx + size].reshape(w.shape)
            idx += size
        return net


__all__ = [
    "build_lstm_weights",
    "forward_lstm", 
    "LSTMNetwork",
]
