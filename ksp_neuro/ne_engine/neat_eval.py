"""NEAT genome evaluation — converts evolved topology to forward pass.

This module provides the bridge between NEAT genomes (which encode both
network architecture and connection weights) and the orbital simulator.
Each genome is converted into a weight dictionary that describes:
- Which nodes exist (input, hidden, output)
- Which connections are active and their weights  
- Activation functions per node

The forward pass computes through this graph dynamically based on topology,
enabling true NEAT where both architecture AND weights evolve.

Usage:
    >>> from ksp_neuro.ne_engine.neat_eval import evaluate_genome
    >>> fitness = evaluate_genome(genome, env_class=OrbitSimulator, ...)
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict


def evaluate_genome(
    genome: "Genome",  # NEAT Genome class
    env_class=None,
    config=None,
    n_episodes: int = 1,
) -> float:
    """Evaluate a single NEAT genome through the environment.

    Converts the genome's topology into a forward-pass graph, runs it on 
    observation inputs from the simulator, and accumulates reward as fitness.

    Parameters
    ----------
    genome : Genome
        A NEAT genome with nodes + connections.
    env_class : type | None
        Environment class (e.g., OrbitSimulator).
    config : Config | None
        Configuration for environment construction.
    n_episodes : int
        Number of episodes to average over.

    Returns
    -------
    fitness : float
        Average reward across episodes.
    """
    from ksp_neuro.sim_env.orbit_sim import OrbitSimulator
    
    weights_dict = genome_to_weights_dict(genome)
    
    total_reward = 0.0
    max_steps_survived = 0

    for ep in range(n_episodes):
        env = env_class(config=config if hasattr(config, 'sim_time_step') else None, seed=ep * 100 + id(genome))
        
        obs = env.reset()
        episode_reward = 0.0
        steps_survived = 0

        for step in range(3600):  # max steps per episode
            x = obs.reshape(-1, 1).astype(np.float64)
            
            # Forward pass through evolved network topology
            raw_output = forward_through_genome(weights_dict, x)
            action = _output_activation(raw_output.flatten())

            obs, reward, done, info = env.step(action)
            episode_reward += float(reward)
            steps_survived += 1

            if done:
                break

        env.close()
        total_reward += episode_reward / n_episodes
        max_steps_survived = max(max_steps_survived, steps_survived)
        
    genome.episode_count = max_steps_survived
    
    return total_reward


def genome_to_weights_dict(genome: "Genome") -> Dict[str, Any]:
    """Convert a NEAT Genome into a weights dictionary for forward pass.

    Returns dict with keys:
        - 'nodes': {node_id: NodeGene}   — all nodes including inputs/outputs
        - 'connections': [ConnectionGene] — enabled connections only  
        - 'input_nodes': [int]            — input node IDs (1..input_size)
        - 'output_nodes': [int]           — output node IDs

    The forward pass uses this dictionary to dynamically compute through 
    the network topology, regardless of structure.
    """
    from ksp_neuro.ne_engine.neat import Genome  # avoid circular import
    
    if not hasattr(genome, 'nodes'):
        raise ValueError("Genome must have 'nodes' and 'connections' attributes")

    nodes = {n.node_id: n for n in genome.nodes}
    
    # Identify input/output nodes (by convention)
    node_ids = sorted(nodes.keys())
    if len(node_ids) < 2:
        raise ValueError(f"Genome must have at least 2 nodes, has {len(node_ids)}")
    
    # Assume first N nodes are inputs, last M are outputs
    # (this convention is set during NEAT initialization)
    input_nodes = list(range(1, min(len(nodes), 17))) if len(nodes) >= 16 else node_ids[:max(4, len(node_ids)//2)]
    
    # Output nodes: those with no outgoing connections to other outputs
    output_candidates = [nid for nid in node_ids 
                         if not any(c.out_node == nid and c.out_node > max(input_nodes) 
                                   for c in genome.connections if hasattr(genome, 'connections'))]
    output_nodes = output_candidates[-8:] if len(output_candidates) >= 8 else node_ids[-8:]

    return {
        "nodes": nodes,
        "connections": [c for c in (genome.connections or [])],
        "input_nodes": input_nodes,
        "output_nodes": output_nodes,
    }


def forward_through_genome(
    weights_dict: Dict[str, Any], 
    x: np.ndarray  # shape (n_input,) or (n_input, 1)
) -> np.ndarray:  # shape (n_output,)
    """Forward pass through a NEAT-encoded network.

    Dynamically computes activation values for each node in topological order,
    regardless of the specific architecture encoded in the genome.

    Parameters
    ----------
    weights_dict : dict
        Output from genome_to_weights_dict().
    x : np.ndarray
        Input vector (observations).

    Returns
    -------
    output : np.ndarray  shape (n_output,)
        Raw output values (tanh squashed for control outputs).
    """
    nodes = weights_dict["nodes"]
    connections = [c for c in weights_dict["connections"] if c.enabled]
    input_nodes = weights_dict["input_nodes"]
    
    # Initialize node activations from inputs
    activations = {}
    
    for i, nid in enumerate(input_nodes):
        val = float(x[i]) if x.ndim == 1 else float(x[i, 0])
        activations[nid] = val

    # Topological sort: process nodes with all inputs available first
    processed = set()
    max_iterations = len(nodes) + 5  # safety limit
    
    for _ in range(max_iterations):
        changed = False
        
        for node_id, node in nodes.items():
            if node_id in processed or node_id in input_nodes:
                continue
            
            # Check all inputs to this node are computed
            incoming = [c for c in connections 
                       if c.out_node == node_id]  # connections feeding INTO this node
            if not incoming:
                continue
                
            all_inputs_ready = True
            weighted_sum = 0.0
            
            for conn in incoming:
                src_id = conn.in_node
                if src_id not in activations:
                    all_inputs_ready = False
                    break
                weighted_sum += activations[src_id] * conn.weight
            
            if all_inputs_ready and all_inputs_ready is True:
                # Apply activation function + bias
                raw = weighted_sum + node.bias
                
                # Activation by name (default relu)
                act_name = getattr(node, 'activation', 'relu') or 'relu'
                
                if act_name == "tanh":
                    activations[node_id] = float(np.tanh(raw))
                elif act_name == "sigmoid":
                    activations[node_id] = float(1 / (1 + np.exp(-np.clip(raw, -500, 500))))
                elif act_name == "elu":
                    activations[node_id] = float(np.where(raw > 0, raw, np.exp(raw) - 1))
                else:  # relu or default
                    activations[node_id] = float(max(0.0, raw))
                
                processed.add(node_id)
                changed = True
        
        if not changed:
            break

    # Collect output values
    output_nodes = weights_dict.get("output_nodes", [])
    
    if not output_nodes:
        raise ValueError("No output nodes found in genome")
    
    outputs = []
    for nid in output_nodes[:8]:  # max 8 control outputs
        val = activations.get(nid, 0.0)
        # Output layer always uses tanh for control values [-1, +1]
        outputs.append(float(np.tanh(val)))

    return np.array(outputs, dtype=np.float64)


def _output_activation(raw: np.ndarray) -> np.ndarray:
    """Map raw network output to valid action space [-1, 1]."""
    # tanh squashes to (-1, 1), directly mapping to our action range
    return np.tanh(raw)


__all__ = ["evaluate_genome", "genome_to_weights_dict", "forward_through_genome"]
