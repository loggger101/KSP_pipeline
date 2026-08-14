"""Training analytics — architecture evolution tracking, learning curves, comparisons.

This module provides tools for analyzing and visualizing neuroevolution training:

1. **Architecture Evolution Tracking** — Monitor how network topologies change over generations
   - Node count distribution across population  
   - Connection density trends
   - Most common architectural patterns
   
2. **Learning Curve Analysis** — Statistical analysis of fitness progression
   - Moving averages, trend detection
   - Convergence rate estimation
   - Early stopping criteria

3. **Agent Comparison** — Side-by-side evaluation of trained agents
   - Performance on multiple scenarios
   - Architecture comparison metrics

Usage:
    >>> from ksp_neuro.ne_engine.analytics import TrainingAnalytics
    
    # Initialize tracking
    analytics = TrainingAnalytics(save_dir="./analytics")
    
    # Record per-generation data
    analytics.record_generation(
        generation=42,
        fitness_scores=[5000.0, 4800.0, ...],
        network_stats={"avg_nodes": 15, "avg_connections": 23},
        best_agent_architecture={...},
    )
    
    # Get analysis results
    convergence = analytics.detect_convergence(threshold=0.01)
    learning_rate = analytics.compute_learning_rate()
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GenerationRecord:
    """Data recorded for a single training generation."""
    
    generation: int
    fitness_scores: List[float] = field(default_factory=list)
    network_stats: Dict[str, float] = field(default_factory=dict)
    num_species: int = 0          # for NEAT tracking
    avg_nodes: float = 0.0        # average node count across population
    avg_connections: float = 0.0  # average connection count
    best_fitness: float = -1e9
    mean_fitness: float = 0.0
    std_fitness: float = 0.0
    worst_fitness: float = 1e9
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "fitness_scores": self.fitness_scores,
            "network_stats": self.network_stats,
            "num_species": self.num_species,
            "avg_nodes": self.avg_nodes,
            "avg_connections": self.avg_connections,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "std_fitness": self.std_fitness,
            "worst_fitness": self.worst_fitness,
        }


@dataclass 
class LearningCurve:
    """Results from learning curve analysis."""
    
    best_fitness_history: List[float] = field(default_factory=list)
    mean_fitness_history: List[float] = field(default_factory=list)
    
    # Trend detection results
    trend_slope: float = 0.0          # fitness improvement per generation (last N gens)
    is_converged: bool = False        # whether training has plateaued
    convergence_generation: int = -1  # first gen where improvement < threshold
    
    # Statistics
    avg_improvement_rate: float = 0.0   # mean delta fitness per generation
    max_fitness: float = 0.0
    final_fitness: float = 0.0


# ---------------------------------------------------------------------------
# Training analytics class
# ---------------------------------------------------------------------------

class TrainingAnalytics:
    """Track and analyze neuroevolution training progress.

    Parameters
    ----------
    save_dir : str | Path
        Directory to save analysis data (JSON files). Auto-created if needed.
    window_size : int
        Window size for moving averages and trend detection.
    
    Usage:
        >>> analytics = TrainingAnalytics(save_dir="./analytics")
        
        # During training loop, record each generation:
        analytics.record_generation(
            generation=gen,
            fitness_scores=scores,
            network_stats={"avg_nodes": 15.2},
        )
        
        # After training, analyze results:
        curve = analytics.get_learning_curve()
        converged_at = analytics.detect_convergence(threshold=0.01)
    """

    def __init__(self, save_dir: str = "./analytics", window_size: int = 50):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        
        # Stored records (in-memory)
        self._records: List[GenerationRecord] = []
        
        # Load existing data if available
        self._load()

    def record_generation(
        self,
        generation: int,
        fitness_scores: List[float],
        network_stats: Dict[str, float] = None,
        num_species: int = 0,
        best_agent_architecture: Dict[str, Any] = None,
    ) -> GenerationRecord:
        """Record data for a single generation.

        Parameters
        ----------
        generation : int
            Current generation number (1-indexed).
        fitness_scores : list[float]
            Fitness scores for all agents in this generation.
        network_stats : dict | None
            Network statistics (avg_nodes, avg_connections, etc.).
        num_species : int
            Number of species (for NEAT tracking).
        best_agent_architecture : dict | None
            Architecture details of the best agent.

        Returns
        -------
        record : GenerationRecord
            The recorded data for this generation.
        """
        record = GenerationRecord(
            generation=generation,
            fitness_scores=list(fitness_scores),
            network_stats=dict(network_stats or {}),
            num_species=num_species,
            avg_nodes=network_stats.get("avg_nodes", 0) if network_stats else 0,
            avg_connections=network_stats.get("avg_connections", 0) if network_stats else 0,
        )

        # Compute summary statistics
        if fitness_scores:
            record.best_fitness = max(fitness_scores)
            record.mean_fitness = float(np.mean(fitness_scores))
            record.std_fitness = float(np.std(fitness_scores))
            record.worst_fitness = min(fitness_scores)

        self._records.append(record)
        
        # Save to disk incrementally
        self._save()
        
        return record

    def get_learning_curve(self, window_size: int = None) -> LearningCurve:
        """Analyze fitness progression and detect convergence.

        Parameters
        ----------
        window_size : int | None
            Window for moving average (uses configured default if None).

        Returns
        -------
        curve : LearningCurve
            Analysis results including trends, convergence detection.
        """
        ws = window_size or self.window_size
        
        # Extract fitness histories
        best_history = [r.best_fitness for r in self._records]
        mean_history = [r.mean_fitness for r in self._records]

        curve = LearningCurve(
            best_fitness_history=best_history,
            mean_fitness_history=mean_history,
        )

        if len(best_history) < 2:
            return curve

        # Trend detection on last N generations
        recent_best = best_history[-ws:] if ws <= len(best_history) else best_history
        if len(recent_best) >= 2:
            x = np.arange(len(recent_best))
            y = np.array(recent_best)
            
            # Linear regression for trend slope
            x_mean = x.mean()
            y_mean = y.mean()
            numerator = float(np.sum((x - x_mean) * (y - y_mean)))
            denominator = float(np.sum((x - x_mean) ** 2))
            
            curve.trend_slope = numerator / max(denominator, 1e-6)

        # Convergence detection: improvement < threshold for last N gens
        if len(best_history) >= ws + 1:
            recent_improvements = []
            for i in range(-ws, -1):
                prev_best = best_history[i - 1] if i > -len(best_history) else best_history[0]
                improvement = abs(best_history[i] - prev_best) / max(abs(prev_best), 1e-6)
                recent_improvements.append(improvement)

            avg_improvement = np.mean(recent_improvements) if recent_improvements else float('inf')
            curve.avg_improvement_rate = float(avg_improvement)
            
            # Consider converged if improvement < 1% over window
            if avg_improvement < 0.01:
                curve.is_converged = True
                curve.convergence_generation = self._records[-ws].generation

        # Store max and final fitness
        if best_history:
            curve.max_fitness = max(best_history)
            curve.final_fitness = best_history[-1]

        return curve

    def detect_convergence(self, threshold: float = 0.01, window_size: int = None) -> Optional[int]:
        """Detect when training has converged (fitness improvement < threshold).

        Parameters
        ----------
        threshold : float
            Relative improvement threshold (e.g., 0.01 = 1%).
        window_size : int | None
            Window for computing average improvement.

        Returns
        -------
        convergence_generation : int or None
            First generation where convergence was detected, or None if not yet converged.
        """
        curve = self.get_learning_curve(window_size)
        return curve.convergence_generation if curve.is_converged else None

    def compare_agents(self, agent_a_arch: Dict[str, Any], agent_b_arch: Dict[str, Any]) -> Dict[str, float]:
        """Compare two agent architectures across multiple metrics.

        Parameters
        ----------
        agent_a_arch : dict
            Architecture of agent A (from best_agent_architecture record).
        agent_b_arch : dict  
            Architecture of agent B.

        Returns
        -------
        comparison : dict with keys:
            - "nodes_diff": node count difference
            - "connections_diff": connection count difference
            - "efficiency_score": architecture efficiency metric
        """
        nodes_a = agent_a_arch.get("num_nodes", 0) if agent_a_arch else 0
        nodes_b = agent_b_arch.get("num_nodes", 0) if agent_b_arch else 0
        
        conn_a = agent_a_arch.get("num_connections", 0) if agent_a_arch else 0
        conn_b = agent_b_arch.get("num_connections", 0) if agent_b_arch else 0

        # Efficiency: more connections per node is generally better (up to a point)
        efficiency_a = conn_a / max(nodes_a, 1)
        efficiency_b = conn_b / max(nodes_b, 1)

        return {
            "nodes_diff": nodes_a - nodes_b,
            "connections_diff": conn_a - conn_b,
            "efficiency_a": efficiency_a,
            "efficiency_b": efficiency_b,
            "efficiency_diff": efficiency_a - efficiency_b,
        }

    def get_population_diversity(self) -> float:
        """Compute population diversity metric (std / mean of fitness)."""
        if not self._records or len(self._records[-1].fitness_scores) < 2:
            return 0.0
        
        scores = self._records[-1].fitness_scores
        mean_score = np.mean(scores)
        
        if abs(mean_score) > 1e-6:
            return float(np.std(scores) / max(abs(mean_score), 1e-6))
        return float(np.std(scores))

    def get_best_generations(self, n: int = 5) -> List[GenerationRecord]:
        """Get the top N generations by best fitness."""
        sorted_records = sorted(self._records, key=lambda r: r.best_fitness, reverse=True)
        return sorted_records[:n]

    def _save(self):
        """Save current records to disk as JSON."""
        data = {
            "records": [r.to_dict() for r in self._records],
            "total_generations": len(self._records),
        }
        
        save_path = self.save_dir / "training_history.json"
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load existing records from disk."""
        load_path = self.save_dir / "training_history.json"
        
        if not load_path.exists():
            return
            
        try:
            with open(load_path, "r") as f:
                data = json.load(f)

            for record_data in data.get("records", []):
                record = GenerationRecord(
                    generation=record_data["generation"],
                    fitness_scores=record_data.get("fitness_scores", []),
                    network_stats=record_data.get("network_stats", {}),
                    num_species=record_data.get("num_species", 0),
                )

                # Compute summary stats if not stored
                scores = record.fitness_scores
                if scores:
                    record.best_fitness = max(scores)
                    record.mean_fitness = float(np.mean(scores))
                    record.std_fitness = float(np.std(scores))
                    record.worst_fitness = min(scores)

                self._records.append(record)
        except (json.JSONDecodeError, KeyError):
            pass  # Corrupted file — start fresh


# Suppress unused import warning for numpy (used conditionally inside methods)
import numpy as np  # noqa: E402

__all__ = [
    "TrainingAnalytics",
    "GenerationRecord",
    "LearningCurve",
]
