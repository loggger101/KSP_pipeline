"""Multi-objective optimization for KSP neuroevolution.

This module implements NSGA-II style non-dominated sorting and crowding 
distance calculation for multi-objective fitness evaluation. Instead of a 
single scalar reward, agents are evaluated on multiple objectives simultaneously:

    1. Orbital height (apoapsis altitude)
    2. Orbital circularity (deviation from circular orbit velocity)  
    3. Fuel efficiency (fuel remaining / fuel used)
    4. Stage completion bonus
    5. Stability penalty (oscillation detection)

Agents are ranked by Pareto dominance: agent A dominates B if A is 
at least as good as B on ALL objectives and strictly better on at least one.

This produces a diverse population exploring the trade-off surface between
competing goals (e.g., high orbit vs fuel efficiency).

Usage:
    >>> from ksp_neuro.ne_engine.multi_obj import MultiObjectiveFitness
    
    # Define objective weights (higher = more important)
    fitness_fn = MultiObjectiveFitness(
        objectives=["apoapsis", "circularity", "fuel_efficiency"],
        weights=[3.0, 2.0, 1.5],      # relative importance
        normalize=True,                 # auto-normalize per generation
        crowding_distance=True,         # maintain diversity via NSGA-II
    )
    
    # Evaluate a population: returns fitness scores + Pareto rank info
    ranks, distances = fitness_fn.evaluate(population_scores)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ObjectiveConfig:
    """Configuration for a single optimization objective."""
    
    name: str                              # human-readable name
    maximize: bool = True                  # higher is better (False if lower is better)
    weight: float = 1.0                    # relative importance in multi-objective
    target_range: Optional[tuple] = None   # desired range [low, high] for bonus


@dataclass 
class ParetoRanking:
    """Results of non-dominated sorting."""
    
    ranks: np.ndarray                      # rank per agent (lower = better)
    crowding_distances: np.ndarray         # diversity metric per agent
    front_0_indices: List[int]             # best front (Pareto-optimal set)
    num_fronts: int                        # total number of non-dominated fronts


# ---------------------------------------------------------------------------
# Multi-objective fitness evaluator
# ---------------------------------------------------------------------------

class MultiObjectiveFitness:
    """NSGA-II style multi-objective fitness evaluation for NE training.

    Parameters
    ----------
    objectives : list[str] | None
        Objective names to track. Default auto-detected from simulation info.
    weights : list[float] | None
        Relative importance of each objective. Higher weight = more important.
    normalize : bool
        Auto-normalize objectives per generation for fair comparison.
    crowding_distance : bool
        Compute NSGA-II crowding distance for diversity maintenance.
    
    Available objectives:
        - "apoapsis"          : Maximum apoapsis altitude achieved [m]
        - "periapsis"         : Minimum periapsis altitude maintained [m]  
        - "circularity"       : How circular the orbit is (1 = perfect circle)
        - "fuel_efficiency"   : Fuel remaining / initial fuel ratio
        - "stage_completion"  : Number of stages completed
        - "stability"         : Low oscillation in attitude/velocity
    """

    AVAILABLE_OBJECTIVES = {
        "apoapsis": ObjectiveConfig("Max Apoapsis Altitude", maximize=True, weight=3.0),
        "periapsis": ObjectiveConfig("Min Periapsis Altitude", maximize=True, weight=1.5),
        "circularity": ObjectiveConfig("Orbital Circularity", maximize=True, weight=2.0),
        "fuel_efficiency": ObjectiveConfig("Fuel Efficiency", maximize=True, weight=1.5),
        "stage_completion": ObjectiveConfig("Stage Completion", maximize=True, weight=1.0),
        "stability": ObjectiveConfig("Flight Stability", maximize=True, weight=1.0),
    }

    def __init__(
        self,
        objectives: List[str] = None,
        weights: List[float] = None,
        normalize: bool = True,
        crowding_distance: bool = False,
    ):
        self.objectives = objectives or ["apoapsis", "circularity", "fuel_efficiency"]
        self.normalize = normalize
        self.crowding_distance = crowding_distance
        
        # Validate objective names
        for obj in self.objectives:
            if obj not in self.AVAILABLE_OBJECTIVES:
                raise ValueError(f"Unknown objective '{obj}'. Available: {list(self.AVAILABLE_OBJECTIVES.keys())}")

        # Set weights (use defaults from config if not provided)
        default_weights = [self.AVAILABLE_OBJECTIVES[obj].weight for obj in self.objectives]
        self.weights = np.array(weights or default_weights, dtype=np.float64)
        
        # Track normalization stats across generations
        self._min_values: Dict[str, float] = {}
        self._max_values: Dict[str, float] = {}

    def evaluate(
        self, 
        episode_scores: List[float],  # raw total rewards per agent
        info_dicts: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate agents on multiple objectives. Returns Pareto ranks and crowding distances.

        Parameters
        ----------
        episode_scores : list[float]
            Raw total reward per agent (from single-objective evaluation).
        info_dicts : list[dict] | None
            Per-agent simulation info dicts containing objective data.

        Returns
        -------
        ranks : np.ndarray  shape (n_agents,) — Pareto rank (lower = better)
        distances : np.ndarray  shape (n_agents,) — crowding distance for diversity
        """
        n = len(episode_scores)
        
        # Build objective matrix: rows = objectives, cols = agents
        obj_matrix = self._compute_objective_values(episode_scores, info_dicts)
        
        if self.normalize and not all(k in self._min_values for k in self.AVAILABLE_OBJECTIVES):
            # First call: initialize normalization bounds
            pass
        
        if self.normalize:
            obj_matrix = self._normalize(obj_matrix)

        # Multi-objective ranking via NSGA-II style non-dominated sorting
        ranks, distances = self._non_dominated_sort(obj_matrix)
        
        return ranks, distances

    def get_pareto_front(self, episode_scores: List[float], 
                         info_dicts: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """Get indices of agents on the Pareto-optimal front (best trade-offs)."""
        n = len(episode_scores)
        obj_matrix = self._compute_objective_values(episode_scores, info_dicts)
        
        if self.normalize:
            obj_matrix = self._normalize(obj_matrix)
            
        _, distances = self._non_dominated_sort(obj_matrix)
        front_0 = [i for i in range(n) if distances[i] > 0 or i == np.argmax(distances)]
        
        return front_0[:min(10, n)]  # top 10 Pareto-optimal agents

    def _compute_objective_values(self, episode_scores: List[float], 
                                   info_dicts: Optional[List[Dict[str, Any]]]) -> np.ndarray:
        """Compute raw objective values from simulation data."""
        import numpy as np
        
        n = len(episode_scores)
        
        # Default to using the total reward if no detailed info available
        if not info_dicts or all(d is None for d in info_dicts):
            obj_matrix = np.array([episode_scores]).T  # single objective: total reward
            
            # Map to named objectives (all equal to total score)
            result = np.zeros((len(self.objectives), n))
            for i, _ in enumerate(result):
                result[i] = episode_scores
            return result

        obj_matrix = np.zeros((len(self.objectives), n), dtype=np.float64)
        
        for j, info in enumerate(info_dicts or [None] * n):
            if info is None:
                continue
                
            # Apoapsis altitude (maximize)
            apo_alt = info.get("apoapsis_alt", 0.0) or info.get("max_altitude", 0.0)
            
            # Fuel efficiency (maximize) — higher fuel remaining ratio is better
            fuel_ratio = info.get("fuel_remaining", 1.0)
            if isinstance(fuel_ratio, dict):
                fuel_ratio = fuel_ratio.get('ratio', 1.0)
                
            # Stage completion (maximize)
            stages = info.get("stage_changes", []) or info.get("stages_completed", 0)
            
            for i, obj_name in enumerate(self.objectives):
                if obj_name == "apoapsis":
                    obj_matrix[i, j] = float(apo_alt) / 1e6  # normalize to ~[0, 2]
                elif obj_name == "periapsis":
                    peri_alt = info.get("periapsis_alt", 0.0) or 0.0
                    obj_matrix[i, j] = float(peri_alt) / 1e6
                elif obj_name == "circularity":
                    # Approximate circularity from apo/peri ratio
                    apo = max(float(apo_alt), 1e3)
                    peri = max(float(info.get("periapsis_alt", 0.0)) or 1e3, 1e3)
                    obj_matrix[i, j] = min(peri / apo, 1.0)
                elif obj_name == "fuel_efficiency":
                    obj_matrix[i, j] = float(fuel_ratio) if isinstance(fuel_ratio, (int, float)) else 0.5
                elif obj_name == "stage_completion":
                    obj_matrix[i, j] = min(float(len(stages)) / 3.0, 1.0) if stages else 0.0
                elif obj_name == "stability":
                    # Estimate from velocity oscillation (higher = more stable)
                    vel = info.get("velocities", []) or []
                    if len(vel) > 5:
                        vel_std = np.std(vel[-10:]) if hasattr(np, 'std') else 0.0
                        obj_matrix[i, j] = max(0.0, 1.0 - vel_std / 100.0)
                    else:
                        obj_matrix[i, j] = 0.5

        return obj_matrix

    def _normalize(self, obj_matrix: np.ndarray) -> np.ndarray:
        """Normalize objective values per-generation for fair comparison."""
        n_obj, n_agents = obj_matrix.shape
        
        if not self._min_values or len(self._min_values) < n_obj:
            # Initialize with current min/max
            self._min_values = {f"obj_{i}": float(np.min(obj_matrix[i])) for i in range(n_obj)}
            self._max_values = {f"obj_{i}": float(np.max(obj_matrix[i])) for i in range(n_obj)}

        result = obj_matrix.copy()
        for i in range(n_obj):
            key_min, key_max = f"obj_{i}", f"obj_{i}"
            min_val = self._min_values.get(key_min, 0.0)
            max_val = self._max_values.get(key_max, 1.0)
            
            if abs(max_val - min_val) > 1e-6:
                result[i] = (obj_matrix[i] - min_val) / (max_val - min_val)
            else:
                result[i] = 0.5  # all same value — neutral score
        
        return result

    def _non_dominated_sort(self, obj_matrix: np.ndarray) -> ParetoRanking:
        """NSGA-II non-dominated sorting algorithm.

        Parameters
        ----------
        obj_matrix : np.ndarray  shape (n_objectives, n_agents)

        Returns
        -------
        ranks : np.ndarray — rank per agent (1 = best front)
        distances : np.ndarray — crowding distance for diversity
        """
        import numpy as np
        
        n_obj, n = obj_matrix.shape
        domination_count = np.zeros(n, dtype=int)    # how many agents dominate this one
        dominated_set: List[List[int]] = [[] for _ in range(n)]  # agents dominated by i
        ranks = np.zeros(n, dtype=int)               # front rank (1-based)
        
        # Find first front (agents not dominated by anyone)
        current_front = []
        
        for i in range(n):
            is_dominated = False
            for j in range(n):
                if i == j:
                    continue
                    
                # Check if agent j dominates agent i
                j_better = np.sum(obj_matrix[:, j] >= obj_matrix[:, i]) == n_obj
                i_better = np.sum(obj_matrix[:, i] >= obj_matrix[:, j]) == n_obj
                
                if j_better and not i_better:
                    is_dominated = True
                    dominated_set[i].append(j)
                elif i_better and not j_better:
                    domination_count[j] += 1
            
            if not is_dominated:
                current_front.append(i)

        # Assign ranks and build subsequent fronts
        front_num = 1
        all_ranks = {}
        
        while current_front:
            for agent in current_front:
                ranks[agent] = front_num
                all_ranks.setdefault(front_num, []).append(agent)
            
            next_front = []
            for agent in current_front:
                for dominated_agent in dominated_set[agent]:
                    domination_count[dominated_agent] -= 1
                    if domination_count[dominated_agent] == 0:
                        next_front.append(dominated_agent)
            
            front_num += 1
            current_front = next_front

        # Compute crowding distances for diversity maintenance
        distances = np.zeros(n, dtype=np.float64)
        
        for rank_val in sorted(all_ranks.keys()):
            indices = all_ranks[rank_val]
            if len(indices) <= 2:
                # All agents on this front get infinite distance (keep them all)
                distances[indices] = float('inf')
                continue

            # Compute crowding distance per objective
            for obj_idx in range(n_obj):
                sorted_indices = sorted(indices, key=lambda i: obj_matrix[obj_idx, i])
                
                # Boundary agents get infinite distance
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                
                obj_range = max(obj_matrix[obj_idx] - min(obj_matrix[obj_idx]), 1e-6)
                
                for k in range(1, len(sorted_indices) - 1):
                    prev_obj = obj_matrix[obj_idx, sorted_indices[k-1]]
                    next_obj = obj_matrix[obj_idx, sorted_indices[k+1]]
                    distances[sorted_indices[k]] += (next_obj - prev_obj) / max(obj_range, 1e-6)

        return ranks, distances


# ---------------------------------------------------------------------------
# Weighted scalarization fallback (single-objective from multi-objective data)
# ---------------------------------------------------------------------------

def weighted_scalarize(
    obj_matrix: np.ndarray,
    weights: List[float] = None,
) -> np.ndarray:
    """Convert multi-objective scores to single scalar via weighted sum.

    Parameters
    ----------
    obj_matrix : np.ndarray  shape (n_objectives, n_agents)
        Objective values per agent.
    weights : list[float] | None
        Weights for each objective. Auto-normalized if not provided.

    Returns
    -------
    scores : np.ndarray  shape (n_agents,) — scalar fitness per agent
    """
    import numpy as np
    
    n_obj, n = obj_matrix.shape
    w = np.array(weights or [1.0] * n_obj)
    
    # Normalize weights to sum to 1
    w = w / max(w.sum(), 1e-6)
    
    # Weighted sum (already normalized objectives)
    scores = obj_matrix.T @ w
    
    return scores


__all__ = [
    "MultiObjectiveFitness",
    "ParetoRanking", 
    "ObjectiveConfig",
    "weighted_scalarize",
]
