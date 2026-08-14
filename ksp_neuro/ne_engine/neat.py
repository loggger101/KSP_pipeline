"""Alternative NE engine: NEAT (NeuroEvolution of Augmenting Topologies).

This module provides a simplified NEAT implementation that evolves both
network weights AND topological structure simultaneously. Unlike the fixed-
architecture GA, NEAT starts with minimal networks and adds complexity
only when beneficial — similar to how Miikkulainen's original NEAT works
for the NERO game.

Key features:
- Starts with no connections, discovers architecture through evolution
- Speciation protects novel structures from being overwhelmed
- Historical marking prevents crossover of non-homologous networks
- Invention operations (add node/edge) drive structural innovation

Reference: Stanley, K.O. & Miikkulainen, R. (2002). 
"Evolving Neural Networks through Augmenting Topologies"
"""

from __future__ import annotations

import copy
import hashlib
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(order=True)
class Genome:
    """A single NEAT genome encoding a neural network.

    Attributes
    ----------
    key : str
        Unique identifier for this genome (hash of innovation numbers).
    fitness : float
        Current generation fitness score.
    best_fitness : float
        Best-ever fitness (for tracking convergence).
    disjoint : int
        Number of non-matching genes when compared to another genome.
    excess : int
        Number of genes ahead of the comparison point.
    compatibility_distance : float
        Distance metric for speciation (smaller = more similar).
    
    innovation_table : dict
        Global mapping from gene type → unique innovation number.
    """

    # Gene identifiers
    key: str = ""
    
    # Fitness tracking
    fitness: float = 0.0
    best_fitness: float = -1e9
    
    # Compatibility metrics (computed during speciation)
    disjoint: int = 0
    excess: int = 0
    compatibility_distance: float = 0.0

    # Genome-specific data
    nodes: List[NodeGene] = field(default_factory=list)
    connections: List[ConnectionGene] = field(default_factory=list)
    
    # Metadata for debugging/tracking
    generation: int = 0
    species_id: Optional[int] = None
    
    def encode(self) -> bytes:
        """Serialize genome to bytes."""
        return pickle.dumps({
            "key": self.key,
            "fitness": self.fitness,
            "best_fitness": self.best_fitness,
            "disjoint": self.disjoint,
            "excess": self.excess,
            "compatibility_distance": self.compatibility_distance,
            "nodes": [n.__dict__ for n in self.nodes],
            "connections": [c.__dict__ for c in self.connections],
            "generation": self.generation,
            "species_id": self.species_id,
        })

    @classmethod
    def decode(cls, data: bytes) -> Genome:
        """Deserialize genome from bytes."""
        obj = pickle.loads(data)
        g = cls()
        for attr in ["key", "fitness", "best_fitness", "disjoint", 
                      "excess", "compatibility_distance"]:
            setattr(g, attr, obj[attr])
        g.nodes = [NodeGene(**n) for n in obj["nodes"]]
        g.connections = [ConnectionGene(**c) for c in obj["connections"]]
        g.generation = obj.get("generation", 0)
        g.species_id = obj.get("species_id")
        return g


@dataclass(order=True)
class NodeGene:
    """Represents a single neuron node.

    Attributes
    ----------
    node_id : int
        Unique identifier for this node.
    in_degree : int
        Number of incoming connections (how many inputs this node receives).
    activation : str
        Activation function ('relu', 'tanh', 'sigmoid', 'elu').
    bias : float
        Node's internal bias value.
    """

    node_id: int = 0
    in_degree: int = 0
    activation: str = "relu"
    bias: float = 0.0


@dataclass(order=True)
class ConnectionGene:
    """Represents a single connection between two nodes.

    Attributes
    ----------
    key : int
        Unique innovation number for this connection type.
    in_node : int
        Source node ID (input).
    out_node : int
        Destination node ID (output).
    weight : float
        Connection weight.
    enabled : bool
        Whether this connection is active.
    """

    key: int = 0
    in_node: int = 0
    out_node: int = 0
    weight: float = 0.0
    enabled: bool = True


# ---------------------------------------------------------------------------
# NEAT Engine
# ---------------------------------------------------------------------------

class NEATEngine:
    """Simplified NEAT (NeuroEvolution of Augmenting Topologies) engine.

    Evolves both network architecture and weights simultaneously through
    genetic algorithm operations including speciation, crossover with 
    historical marking, and structural mutation operators.

    Parameters
    ----------
    input_size : int
        Number of input features.
    output_size : int
        Number of control outputs.
    population_size : int
    elite_ratio : float
    mutation_rate : float  per-gene probability
    mutation_strength : float  std-dev for Gaussian weights
    crossover_rate : float
    compatibility_threshold : float  speciation distance threshold
    """

    def __init__(
        self,
        input_size: int = 16,
        output_size: int = 8,
        population_size: int = 200,
        elite_ratio: float = 0.1,
        mutation_rate: float = 0.3,
        mutation_strength: float = 0.1,
        crossover_rate: float = 0.9,
        compatibility_threshold: float = 3.0,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.compatibility_threshold = compatibility_threshold

        # Global innovation counter (shared across all genomes)
        self._innovation_counter = 0
        self._rng = random.Random(42)

        self._population: List[Genome] = []
        self._species: Dict[int, List[Genome]] = {}  # species_id → [genomes]
        self._generation = 0

        # History tracking
        self.history: Dict[str, List[float]] = {
            "best_fitness": [],
            "mean_fitness": [],
            "worst_fitness": [],
            "std_fitness": [],
            "num_species": [],
            "avg_nodes": [],
            "avg_connections": [],
        }

    def initialize_population(
        self, 
        initial_nodes: int = 2,
        initial_weight_range: float = 0.5,
    ) -> List[Genome]:
        """Create initial population of minimal genomes (no connections).

        Parameters
        ----------
        initial_nodes : int
            Number of input/output nodes to create initially.
            Total = input_size + output_size + hidden nodes.
        initial_weight_range : float
            Range for random weight initialization [-range, +range].
        """
        self._population = []

        # Create input and output node IDs (fixed)
        input_nodes = list(range(1, self.input_size + 1))     # 1..input_size
        output_nodes = list(range(self.input_size + 1, 
                                   self.input_size + self.output_size + 1))

        for _ in range(self.population_size):
            genome = Genome()
            
            # Create input nodes (no connections yet)
            for nid in input_nodes:
                node = NodeGene(node_id=nid, in_degree=0, activation="relu")
                genome.nodes.append(node)
            
            # Create output nodes
            for nid in output_nodes:
                node = NodeGene(node_id=nid, in_degree=0, activation="tanh")
                genome.nodes.append(node)

            genome.key = self._compute_key(genome)
            genome.fitness = -1e9
            genome.best_fitness = -1e9
            genome.generation = 0
            
            self._population.append(genome)

        return list(self._population)

    def evaluate_population(
        self,
        eval_fn,  # callable: (genome_weights_dict) -> float fitness
        **eval_kwargs,
    ) -> List[float]:
        """Evaluate all genomes in the population.

        Parameters
        ----------
        eval_fn : callable
            Function that takes a genome's weight dict and returns fitness score.
            Signature: eval_fn(weights_dict, **kwargs) -> float
        **eval_kwargs
            Additional keyword arguments passed to eval_fn.

        Returns
        -------
        fitness_scores : list[float]
        """
        scores = []

        for genome in self._population:
            # Convert genome to weight dict (node_id → weights, etc.)
            weights_dict = self._genome_to_weights(genome)
            fitness = eval_fn(weights_dict, **eval_kwargs)
            
            genome.fitness = fitness
            genome.best_fitness = max(genome.best_fitness, fitness)
            scores.append(fitness)

        # Update history
        self.history["best_fitness"].append(max(scores))
        self.history["mean_fitness"].append(float(np.mean(scores)))
        self.history["worst_fitness"].append(min(scores))
        self.history["std_fitness"].append(float(np.std(scores)))

        return scores

    def evolve(self) -> List[Genome]:
        """Perform one generation of NEAT evolution.

        Returns
        -------
        new_population : list[Genome]
        """
        import numpy as np  # local import to avoid dependency on numpy for non-NE usage
        
        # Sort by fitness descending within each species
        self._speciate()
        
        n_elites = max(1, int(self.population_size * self.elite_ratio))

        # Create new population from species (weighted by average fitness)
        total_species_fitness = sum(
            np.mean([g.fitness for g in sp]) if sp else 0.0
            for sp in self._species.values()
        )

        new_population: List[Genome] = []
        
        # Elitism: preserve top genomes from each species
        elite_count = 0
        for sp_id, sp_genomes in self._species.items():
            sp_genomes.sort(key=lambda g: g.fitness, reverse=True)
            n_elites_here = max(1, int(n_elites * len(sp_genomes) / len(self._population)))
            new_population.extend(sp_genomes[:n_elites_here])
            elite_count += n_elites_here

        # Generate offspring proportional to species fitness
        for sp_id, sp_genomes in self._species.items():
            avg_fitness = np.mean([g.fitness for g in sp_genomes]) if sp_genomes else 0
            num_offspring = max(1, int((avg_fitness / total_species_fitness) * 
                                       (self.population_size - elite_count)))

            for _ in range(num_offspring):
                parent_a, parent_b = self._tournament_selection(sp_genomes, k=5)
                
                if self._rng.random() < self.crossover_rate:
                    child = self._crossover(parent_a, parent_b)
                else:
                    # Clone with mutation
                    child = copy.deepcopy(self._rng.choice([parent_a, parent_b]))

                # Structural mutations (add node or edge)
                child = self._mutate_structure(child)
                
                # Weight mutations
                child = self._mutate_weights(child)

                child.generation = self._generation + 1
                new_population.append(child)

        self._population = new_population[:self.population_size]
        self._generation += 1
        return list(self._population)

    def get_best_genome(self) -> Genome:
        """Return the genome with highest best-ever fitness."""
        return max(self._population, key=lambda g: g.best_fitness)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save population to disk (pickle)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "generation": self._generation,
            "population": [g.encode() for g in self._population],
            "species": {k: [g.key for g in v] for k, v in self._species.items()},
            "innovation_counter": self._innovation_counter,
            "history": self.history,
        }
        with open(p, "wb") as fh:
            pickle.dump(data, fh)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load population from disk (pickle)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        with open(p, "rb") as fh:
            data = pickle.load(fh)

        self._generation = data["generation"]
        self.history = data["history"]
        self._innovation_counter = data.get("innovation_counter", 0)
        
        self._population = [Genome.decode(enc) for enc in data["population"]]
        # Restore species assignments
        self._speciate()

    def get_history(self) -> Dict[str, List[float]]:
        """Return copy of fitness history."""
        return {k: list(v) for k, v in self.history.items()}

    # ---- private helpers ---------------------------------------------------
    def _genome_to_weights(self, genome: Genome) -> Dict[str, Any]:
        """Convert a NEAT genome to a weight dictionary for evaluation.

        Returns dict mapping node_id → {"weights": [...], "bias": ...}
        plus connection metadata for forward pass computation.
        """
        return {
            "nodes": {n.node_id: n for n in genome.nodes},
            "connections": [c for c in genome.connections if c.enabled],
            "input_nodes": list(range(1, self.input_size + 1)),
            "output_nodes": list(range(self.input_size + 1, 
                                        self.input_size + self.output_size + 1)),
        }

    def _compute_key(self, genome: Genome) -> str:
        """Compute unique key for a genome based on its connection innovation numbers."""
        keys = sorted([c.key for c in genome.connections])
        hash_input = ":".join(str(k) for k in keys) + f":{len(genome.nodes)}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def _speciate(self) -> None:
        """Assign genomes to species based on compatibility distance."""
        self._species = {}  # Clear existing species
        
        for genome in self._population:
            if not self._species:  # First genome starts first species
                sp_id = 0
                new_genome = copy.deepcopy(genome)
                new_genome.species_id = 0
                self._species[sp_id] = [new_genome]
                continue

            # Find closest existing species
            best_sp, best_dist = None, float("inf")
            for sp_id, sp_genomes in self._species.items():
                avg_compat = np.mean([self._compatibility_distance(g1, g2) 
                                      for g1 in [genome] for g2 in sp_genomes])
                if avg_compat < best_dist:
                    best_dist = avg_compat
                    best_sp = sp_id

            # Check if within threshold or create new species
            if best_dist < self.compatibility_threshold and best_sp is not None:
                self._species[best_sp].append(genome)
            else:
                new_sp_id = max(self._species.keys()) + 1 if self._species else 0
                genome.species_id = new_sp_id
                self._species[new_sp_id] = [genome]

        # Update history with speciation stats
        species_sizes = [len(sp) for sp in self._species.values()]
        avg_nodes = np.mean([np.mean([len(g.nodes) for g in sp]) 
                             for sp in self._species.values()])
        avg_conn = np.mean([np.mean([len(g.connections) for g in sp]) 
                            for sp in self._species.values()])

        self.history["num_species"].append(len(species_sizes))
        self.history["avg_nodes"].append(float(avg_nodes))
        self.history["avg_connections"].append(float(avg_conn))

    def _compatibility_distance(self, genome_a: Genome, genome_b: Genome) -> float:
        """Compute compatibility distance between two genomes.

        Distance = c1 * disjoint / N + c2 * excess / N + c3 * avg_weight_diff
        where N is the max number of connections in either genome.
        """
        import numpy as np
        
        # Find matching and non-matching genes by innovation key
        keys_a = {c.key for c in genome_a.connections}
        keys_b = {c.key for c in genome_b.connections}

        disjoint_keys = keys_a.symmetric_difference(keys_b)
        n_disjoint = len(disjoint_keys)
        common_keys = keys_a & keys_b
        
        # Excess genes (those beyond the max of matching)
        max_conn = max(len(genome_a.connections), len(genome_b.connections))
        excess = abs(len(genome_a.connections) - len(genome_b.connections))

        # Average weight difference for matching connections
        w_diffs = []
        conn_dict_a = {c.key: c.weight for c in genome_a.connections}
        conn_dict_b = {c.key: c.weight for c in genome_b.connections}
        
        for key in common_keys:
            w_diffs.append(abs(conn_dict_a[key] - conn_dict_b[key]))

        avg_w_diff = np.mean(w_diffs) if w_diffs else 0.0
        
        # Weighted distance (hyperparameters from Stanley paper)
        c1, c2, c3 = 1.0, 1.0, 0.4
        distance = (c1 * n_disjoint / max(max_conn, 1) + 
                    c2 * excess / max(max_conn, 1) + 
                    c3 * avg_w_diff)

        # Store for reference
        genome_a.disjoint = n_disjoint
        genome_b.disjoint = n_disjoint
        genome_a.excess = excess
        genome_b.excess = excess
        
        return distance

    def _tournament_selection(self, genomes: List[Genome], k: int = 5) -> Tuple[Genome, Genome]:
        """Tournament selection returning two distinct parents."""
        n = len(genomes)
        idx1 = self._rng.sample(range(n), min(k, n))
        idx2 = self._rng.sample(range(n), min(k, n))

        fit1 = [genomes[i].fitness for i in idx1]
        fit2 = [genomes[i].fitness for i in idx2]

        parent_a = genomes[idx1[np.argmax(fit1)]]
        
        # Find parent_b different from parent_a
        candidates = [genomes[i] for i in idx2 if genomes[i] is not parent_a]
        if candidates:
            fit_cands = [g.fitness for g in candidates]
            parent_b = candidates[np.argmax(fit_cands)]
        else:
            parent_b = self._rng.choice(genomes)
            
        return parent_a, parent_b

    def _crossover(self, genome_a: Genome, genome_b: Genome) -> Genome:
        """NEAT crossover with historical marking.

        Matching genes (same innovation number) are inherited from the 
        fitter parent. Excess/disjoint genes come from the dominant parent.
        """
        child = copy.deepcopy(genome_a) if genome_a.fitness >= genome_b.fitness else copy.deepcopy(genome_b)
        
        # Determine dominant parent
        dom_genome = genome_a if genome_a.fitness >= genome_b.fitness else genome_b
        rec_genome = genome_b if genome_a.fitness >= genome_b.fitness else genome_a

        # Build connection lookup for both parents
        dom_conns = {c.key: c for c in dom_genome.connections}
        rec_conns = {c.key: c for c in rec_genome.connections}

        new_connections = []
        
        # Process common genes (same innovation number) — inherit from fitter parent
        common_keys = set(dom_genome.connections.keys()) & set(rec_genome.connections.keys())
        for key in common_keys:
            if key in dom_conns and key in rec_conns:
                conn_a = dom_conns[key]
                conn_b = rec_conns[key]
                
                # Randomly inherit from either parent (50/50)
                if self._rng.random() < 0.5:
                    child_conn = copy.deepcopy(conn_a)
                else:
                    child_conn = copy.deepcopy(conn_b)
                
                new_connections.append(child_conn)

        # Inherit excess/disjoint genes from dominant parent
        dom_keys = set(c.key for c in dom_genome.connections)
        rec_keys = set(c.key for c in rec_genome.connections)
        
        excess_disjoint = dom_keys.symmetric_difference(dom_keys & rec_keys)
        for key in excess_disjoint:
            if key in dom_conns:
                new_connections.append(copy.deepcopy(dom_conns[key]))

        child.connections = sorted(new_connections, key=lambda c: c.key)
        return child

    def _mutate_structure(self, genome: Genome) -> Genome:
        """Apply structural mutation (add node or edge)."""
        import numpy as np
        
        # Mutation probabilities (can be tuned via config)
        prob_add_node = 0.04
        prob_add_edge = 0.15

        # Add new connection to existing nodes
        if self._rng.random() < prob_add_edge:
            genome = self._mutate_add_connection(genome)

        # Add new hidden node (split an existing connection)
        if len(genome.connections) > 0 and self._rng.random() < prob_add_node:
            genome = self._mutate_add_node(genome)

        return genome

    def _mutate_add_edge(self, genome: Genome) -> Genome:
        """Add a new random connection."""
        import numpy as np
        
        # Pick two nodes that don't already have a direct connection
        node_ids = [n.node_id for n in genome.nodes]
        
        for _ in range(10):  # Try up to 10 times
            in_node, out_node = self._rng.sample(node_ids, 2)
            
            # Check no existing connection
            if not any(c.in_node == in_node and c.out_node == out_node 
                       for c in genome.connections):
                # Don't connect output nodes to other outputs
                input_range = range(1, self.input_size + 1)
                output_range = range(self.input_size + 1, 
                                     self.input_size + self.output_size + 1)
                
                if (out_node in output_range and in_node in output_range):
                    continue
                
                # Create new connection with innovation number
                conn = ConnectionGene(
                    key=self._next_innovation(),
                    in_node=in_node,
                    out_node=out_node,
                    weight=float(np.random.normal(0, self.mutation_strength)),
                    enabled=True,
                )
                genome.connections.append(conn)
                return genome

        return genome  # No valid connection found

    def _mutate_add_node(self, genome: Genome) -> Genome:
        """Insert a new hidden node on a random existing connection."""
        import numpy as np
        
        if not genome.connections:
            return genome

        # Pick a random enabled connection to split
        enabled = [c for c in genome.connections if c.enabled]
        if not enabled:
            return genome
            
        target_conn = self._rng.choice(enabled)
        
        # Create new hidden node with unique ID
        max_node_id = max(n.node_id for n in genome.nodes)
        new_node_id = max_node_id + 1
        
        # New node inherits activation from the connection it replaces
        new_node = NodeGene(
            node_id=new_node_id,
            in_degree=0,
            activation="relu",
            bias=float(np.random.normal(0, 0.1)),
        )
        genome.nodes.append(new_node)

        # Update existing connection: source → new_node (weight = 1.0)
        target_conn.weight = 1.0 - target_conn.weight
        
        # Add new connections: in_node → new_node and new_node → out_node
        new_conn_1 = ConnectionGene(
            key=self._next_innovation(),
            in_node=target_conn.in_node,
            out_node=new_node_id,
            weight=1.0,
            enabled=True,
        )
        new_conn_2 = ConnectionGene(
            key=self._next_innovation(),
            in_node=new_node_id,
            out_node=target_conn.out_node,
            weight=float(np.random.normal(0, self.mutation_strength)),
            enabled=True,
        )

        genome.connections.extend([new_conn_1, new_conn_2])
        return genome

    def _mutate_weights(self, genome: Genome) -> Genome:
        """Gaussian mutation on connection weights."""
        import numpy as np
        
        for conn in genome.connections:
            if self._rng.random() < self.mutation_rate:
                # Weight mutation (small perturbation)
                conn.weight += float(np.random.normal(0, self.mutation_strength))
                
                # Bias mutation on output nodes
                node = next((n for n in genome.nodes if n.node_id == conn.out_node), None)
                if node and conn.out_node > self.input_size:  # output node
                    node.bias += float(np.random.normal(0, self.mutation_strength * 0.1))

        return genome

    def _next_innovation(self) -> int:
        """Get next unique innovation number."""
        self._innovation_counter += 1
        return self._innovation_counter


# Suppress unused import warning for numpy (used conditionally inside methods)
import numpy as np  # noqa: E402

__all__ = [
    "NEATEngine",
    "Genome",
    "NodeGene", 
    "ConnectionGene",
]
