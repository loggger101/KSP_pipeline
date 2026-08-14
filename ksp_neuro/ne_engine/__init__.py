"""Neuroevolution engine — population-based genetic algorithms for neural net training."""

from __future__ import annotations

import json
import pickle
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Agent:
    """A single neural-network agent in the population.

    Attributes
    ----------
    weights : list[np.ndarray]
        Layer-by-layer weight matrices (including biases).
        Shape for layer *i*: ``(input_size_i, output_size_i)`` for weights,
        ``(output_size_i,)`` for biases.
    fitness : float = 0.0
        Current generation fitness score.
    best_fitness : float = -np.inf
        Best-ever fitness (for tracking convergence).
    episode_count : int = 0
        How many episodes this agent has survived.
    metadata : dict = {}
        Arbitrary extra data (generation born, parent IDs, etc.).
    """

    weights: List[np.ndarray] = field(default_factory=list)
    fitness: float = 0.0
    best_fitness: float = -np.inf
    episode_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def encode(self) -> bytes:
        """Serialize to bytes for checkpoint / transmission."""
        data = {
            "weights": [w.tolist() for w in self.weights],
            "fitness": self.fitness,
            "best_fitness": self.best_fitness,
            "episode_count": self.episode_count,
            "metadata": self.metadata,
        }
        return pickle.dumps(data)

    @classmethod
    def decode(cls, data: bytes) -> Agent:
        """Deserialize from bytes."""
        obj = pickle.loads(data)
        agent = cls()
        agent.weights = [np.array(w, dtype=np.float32) for w in obj["weights"]]
        agent.fitness = obj["fitness"]
        agent.best_fitness = obj["best_fitness"]
        agent.episode_count = obj["episode_count"]
        agent.metadata = obj.get("metadata", {})
        return agent


# ---------------------------------------------------------------------------
# Neural network forward pass (pure numpy — no PyTorch dependency)
# ---------------------------------------------------------------------------

def _build_network_weights(
    input_size: int,
    output_size: int,
    hidden_layers: List[int],
    activation: str = "relu",
    rng: np.random.Generator | None = None,
) -> List[np.ndarray]:
    """Xavier/He initialization for a fully-connected feed-forward network.

    Returns weight matrices only (biases omitted for simplicity in NE).
    """
    if rng is None:
        rng = np.random.default_rng()

    layers = [input_size] + hidden_layers + [output_size]
    w_list: List[np.ndarray] = []

    for i in range(len(layers) - 1):
        fan_in, fan_out = layers[i], layers[i + 1]
        # He initialization (works with ReLU-family activations)
        if activation in ("relu", "elu"):
            scale = np.sqrt(2.0 / fan_in)
        else:
            scale = np.sqrt(2.0 / (fan_in + fan_out))

        W = rng.standard_normal((fan_in, fan_out)) * scale
        w_list.append(W)

    return w_list


def _forward(
    weights: List[np.ndarray],
    x: np.ndarray,
    activation: str = "relu",
) -> np.ndarray:
    """Forward pass through the network. *x* shape: (input_size, 1)."""
    # input shape: (input_size, 1) → transpose to (1, input_size) for matmul
    h = x.T.astype(np.float64)

    n_layers = len(weights)

    for i in range(n_layers):
        W = weights[i]
        h = h @ W
        # apply activation on hidden layers only (skip last layer — tanh applied later)
        if i < n_layers - 1:
            if activation == "relu":
                h = np.maximum(h, 0.0)
            elif activation == "elu":
                h = np.where(h > 0, h, np.exp(h) - 1)
            elif activation == "tanh":
                h = np.tanh(h)
            elif activation == "sigmoid":
                h = 1 / (1 + np.exp(-np.clip(h, -500, 500)))

    # Return as column vector (output_size, 1) for consistency
    return h.T if h.ndim == 2 else h


def _activation_fn(activation: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return activation function by name."""
    if activation == "relu":
        return lambda x: np.maximum(x, 0.0)
    elif activation == "tanh":
        return np.tanh
    elif activation == "sigmoid":
        return lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    elif activation == "elu":
        return lambda x: np.where(x > 0, x, np.exp(x) - 1)
    else:
        raise ValueError(f"Unknown activation: {activation}")


def _output_activation(action_dim: int, raw_output: np.ndarray) -> np.ndarray:
    """Map network output to valid action space [-1, 1]."""
    # tanh squashes to (-1, 1), which maps directly to our action range
    return np.tanh(raw_output.flatten())


# ---------------------------------------------------------------------------
# Genetic Algorithm Engine
# ---------------------------------------------------------------------------

class NeuroEvolutionEngine:
    """Population-based neuroevolution engine.

    Manages a population of agents (neural networks) and applies
    evolutionary operations each generation: selection, crossover, mutation,
    and fitness evaluation.

    Parameters
    ----------
    input_size : int
    output_size : int
    hidden_layers : list[int]
    activation : str
        One of ``relu``, ``tanh``, ``sigmoid``, ``elu``.
    population_size : int
    elite_ratio : float
        Fraction of top agents preserved unchanged each generation.
    crossover_rate : float  [0, 1]
    mutation_rate : float  [0, 1], per-gene probability
    mutation_strength : float
        Standard-deviation multiplier for Gaussian mutation noise.
    fitness_function : str
        Which reward signal to use (passed through from config).
    """

    def __init__(
        self,
        input_size: int = 16,
        output_size: int = 8,
        hidden_layers: List[int] = None,
        activation: str = "relu",
        population_size: int = 200,
        elite_ratio: float = 0.1,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.3,
        mutation_strength: float = 0.1,
        fitness_function: str = "distance",
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers or [64, 64]
        self.activation = activation
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.fitness_function = fitness_function

        self._rng = np.random.default_rng()
        self._population: List[Agent] = []
        self._generation = 0

        # history tracking
        self.history: Dict[str, List[float]] = {
            "best_fitness": [],
            "mean_fitness": [],
            "worst_fitness": [],
            "std_fitness": [],
            "episode_counts": [],
        }

    def initialize_population(self) -> List[Agent]:
        """Create the initial random population."""
        self._population = []
        for _ in range(self.population_size):
            agent = Agent()
            agent.weights = _build_network_weights(
                self.input_size, self.output_size, self.hidden_layers,
                self.activation, rng=self._rng,
            )
            agent.fitness = -np.inf
            agent.best_fitness = -np.inf
            agent.episode_count = 0
            agent.metadata = {"generation": 0}
            self._population.append(agent)

        return list(self._population)

    def evaluate_population(
        self,
        env_class: type,
        config: Any,
        n_episodes: int = 1,
        seed_offset: int = 0,
    ) -> List[float]:
        """Run each agent through *n_episodes* of the environment.

        Parameters
        ----------
        env_class : type
            Environment class (e.g., ``OrbitSimulator``).
        config : Config | dict
            Configuration object or dict for constructing environments.
        n_episodes : int
            Episodes per agent.
        seed_offset : int
            Base random seed offset for reproducibility.

        Returns
        -------
        fitness_scores : list[float]
        """
        scores: List[float] = []

        for i, agent in enumerate(self._population):
            total_reward = 0.0
            max_episodes_survived = 0

            for ep in range(n_episodes):
                seed = int(self._rng.integers(0, 2**31)) + seed_offset * self.population_size + i * n_episodes + ep
                env = env_class(config=config if hasattr(config, 'sim_time_step') else None, seed=seed)

                obs = env.reset()
                episode_reward = 0.0
                steps_survived = 0

                while not (done := False):
                    # forward pass through agent's network
                    x = obs.reshape(-1, 1).astype(np.float64)
                    raw_output = _forward(agent.weights, x, self.activation)

                    action = _output_activation(self.output_size, raw_output)

                    obs, reward, done, info = env.step(action)
                    episode_reward += float(reward)
                    steps_survived += 1

                    if done:
                        break

                env.close()
                total_reward += episode_reward / n_episodes
                max_episodes_survived = max(max_episodes_survived, steps_survived)

            agent.fitness = total_reward
            agent.best_fitness = max(agent.best_fitness, total_reward)
            agent.episode_count = max_episodes_survived
            agent.metadata["generation"] = self._generation
            scores.append(total_reward)

        # update history
        self.history["best_fitness"].append(max(scores))
        self.history["mean_fitness"].append(float(np.mean(scores)))
        self.history["worst_fitness"].append(min(scores))
        self.history["std_fitness"].append(float(np.std(scores)))
        self.history["episode_counts"].append(max_episodes_survived)

        return scores

    def evaluate_population_batch(
        self,
        n_agents_per_episode: int = 100,
        max_steps_per_episode: int = 3600,
        dt: float = 0.1,
        seed_offset: int = 0,
    ) -> List[float]:
        """Evaluate population using batched simulation (fast path).

        Instead of running each agent's episode sequentially, this method
        runs all episodes in parallel through the vectorized simulator.
        This provides ~10-50x speedup for populations > 50 agents.

        Each agent gets its own independent simulation instance with
        unique initial conditions (random velocity perturbation).

        Parameters
        ----------
        n_agents_per_episode : int
            How many parallel instances to use per evaluation cycle.
            If population_size > this, multiple cycles are run.
        max_steps_per_episode : int
            Maximum steps before forced episode end.
        dt : float
            Physics timestep in seconds.
        seed_offset : int
            Base random seed offset for reproducibility.

        Returns
        -------
        fitness_scores : list[float]
        """
        from ksp_neuro.sim_env.batch_env import BatchEnv

        pop = self._population
        n_agents = len(pop)
        scores: List[float] = []

        # Process agents in batches of n_agents_per_episode
        for batch_start in range(0, n_agents, n_agents_per_episode):
            batch_end = min(batch_start + n_agents_per_episode, n_agents)
            batch_size = batch_end - batch_start
            agent_indices = list(range(batch_start, batch_end))

            env = BatchEnv(
                n_agents=batch_size,
                max_steps_per_episode=max_steps_per_episode,
                dt=dt,
                seed=seed_offset + batch_start * 1000,
            )

            obs = env.reset()
            episode_rewards = np.zeros(batch_size, dtype=np.float64)
            steps_survived = np.zeros(batch_size, dtype=int)

            for step in range(max_steps_per_episode):
                # Forward pass through each agent's network
                actions = np.zeros((batch_size, len(_ACTION_KEYS)), dtype=np.float32)
                for i in range(batch_size):
                    x_i = obs[i].reshape(-1, 1).astype(np.float64)  # [input_size, 1]
                    raw_output = _forward(pop[agent_indices[i]].weights, x_i, self.activation)
                    action_i = _output_activation(self.output_size, raw_output)
                    actions[i] = action_i

                obs, rewards, dones, info = env.step(actions)
                episode_rewards += np.where(dones, 0.0, rewards)  # only accumulate for active agents
                steps_survived += np.where(~dones, 1, 0)

                if np.all(dones):
                    break

            # Record fitness scores
            env.close()
            for i, agent_idx in enumerate(agent_indices):
                pop[agent_idx].fitness = float(episode_rewards[i])
                pop[agent_idx].best_fitness = max(pop[agent_idx].best_fitness, episode_rewards[i])
                pop[agent_idx].episode_count = int(steps_survived[i])
                pop[agent_idx].metadata["generation"] = self._generation
                scores.append(float(episode_rewards[i]))

        # Update history
        self.history["best_fitness"].append(max(scores))
        self.history["mean_fitness"].append(float(np.mean(scores)))
        self.history["worst_fitness"].append(min(scores))
        self.history["std_fitness"].append(float(np.std(scores)))

        return scores

    def evolve(self) -> List[Agent]:
        """Perform one generation of evolutionary operations.

        Returns
        -------
        new_population : list[Agent]
        """
        # sort by fitness descending
        sorted_agents = sorted(self._population, key=lambda a: a.fitness, reverse=True)

        # determine elite count
        n_elites = max(1, int(self.population_size * self.elite_ratio))
        elites = list(sorted_agents[:n_elites])  # preserved unchanged

        # fitness-proportionate selection (with offset to avoid negative probs)
        min_fit = min(a.fitness for a in sorted_agents)
        adjusted_fitness = [a.fitness - min_fit + 1e-6 for a in sorted_agents]
        total_fit = sum(adjusted_fitness)
        probabilities = [f / total_fit for f in adjusted_fitness]

        # generate new population
        new_population: List[Agent] = list(elites)

        while len(new_population) < self.population_size:
            parent1, parent2 = self._tournament_selection(probabilities, k=5)

            if self._rng.random() < self.crossover_rate:
                child_weights = self._crossover(parent1.weights, parent2.weights)
            else:
                # clone parent 1
                child_weights = [w.copy() for w in parent1.weights]

            # mutation
            child_weights = self._mutate(child_weights)

            child = Agent(
                weights=child_weights,
                fitness=-np.inf,
                best_fitness=-np.inf,
                episode_count=0,
                metadata={"generation": self._generation + 1},
            )
            new_population.append(child)

        self._population = new_population
        self._generation += 1
        return list(new_population)

    def get_best_agent(self) -> Agent:
        """Return the agent with the highest best-ever fitness."""
        return max(self._population, key=lambda a: a.best_fitness)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save population to disk (pickle)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "generation": self._generation,
            "population": [a.encode() for a in self._population],
            "history": self.history,
            "config": {
                "input_size": self.input_size,
                "output_size": self.output_size,
                "hidden_layers": self.hidden_layers,
                "activation": self.activation,
            },
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
        self.population_size = len(data["population"])  # restore if changed
        self._population = [Agent.decode(enc) for enc in data["population"]]

    def get_history(self) -> Dict[str, List[float]]:
        """Return copy of fitness history."""
        return {k: list(v) for k, v in self.history.items()}

    # ---- private evolutionary operations -----------------------------------
    def _tournament_selection(
        self, probabilities: List[float], k: int = 5
    ) -> Tuple[Agent, Agent]:
        """Tournament selection returning two distinct parents."""
        n = len(self._population)
        idx1 = self._rng.choice(n, size=k, replace=False)
        idx2 = self._rng.choice(n, size=k, replace=False)

        fit1 = [self._population[i].fitness for i in idx1]
        fit2 = [self._population[i].fitness for i in idx2]

        parent_a = self._population[idx1[np.argmax(fit1)]]
        parent_b = self._population[idx2[np.argmax(fit2)]]

        # ensure two distinct parents
        if parent_a is not parent_b:
            return parent_a, parent_b

        # pick a different winner from idx2 (avoiding idx1 winner)
        loser_idx = np.argmax(fit1)
        for i in idx2:
            if i != idx1[loser_idx]:
                parent_c = self._population[i]
                return parent_a, parent_c

        # fallback: pick any other agent
        for i in range(n):
            if i != idx1[np.argmax(fit1)]:
                return parent_a, self._population[i]
        return parent_a, parent_b  # worst case

    def _crossover(
        self, w1: List[np.ndarray], w2: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Uniform crossover between two parent weight lists."""
        child = []
        for a, b in zip(w1, w2):
            if a.shape != b.shape:
                # mismatched layers — clone parent 1 (shouldn't happen)
                child.append(a.copy())
            else:
                mask = self._rng.random(a.shape) > 0.5
                child.append(np.where(mask, a, b).copy())
        return child

    def _mutate(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Gaussian mutation on weight matrices."""
        mutated = []
        for w in weights:
            if isinstance(w, np.ndarray) and w.ndim >= 2:
                noise = self._rng.normal(0, self.mutation_strength, size=w.shape).astype(w.dtype)
                mask = self._rng.random(w.shape) < self.mutation_rate
                mutated.append(np.where(mask, w + noise, w))
            else:
                mutated.append(w.copy())
        return mutated


# Import action keys for batch evaluation
from ksp_neuro.sim_env.vectorized_sim import _ACTION_KEYS  # noqa: E402

__all__ = [
    "NeuroEvolutionEngine",
    "Agent",
    "_build_network_weights",
    "_forward",
    "_output_activation",
]
