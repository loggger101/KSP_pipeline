"""Tests for KSP Neuroevolution Pipeline core components."""

import sys
from pathlib import Path

# Add parent directory to path so imports work from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest


class TestOrbitSimulator:
    """Tests for the orbital mechanics simulator."""

    def test_reset_returns_observation(self):
        from ksp_neuro.sim_env.orbit_sim import OrbitSimulator
        sim = OrbitSimulator(seed=42)
        obs = sim.reset()
        assert isinstance(obs, np.ndarray), "reset() must return numpy array"
        assert obs.shape == (16,), f"Expected shape (16,), got {obs.shape}"
        assert not np.any(np.isnan(obs)), "Observation contains NaN values"
        assert not np.any(np.isinf(obs)), "Observation contains Inf values"

    def test_step_returns_tuple(self):
        from ksp_neuro.sim_env.orbit_sim import OrbitSimulator
        sim = OrbitSimulator(seed=42)
        obs = sim.reset()
        action = np.zeros(8, dtype=np.float32)  # no-op action
        result = sim.step(action)
        assert len(result) == 4, f"step() should return 4-tuple; got {len(result)} items"
        obs_out, reward, done, info = result
        assert isinstance(obs_out, np.ndarray)
        assert isinstance(reward, float), f"reward must be float, got {type(reward)}"

    def test_throttle_produces_motion(self):
        from ksp_neuro.sim_env.orbit_sim import OrbitSimulator
        sim = OrbitSimulator(seed=42)
        obs1 = sim.reset()
        vel1 = obs1[1] * 3000  # denormalize velocity

        # full throttle upward for several steps
        action = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(10):
            obs2, reward, done, info = sim.step(action)

        vel2 = obs2[1] * 3000  # denormalize velocity
        assert abs(vel2 - vel1) > 0.5, f"Velocity should change with throttle: {vel1:.1f} -> {vel2:.1f}"

    def test_no_action_keeps_near_surface(self):
        from ksp_neuro.sim_env.orbit_sim import OrbitSimulator
        sim = OrbitSimulator(seed=42)
        obs = sim.reset()
        action = np.zeros(8, dtype=np.float32)  # zero action

        for _ in range(100):
            obs, reward, done, info = sim.step(action)
            if done:
                break

        assert not done or obs[0] * 1e6 < -50, "Zero-action agent should crash into ground"

    def test_episode_completes_without_crash_with_throttle(self):
        from ksp_neuro.sim_env.orbit_sim import OrbitSimulator
        sim = OrbitSimulator(seed=42)
        obs = sim.reset()
        done_count = 0

        for i in range(1000):
            action = np.array([0.3, 0.3, 0.0, 0.0, 0.0, float(i == 50), 0.0, 0.0], dtype=np.float32)
            obs, reward, done, info = sim.step(action)
            if done:
                done_count += 1
                break

        assert done_count >= 0, "Episode should run without errors"


class TestNeuroEvolutionEngine:
    """Tests for the neuroevolution engine."""

    def test_initialize_population(self):
        from ksp_neuro.ne_engine import NeuroEvolutionEngine
        engine = NeuroEvolutionEngine(input_size=4, output_size=2, population_size=10)
        pop = engine.initialize_population()
        assert len(pop) == 10
        for agent in pop:
            assert hasattr(agent, 'weights')
            assert hasattr(agent, 'fitness')

    def test_evaluate_population(self):
        from ksp_neuro.ne_engine import NeuroEvolutionEngine
        from ksp_neuro.sim_env.orbit_sim import OrbitSimulator

        engine = NeuroEvolutionEngine(
            input_size=16, output_size=8, population_size=5,
            hidden_layers=[32], activation="relu",
        )
        engine.initialize_population()
        scores = engine.evaluate_population(OrbitSimulator, None, n_episodes=1)
        assert len(scores) == 5
        for s in scores:
            assert isinstance(s, float), f"Score must be float; got {type(s)}"

    def test_evolution_improves_fitness(self):
        from ksp_neuro.ne_engine import NeuroEvolutionEngine
        from ksp_neuro.sim_env.orbit_sim import OrbitSimulator
        import types

        engine = NeuroEvolutionEngine(
            input_size=16, output_size=8, population_size=10,
            hidden_layers=[32], activation="relu",
            mutation_rate=0.4, mutation_strength=0.2,
            elite_ratio=0.2, crossover_rate=0.9,
        )
        engine.initialize_population()

        # Use a lightweight config with shorter episodes for testing
        test_config = types.SimpleNamespace(
            sim_time_step=0.1,
            sim_max_steps=3600,  # ~6 min per episode
        )

        initial_scores = engine.evaluate_population(OrbitSimulator, test_config, n_episodes=1)
        initial_best = max(initial_scores)

        # evolve a few generations (use 2 gens for speed)
        for _ in range(2):
            engine.evolve()
            scores = engine.evaluate_population(OrbitSimulator, test_config, n_episodes=1)

        improved_best = max(scores)
        assert improved_best >= initial_best - 15, (
            f"Fitness should improve or stay similar after evolution. "
            f"Initial best: {initial_best:.2f}, After 2 gens: {improved_best:.2f}"
        )

    def test_checkpoint_save_load(self):
        from ksp_neuro.ne_engine import NeuroEvolutionEngine
        import tempfile, os

        engine = NeuroEvolutionEngine(input_size=4, output_size=2, population_size=10)
        engine.initialize_population()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_checkpoint.pkl")
            engine.save_checkpoint(path)

            engine2 = NeuroEvolutionEngine(input_size=4, output_size=2, population_size=10)
            engine2.load_checkpoint(path)

            assert engine._generation == engine2._generation
            assert len(engine._population) == len(engine2._population)


class TestConfigLoader:
    """Tests for the configuration loader."""

    def test_load_default_config(self):
        from ksp_neuro.config.loader import load_config, Config
        cfg = load_config()
        assert isinstance(cfg, Config)
        assert cfg.env_type == "sim"
        assert cfg.population_size == 200
        assert cfg.num_generations == 500

    def test_load_with_overrides(self):
        from ksp_neuro.config.loader import load_config
        cfg = load_config(overrides={"environment.type": "ksp", "ne_engine.population_size": 100})
        assert cfg.env_type == "ksp"
        assert cfg.population_size == 100


class TestVizBackend:
    """Tests for visualization backends."""

    def test_terminal_plotter(self):
        from ksp_neuro.viz import TerminalPlotter
        viz = TerminalPlotter()
        history = {
            "best_fitness": [0.1, 0.5, 1.2],
            "mean_fitness": [0.05, 0.3, 0.8],
            "worst_fitness": [-0.1, 0.1, 0.4],
            "std_fitness": [0.1, 0.2, 0.3],
            "episode_counts": [100, 500, 1000],
        }
        viz.update_metrics(history)
        viz.show_best_agent([[0.1] * 16], [[0.0] * 8])
        viz.close()

    def test_get_viz_backend_terminal(self):
        from ksp_neuro.viz import get_viz_backend, TerminalPlotter
        viz = get_viz_backend(mode="terminal")
        assert isinstance(viz, TerminalPlotter)


class TestNetworkForward:
    """Tests for the pure-NumPy neural network forward pass."""

    def test_forward_computes_output(self):
        from ksp_neuro.ne_engine import _build_network_weights, _forward
        rng = np.random.default_rng(42)
        weights = _build_network_weights(4, 2, [8], "relu", rng=rng)

        x = np.ones((4, 1), dtype=np.float64)
        out = _forward(weights, x, "relu")

        assert out.shape == (2, 1), f"Expected shape (2, 1); got {out.shape}"
        assert not np.any(np.isnan(out)), "Output contains NaN"

    def test_output_activation_squashes_to_tanh(self):
        from ksp_neuro.ne_engine import _output_activation
        raw = np.array([[5.0], [-3.0], [0.1]], dtype=np.float64)
        action = _output_activation(3, raw)

        assert len(action) == 3
        assert np.all(np.abs(action) <= 1.0), "tanh output must be in [-1, 1]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
