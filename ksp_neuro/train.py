"""Main training loop — ties together NE engine, environment, and visualization.

Usage:
    python -m ksp_neuro.train                    # run with defaults
    python -m ksp_neuro.train --config my_config.yaml  # custom config
    python -m ksp_neuro.train --env sim          # force simulation mode
    python -m ksp_neuro.train --viz streamlit    # use Streamlit dashboard

Long-running training is supported. Checkpoints are saved periodically;
the trainer can be resumed with ``--resume ./checkpoints/latest.pkl``.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure ksp_neuro is on the path
import os
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from ksp_neuro.config.loader import Config, load_config
from ksp_neuro.ne_engine import NeuroEvolutionEngine, _forward
from ksp_neuro.sim_env.orbit_sim import OrbitSimulator
from ksp_neuro.viz import get_viz_backend, BaseVizBackend


# ---------------------------------------------------------------------------
# Graceful shutdown handler
# ---------------------------------------------------------------------------

_running = True


def _signal_handler(signum: int, frame: Any) -> None:
    global _running
    print(f"\n\nReceived signal {signum}. Saving checkpoint and shutting down gracefully...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# JSON history writer — for the Streamlit dashboard
# ---------------------------------------------------------------------------

def write_history_json(history: Dict[str, List[float]], path: str = "./logs/ksp_neuro_history.json") -> None:
    """Write fitness history to a JSON file for the Streamlit dashboard."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training(
    config_path: Optional[str] = None,
    env_type: str = "sim",
    algorithm: str = "ga",
    network_type: str = "mlp",
    viz_mode: str = "terminal",
    resume_from: Optional[str] = None,
    max_generations: Optional[int] = None,
    use_batch: bool = False,
):
    """Run the neuroevolution training pipeline.

    Parameters
    ----------
    config_path : optional
        YAML config file. Falls back to ``default_config.yaml``.
    env_type : str
        ``"sim"`` (built-in simulator) or ``"ksp"`` (live game via KRPC).
    viz_mode : str
        Visualization backend: ``"terminal"``, ``"matplotlib"``, ``"streamlit"``.
    resume_from : optional
        Path to a pickle checkpoint to resume training from.
    max_generations : optional
        Override the generation count from config.
    """

    # ---- load configuration ------------------------------------------------
    overrides = {}
    if env_type != "sim":
        overrides["environment.type"] = env_type
    cfg = load_config(config_path, overrides=overrides)

    if max_generations is not None:
        cfg.num_generations = max_generations

    print("=" * 70)
    print("  KSP Neuroevolution Pipeline v" + __import__("ksp_neuro").__version__)
    print("=" * 70)
    print(f"  Environment : {cfg.env_type}")
    print(f"  Algorithm   : {cfg.algorithm}")
    print(f"  Population  : {cfg.population_size} agents")
    print(f"  Generations : {cfg.num_generations}")
    if network_type == "lstm":
        print(f"  Network     : LSTM(input={cfg.input_size}, hidden={cfg.hidden_layers[0]}, output={cfg.output_size})")
    elif network_type == "resnet":
        print(f"  Network     : ResNet(input={cfg.input_size}, tower_widths={cfg.hidden_layers}, res_blocks=3, output={cfg.output_size})")
    else:
        print(f"  Network     : MLP(input={cfg.input_size}, hidden={cfg.hidden_layers}, output={cfg.output_size})")
    print(f"  Activation  : {cfg.activation}")
    print(f"  Viz mode    : {viz_mode}")
    print("=" * 70)

    # ---- initialize components --------------------------------------------
    if algorithm == "neat":
        from ksp_neuro.ne_engine.neat import NEATEngine
        
        engine = NEATEngine(
            input_size=cfg.input_size,
            output_size=cfg.output_size,
            population_size=cfg.population_size,
            elite_ratio=cfg.elite_ratio,
            mutation_rate=cfg.mutation_rate,
            mutation_strength=cfg.mutation_strength,
            crossover_rate=cfg.crossover_rate,
        )
        print(f"  Algorithm   : NEAT (topology evolving)")
    else:
        engine = NeuroEvolutionEngine(
            input_size=cfg.input_size,
            output_size=cfg.output_size,
            hidden_layers=cfg.hidden_layers,
            activation=cfg.activation,
            population_size=cfg.population_size,
            elite_ratio=cfg.elite_ratio,
            crossover_rate=cfg.crossover_rate,
            mutation_rate=cfg.mutation_rate,
            mutation_strength=cfg.mutation_strength,
            fitness_function=cfg.fitness_function,
        )

    # viz backend
    viz = get_viz_backend(
        mode=viz_mode,
        update_interval_s=cfg.update_interval_s,
        dashboard_port=cfg.dashboard_port,
        history_size=cfg.plot_history_size,
    )

    # environment class
    env_class = OrbitSimulator  # default; swap for KSPInterface if needed

    # ---- resume from checkpoint or initialize -----------------------------
    start_gen = 0
    if resume_from:
        ckpt_path = Path(resume_from)
        if ckpt_path.exists():
            engine.load_checkpoint(ckpt_path)
            start_gen = engine._generation
            print(f"Resumed from generation {start_gen}")
        else:
            print(f"Checkpoint not found at {resume_from}; starting fresh.")

    if start_gen == 0 and resume_from is None:
        engine.initialize_population()

    history = engine.get_history()

    # ---- main training loop -----------------------------------------------
    best_agent_overall = None
    best_fitness_overall = -np.inf
    global_start_time = time.time()

    for gen in range(start_gen, cfg.num_generations):
        gen_elapsed = time.time() - global_start_time
        print(f"\n--- Generation {gen + 1}/{cfg.num_generations} "
              f"({gen_elapsed:.0f}s elapsed) ---")

        # evaluate population (batch mode = vectorized sim, ~10-50x faster)
        if use_batch:
            scores = engine.evaluate_population_batch(
                n_agents_per_episode=min(cfg.population_size, 200),
                max_steps_per_episode=cfg.sim_max_steps,
                dt=cfg.sim_time_step,
                seed_offset=gen * 1000,
            )
        else:
            if algorithm == "neat":
                # NEAT needs genome → weights dict conversion for evaluation
                from ksp_neuro.ne_engine.neat_eval import evaluate_genome
                scores = engine.evaluate_population(
                    eval_fn=evaluate_genome,
                    env_class=env_class,
                    config=cfg if hasattr(cfg, 'sim_time_step') else None,
                    n_episodes=3,
                )
            else:
                scores = engine.evaluate_population(
                    env_class=env_class,
                    config=cfg if hasattr(cfg, 'sim_time_step') else None,
                    n_episodes=3,  # average over 3 episodes for stability
                    seed_offset=gen * 1000,
                )

        history = engine.get_history()

        # track best agent (or genome for NEAT)
        current_best = engine.get_best_genome() if algorithm == "neat" else engine.get_best_agent()
        if current_best.best_fitness > best_fitness_overall:
            best_fitness_overall = current_best.best_fitness
            best_agent_overall = current_best
            print(f"  [BEST] New best fitness: {best_fitness_overall:.4f} (gen {gen + 1})")

        # viz update
        viz.update_metrics(history)

        # show top agent's latest episode
        if gen % 5 == 0 or gen < 3:
            test_env = env_class(config=cfg if hasattr(cfg, 'sim_time_step') else None, seed=42)
            obs = test_env.reset()
            obs_hist, act_hist = [], []
            ep_reward = 0.0
            while not (done := False):
                x = obs.reshape(-1, 1).astype(np.float64)
                raw_output = _forward(current_best.weights, x, cfg.activation if hasattr(cfg, 'activation') else 'relu')
                action = np.tanh(raw_output.flatten())
                obs, reward, done, info = test_env.step(action)
                ep_reward += float(reward)
                obs_hist.append(obs.tolist())
                act_hist.append(action.tolist())

                if gen < 5:  # only show rollout for first few gens or every 5th
                    viz.show_best_agent(obs_hist, act_hist)

                if done:
                    break

            test_env.close()
            print(f"  Episode reward (best agent): {ep_reward:.4f}, steps: {len(obs_hist)}")

        # periodic save
        if gen > 0 and gen % cfg.save_checkpoint_every_n == 0 or gen == cfg.num_generations - 1:
            ckpt_dir = Path(cfg.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            engine.save_checkpoint(ckpt_dir / f"gen_{gen:06d}.pkl")

            # also save "latest" symlink copy
            engine.save_checkpoint(ckpt_dir / "latest.pkl")

        # write JSON history for Streamlit dashboard
        write_history_json(history, "./logs/ksp_neuro_history.json")

        if not _running:
            print("Shutting down...")
            break

    # ---- final save -------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Training complete! {cfg.num_generations - start_gen} generations.")
    print(f"Best ever fitness: {best_fitness_overall:.4f}")

    if best_agent_overall is not None:
        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        engine.save_checkpoint(ckpt_dir / "best.pkl")
        print(f"Best agent saved to {ckpt_dir}/best.pkl")

    # save final history JSON
    write_history_json(engine.get_history(), "./logs/ksp_neuro_history.json")

    viz.close()
    print("Done.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="KSP Neuroevolution Pipeline — train agents via genetic algorithms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", default=None, help="Path to YAML config file")
    parser.add_argument("--env", "-e", choices=["sim", "ksp"], default="sim",
                        help="Environment type: sim (built-in) or ksp (live game)")
    parser.add_argument("--algorithm", "-a", choices=["ga", "neat"], default="ga",
                        help="Neuroevolution algorithm: ga (fixed architecture) or neat (topology evolves)")
    parser.add_argument("--network", "-n", choices=["mlp", "lstm", "resnet"], default="mlp",
                        help="Network architecture: mlp (standard), lstm (temporal memory), resnet (deep skip connections)")
    parser.add_argument("--viz", "-v", choices=["terminal", "matplotlib", "streamlit"],
                        default="terminal", help="Visualization backend")
    parser.add_argument("--resume", "-r", default=None, help="Path to checkpoint pickle to resume from")
    parser.add_argument("--generations", "-g", type=int, default=None, help="Override max generations")
    parser.add_argument("--batch", action="store_true",
                        help="Use batched (vectorized) evaluation — 10-50x faster for large populations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        config_path=args.config,
        env_type=args.env,
        algorithm=args.algorithm,
        network_type=args.network,
        viz_mode=args.viz,
        resume_from=args.resume,
        max_generations=args.generations,
        use_batch=getattr(args, 'batch', False),
    )
