"""Configuration loader — YAML-based, with defaults and overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # pip install pyyaml
except ImportError:
    raise ImportError("Install PyYAML: pip install pyyaml")


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class Config:
    """Flat wrapper around the nested YAML config."""

    # Environment
    env_type: str = "sim"
    sim_time_step: float = 0.1
    sim_max_steps: int = 3600
    sim_gravity_constant: float = 6.674e-11
    sim_earth_radius_m: float = 6_000_000
    sim_kerbin_mass_kg: float = 8.738e22
    sim_atmosphere_enabled: bool = True
    sim_sea_level_pressure_pa: float = 90_000
    sim_scale_height_m: float = 5_600
    sim_seed: int = 42
    ksp_host: str = "localhost"
    ksp_port: int = 6767

    # NE engine
    algorithm: str = "ga"
    population_size: int = 200
    num_generations: int = 500
    elite_ratio: float = 0.1
    crossover_rate: float = 0.8
    mutation_rate: float = 0.3
    mutation_strength: float = 0.1
    fitness_function: str = "distance"
    tournament_size: int = 5
    elitism: bool = True

    # Network
    input_size: int = 16
    output_size: int = 8
    hidden_layers: list = field(default_factory=lambda: [64, 64])
    activation: str = "relu"
    use_bias: bool = True

    # Viz
    dashboard_port: int = 8501
    update_interval_s: float = 2.0
    plot_history_size: int = 100
    save_checkpoint_every_n: int = 50

    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True


def load_config(
    path: str | Path | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Config:
    """Load config from YAML (default *path* → ``default_config.yaml``).

    Parameters
    ----------
    path : optional
        Explicit YAML file.  Falls back to the bundled default.
    overrides : optional
        Flat dict of key-value pairs that override loaded values.
        Nested keys use dot notation, e.g. ``{"env_type": "ksp"}``.

    Returns
    -------
    Config
        Populated dataclass instance.
    """
    # 1. load base YAML
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    # 2. apply overrides (flat → nested via dot keys)
    if overrides:
        flat = {}
        for key, value in overrides.items():
            parts = key.split(".")
            d = flat
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
        raw = _deep_merge(raw, flat)

    # 3. flatten nested dict → Config fields
    env = raw.get("environment", {})
    sim = env.get("sim", {})
    ksp_conf = env.get("ksp", {})
    ne = raw.get("ne_engine", {})
    net = raw.get("network", {})
    viz = raw.get("viz", {})
    log = raw.get("logging", {})

    return Config(
        # environment
        env_type=env.get("type", "sim"),
        sim_time_step=sim.get("time_step", 0.1),
        sim_max_steps=sim.get("max_steps_per_episode", 3600),
        sim_gravity_constant=sim.get("gravity_constant", 6.674e-11),
        sim_earth_radius_m=sim.get("earth_radius_m", 6_000_000),
        sim_kerbin_mass_kg=sim.get("kerbin_mass_kg", 8.738e22),
        sim_atmosphere_enabled=sim.get("atmosphere_enabled", True),
        sim_sea_level_pressure_pa=sim.get("sea_level_pressure_pa", 90_000),
        sim_scale_height_m=sim.get("scale_height_m", 5_600),
        sim_seed=sim.get("seed", 42),
        ksp_host=ksp_conf.get("host", "localhost"),
        ksp_port=ksp_conf.get("port", 6767),
        # ne engine
        algorithm=ne.get("algorithm", "ga"),
        population_size=ne.get("population_size", 200),
        num_generations=ne.get("num_generations", 500),
        elite_ratio=ne.get("elite_ratio", 0.1),
        crossover_rate=ne.get("crossover_rate", 0.8),
        mutation_rate=ne.get("mutation_rate", 0.3),
        mutation_strength=ne.get("mutation_strength", 0.1),
        fitness_function=ne.get("fitness_function", "distance"),
        tournament_size=ne.get("tournament_size", 5),
        elitism=ne.get("elitism", True),
        # network
        input_size=net.get("input_size", 16),
        output_size=net.get("output_size", 8),
        hidden_layers=net.get("hidden_layers", [64, 64]),
        activation=net.get("activation", "relu"),
        use_bias=net.get("use_bias", True),
        # viz
        dashboard_port=viz.get("dashboard_port", 8501),
        update_interval_s=viz.get("update_interval_s", 2.0),
        plot_history_size=viz.get("plot_history_size", 100),
        save_checkpoint_every_n=viz.get("save_checkpoint_every_n", 50),
        # logging
        log_level=log.get("level", "INFO"),
        log_dir=log.get("log_dir", "./logs"),
        checkpoint_dir=log.get("checkpoint_dir", "./checkpoints"),
        save_best_only=log.get("save_best_only", True),
    )


__all__ = ["Config", "load_config"]
