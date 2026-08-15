# KSP Neuroevolution Pipeline

Genetic / evolutionary neural network training pipeline for Kerbal Space Program (KSP). Trains agents to perform orbital maneuvers, landings, and spaceflight tasks using neuroevolution of augmenting topologies (NEAT), multi-objective optimization (NSGA-II), and distributed GPU-accelerated evaluation.

## Architecture Overview

```
ksp_pipeline/
├── ksp_neuro/
│   ├── __init__.py
│   ├── train.py                  # Main CLI entry point
│   ├── ne_engine/                # Neuroevolution core
│   │   ├── torch_engine.py       # PyTorch GPU evaluation engine (TorchMLP, TorchLSTM, TorchResNet)
│   │   ├── neat_evaluator.py     # NEAT genome representation & dynamic graph building
│   │   ├── starter_agents.py     # Pre-trained starter agent definitions + weight loading
│   │   ├── multi_obj.py          # NSGA-II multi-objective fitness evaluator
│   │   ├── analytics.py          # Training analytics: learning curves, convergence, diversity
│   │   └── weights/              # Pre-computed starter agent pickle weights
│   ├── training/                 # Training infrastructure
│   │   ├── distributed.py        # Ray-based parallel evaluation engine
│   │   └── checkpoint.py         # Checkpoint save/load with cross-worker sync
│   ├── ksp_interface/            # KSP live game integration
│   │   ├── hardened.py           # Production kRPC interface (connection pooling, auto-reconnect)
│   │   └── state_parser.py       # Orbital/state parsing from kRPC messages
│   ├── viz/                      # Visualization & monitoring
│   │   ├── analytics_dashboard.py  # Streamlit real-time dashboard
│   │   └── agent_viewer.py       # Per-agent episode replay viewer
│   └── sim_env/                  # Simulation environment (no KSP required)
│       ├── vectorized_sim.py     # Batched orbital simulator (~350K ticks/sec on CPU)
│       └── orbit_simulator.py    # Single-agent high-fidelity orbital mechanics
├── tests/                        # Unit + integration tests
├── data/
│   └── checkpoints/              # Saved training checkpoints
├── pyproject.toml
└── README.md
```

## Quick Start

### Installation

```bash
# Core dependencies
pip install -e ".[dev]"

# GPU support (optional, requires CUDA toolkit)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# KSP live mode (requires KRPC mod installed in KSP)
pip install krpc
```

### Training Modes

**Simulation-only (no KSP required):**
```bash
python -m ksp_neuro.train \
    --algorithm neat \
    --network mlp \
    --simulator vectorized \
    --population-size 200 \
    --generations 500 \
    --task vertical_burn \
    --batch-size 64 \
    --output-dir data/checkpoints
```

**Distributed training (multi-GPU/multi-machine):**
```bash
# On head node:
python -m ksp_neuro.train \
    --algorithm neat \
    --network mlp \
    --distributed \
    --num-workers 4 \
    --gpus 0,1,2,3 \
    --population-size 400 \
    --generations 1000

# On worker nodes (same command):
python -m ksp_neuro.train \
    --algorithm neat \
    --network mlp \
    --distributed \
    --num-workers 4 \
    --gpus 0,1,2,3 \
    --population-size 400 \
    --generations 1000
```

**KSP live mode (requires KRPC running):**
```bash
python -m ksp_neuro.train \
    --algorithm neat \
    --network lstm \
    --mode live \
    --krpc-host localhost \
    --krpc-port 5000 \
    --population-size 100 \
    --generations 200
```

**Resume from checkpoint:**
```bash
python -m ksp_neuro.train \
    --resume data/checkpoints/run_042/checkpoint_best.pkl \
    --generations 500
```

### Analytics Dashboard

```bash
streamlit run ksp_neuro/viz/analytics_dashboard.py \
    --data-dir data/checkpoints \
    --auto-refresh 5
```

## Training Tasks

| Task | Description | Observation Space | Action Space |
|------|-------------|-------------------|--------------|
| `vertical_burn` | Ascend vertically, stabilize at altitude | 12-dim state | throttle (0-1) |
| `orbital_insertion` | Reach circular orbit around Kerbin | 15-dim state | throttle + pitch/yaw |
| `kerbin_orbit` | Full Kerbin orbit insertion & stabilization | 18-dim state | full control surfaces |
| `duna_transfer` | Hohmann transfer to Duna | 20-dim state | multi-stage thrust profiles |
| `landing` | Powered descent landing on any body | 22-dim state | throttle + retrograde targeting |

## Multi-Objective Fitness (NSGA-II)

Each agent is evaluated on multiple objectives simultaneously:
- **Altitude achieved** — higher is better
- **Delta-V efficiency** — fuel remaining / delta-v spent ratio
- **Maneuver accuracy** — distance from target orbit/altitude
- **Stability index** — variance of control inputs (lower = more stable)

Pareto front tracking maintains diversity across the population. Agents on the Pareto front get bonus reproduction probability.

## NEAT Genome Encoding

Genomes encode:
- **Topology**: node types (input, bias, hidden, output), connection weights, enabled/disabled flags
- **Innovation IDs**: track historical mutations for crossover compatibility
- **Fitness history**: per-generation fitness tracking for speciation

Dynamic graph building at evaluation time allows evolved architectures to be converted to PyTorch modules on-the-fly.

## Simulation Performance

| Mode | Ticks/sec (population=100) | Notes |
|------|---------------------------|-------|
| Single-agent OrbitSimulator | ~5K | High-fidelity, CPU-bound |
| Vectorized batched sim | ~350K | NumPy vectorization, 100x speedup |
| GPU-accelerated (torch) | N/A | Forward passes only; sim remains CPU |

## Requirements

- Python >= 3.9
- PyTorch 2.x with CUDA support (optional but recommended)
- KSP + kRPC mod (for live mode training)
- Ray cluster (for distributed mode)

## License

MIT
