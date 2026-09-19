# KSP Neuroevolution Pipeline

Evolutionary neural networks trained to play Kerbal Space Program. Agents learn orbital mechanics, rocket staging, atmospheric flight through genetic algorithms - no human demonstrations or gradient-based RL required.

## Quick Start

### Install Dependencies
```bash
pip install numpy pyyaml streamlit plotly pytest
```

### Run Training (Simulation Mode)
The built-in orbital simulator lets you train agents immediately without KSP installed:
```bash
python -m ksp_neuro.train --env sim --viz terminal --generations 100
```

### Streamlit Dashboard (Recommended for Long Runs)
```bash
streamlit run ksp_neuro/viz/dashboard.py --server.port 8501 &
python -m ksp_neuro.train --env sim --viz streamlit --generations 500
```

## Architecture

```
ksp_neuro/
    train.py          # Main entry point / training loop
    config/           # YAML configuration system
    ne_engine/        # Neuroevolution engine (GA)
    sim_env/          # Orbital mechanics simulator
    ksp_interface/    # Live KSP game interaction (kRPC)
    viz/              # Visualization backends + dashboard
    utils/            # Shared utilities

tests/                # Unit tests
requirements.txt      # Python dependencies
```

## Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--config, -c` | YAML config file path | bundled default |
| `--env, -e` | Environment type (sim or ksp) | sim |
| `--viz, -v` | Viz backend (terminal, matplotlib, streamlit) | terminal |
| `--resume, -r` | Checkpoint pickle to resume from | none |
| `--generations, -g` | Override max generations | from config |

## Simulation Features

- 3D Newtonian gravity around Kerbin (KSP Earth analog)
- Exponential atmosphere model with drag
- Stage management with mass jettisoning
- Attitude control (pitch/yaw/roll)
- SAS auto-stabilization toggle
- Landing gear and brakes

## Observation Vector (16 features)

Altitude, velocity, flight path angle, pitch, yaw, throttle, stage count, fuel ratio, mass, northing/easting, radial velocity, estimated apoapsis/periapsis, SAS status.

## Action Vector (8 outputs)

Throttle, pitch offset, yaw offset, roll rate, SAS toggle, stage fire, gear toggle, brake force.

## Training Pipeline

1. Initialize random population of neural networks
2. Evaluate each agent through multiple episodes in the environment
3. Record fitness (altitude achieved + orbital velocity bonus - fuel penalty)
4. Evolve: tournament selection -> crossover -> Gaussian mutation -> elitism
5. Save checkpoint every N generations
6. Visualize live metrics via terminal or Streamlit dashboard

## Live KSP Mode

To train against the actual game, install the kRPC mod in KSP and run with `--env ksp`. See https://krpc.github.io for setup instructions.

## Roadmap

- NEAT integration (evolving network topologies)
- Multi-objective fitness (apoapsis, circularity, fuel efficiency)
- GPU-accelerated evaluation
- KSP2 compatibility
- Pre-trained agent packs
- Distributed training support

## License

MIT, see [`LICENSE`](LICENSE).
