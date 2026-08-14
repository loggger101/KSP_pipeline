"""Real-time Streamlit dashboard for KSP Neuroevolution training metrics.

Run with:  ``streamlit run ksp_neuro/viz/dashboard.py --server.port 8501``

The dashboard polls a JSON file written by the trainer every few seconds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import streamlit as st
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    st.error("Install Plotly for charts: pip install plotly")
    raise


# ---------------------------------------------------------------------------
# Data loading — reads the JSON checkpoint file written by the trainer
# ---------------------------------------------------------------------------

def load_history(checkpoint_path: str = "./logs/ksp_neuro_history.json") -> Dict[str, List[float]]:
    """Load fitness history from the trainer's JSON log."""
    path = Path(checkpoint_path)
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return {k: list(v) for k, v in data.items()}
    except (json.JSONDecodeError, KeyError):
        return {}


# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------

def build_fitness_chart(history: Dict[str, List[float]]) -> go.Figure:
    """Best / Mean / Worst fitness over generations."""
    best = history.get("best_fitness", [])
    mean = history.get("mean_fitness", [])
    worst = history.get("worst_fitness", [])
    gens = list(range(1, len(best) + 1)) if best else [0]

    fig = make_subplots(rows=1, cols=1, subplot_titles=["Fitness Over Generations"])

    fig.add_trace(go.Scatter(x=gens, y=best, name="Best", line=dict(color="#2ecc71", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=gens, y=mean, name="Mean", line=dict(color="#f1c40f", width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=gens, y=worst, name="Worst", line=dict(color="#e74c3c", width=2, dash="dot")), row=1, col=1)

    fig.update_layout(
        title_text="Neuroevolution Fitness Progression",
        height=350,
        showlegend=True,
    )
    return fig


def build_std_chart(history: Dict[str, List[float]]) -> go.Figure:
    """Population diversity (std dev) over generations."""
    std = history.get("std_fitness", [])
    gens = list(range(1, len(std) + 1)) if std else [0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=std, name="Std Dev", line=dict(color="#3498db", width=2)))
    fig.update_layout(
        title_text="Population Diversity (Std Dev)",
        height=300,
        xaxis_title="Generation",
        yaxis_title="Fitness Std Dev",
    )
    return fig


def build_steps_chart(history: Dict[str, List[float]]) -> go.Figure:
    """Max steps survived per generation."""
    steps = history.get("episode_counts", [])
    gens = list(range(1, len(steps) + 1)) if steps else [0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=steps, name="Max Steps", line=dict(color="#9b59b6", width=2)))
    fig.update_layout(
        title_text="Agent Longevity (Max Steps Survived)",
        height=300,
        xaxis_title="Generation",
        yaxis_title="Steps",
    )
    return fig


# ---------------------------------------------------------------------------
# Main dashboard layout
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="KSP Neuroevolution Dashboard", layout="wide")
    st.title("🚀 KSP Neuroevolution Dashboard")

    # sidebar controls
    checkpoint_path = st.sidebar.text_input(
        "History JSON Path", value="./logs/ksp_neuro_history.json"
    )
    refresh_rate = st.sidebar.slider("Refresh Rate (s)", 1, 30, 5)

    st.divider()

    # auto-refresh loop
    placeholder = st.empty()

    while True:
        history = load_history(checkpoint_path)

        with placeholder.container():
            if not history or "best_fitness" not in history:
                st.info("Waiting for training data... Make sure the trainer is running and writing to the checkpoint file.")
                time.sleep(refresh_rate)
                continue

            best = history["best_fitness"]
            mean = history.get("mean_fitness", [])
            worst = history.get("worst_fitness", [])

            # summary cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Current Gen", len(best))
            with c2:
                st.metric("Best Fitness", f"{best[-1]:.4f}" if best else "N/A")
            with c3:
                st.metric("Mean Fitness", f"{mean[-1]:.4f}" if mean else "N/A")
            with c4:
                st.metric("Worst Fitness", f"{worst[-1]:.4f}" if worst else "N/A")

            # charts
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(build_fitness_chart(history), use_container_width=True)
            with col2:
                st.plotly_chart(build_std_chart(history), use_container_width=True)

            st.plotly_chart(build_steps_chart(history), use_container_width=True)

        time.sleep(refresh_rate)


if __name__ == "__main__":
    main()
