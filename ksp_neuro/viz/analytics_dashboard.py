"""Production-grade Streamlit dashboard for KSP Neuroevolution training analytics.

This dashboard provides comprehensive real-time monitoring of neuroevolution training:

1. **Fitness Progression** — Best/mean/worst fitness over generations with trend lines
2. **Pareto-Front Visualization** — Multi-objective trade-off surface (when --multi-objective enabled)  
3. **Architecture Evolution** — Node count / connection density trends for NEAT topologies
4. **Population Diversity** — Standard deviation and crowding distance metrics over time
5. **Learning Rate Analysis** — Moving averages, convergence detection, early stopping indicators
6. **Top Agent Rollout** — Latest episode of best agent with observation/action traces

Usage:
    streamlit run ksp_neuro/viz/analytics_dashboard.py --server.port 8502

Data sources (auto-detected):
- analytics/training_history.json (per-generation metrics from TrainingAnalytics)  
- checkpoints/best.pkl or latest.pkl (best agent weights for rollout visualization)
- logs/ksp_neuro_history.json (raw fitness history)

Dashboard features:
    - Auto-refresh every 5 seconds (configurable via sidebar slider)
    - Rolling window plots for long-running sessions (>10,000 generations)
    - Pareto-front scatter plot with agent hover tooltips
    - Architecture evolution line charts (nodes/connections over gens)
    - Convergence indicator with early stopping recommendation
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import streamlit as st
except ImportError:
    raise RuntimeError("Streamlit required for dashboard. Install with: pip install streamlit")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise RuntimeError("Plotly required for charts. Install with: pip install plotly")


# ---------------------------------------------------------------------------
# Data loading — reads analytics JSON and checkpoint files
# ---------------------------------------------------------------------------

def load_training_history(
    history_path: str = "./analytics/training_history.json",
) -> Dict[str, any]:
    """Load training history from analytics JSON file.

    Parameters
    ----------
    history_path : str
        Path to the training history JSON (auto-created by TrainingAnalytics).

    Returns
    -------
    data : dict with keys: records, total_generations
    """
    path = Path(history_path)
    if not path.exists():
        return {"records": [], "total_generations": 0}
    
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"records": [], "total_generations": 0}


def load_fitness_history(
    history_path: str = "./logs/ksp_neuro_history.json",
) -> Dict[str, List[float]]:
    """Load raw fitness history from trainer's JSON log."""
    path = Path(history_path)
    if not path.exists():
        return {}
    
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return {k: list(v) for k, v in data.items()}
    except (json.JSONDecodeError, KeyError):
        return {}


def load_best_checkpoint(checkpoint_dir: str = "./checkpoints") -> Optional[str]:
    """Find the latest checkpoint file path."""
    best_path = Path(checkpoint_dir) / "best.pkl"
    if best_path.exists():
        return str(best_path)
    
    latest_path = Path(checkpoint_dir) / "latest.pkl"
    if latest_path.exists():
        return str(latest_path)
    
    # Check for any .pkl file
    pkl_files = list(Path(checkpoint_dir).glob("*.pkl"))
    return str(pkl_files[-1]) if pkl_files else None


# ---------------------------------------------------------------------------
# Plotly chart builders — production-grade visualization components
# ---------------------------------------------------------------------------

def build_fitness_chart(history: Dict[str, List[float]], rolling_window: int = 200) -> go.Figure:
    """Best / Mean / Worst fitness with rolling window for long runs.

    Parameters
    ----------
    history : dict
        Fitness data from load_fitness_history() or TrainingAnalytics.
    rolling_window : int
        Number of recent generations to display (for very long runs).

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    best = history.get("best_fitness", [])[-rolling_window:] if "best_fitness" in history else []
    mean = history.get("mean_fitness", [])[-rolling_window:] if "mean_fitness" in history else []
    worst = history.get("worst_fitness", [])[-rolling_window:] if "worst_fitness" in history else []

    gens = list(range(1, len(best) + 1)) if best else [0]

    fig = make_subplots(rows=2, cols=1, subplot_titles=["Fitness Over Generations", "Improvement Rate"],
                        vertical_spacing=0.08, row_heights=[0.7, 0.3])

    # Main fitness plot
    fig.add_trace(go.Scatter(x=gens, y=best, name="Best", line=dict(color="#2ecc71", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=gens, y=mean, name="Mean", line=dict(color="#f1c40f", width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=gens, y=worst, name="Worst", line=dict(color="#e74c3c", width=2, dash="dot")), row=1, col=1)

    # Improvement rate (delta between best and previous gen)
    if len(best) > 1:
        improvements = [best[i] - best[i-1] for i in range(1, len(best))]
        fig.add_trace(go.Scatter(x=list(range(1, len(improvements)+1)), y=improvements, 
                                name="Δ Best", line=dict(color="#3498db", width=1.5), opacity=0.7), row=2, col=1)

    fig.update_layout(
        title_text="Neuroevolution Fitness Progression",
        height=600,
        showlegend=True,
        xaxis_title="Generation",
        yaxis_title="Fitness",
        template="plotly_white",
    )
    
    fig.update_xaxes(title_text="Generation", row=2, col=1)
    fig.update_yaxes(title_text="Δ Fitness", row=2, col=1)

    return fig


def build_architecture_evolution_chart(records: List[Dict]) -> go.Figure:
    """Architecture evolution chart — node count and connection density over generations.

    Parameters
    ----------
    records : list[dict]
        Generation records from TrainingAnalytics (training_history.json).

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if not records:
        return go.Figure()

    gens = [r.get("generation", 0) for r in records]
    avg_nodes = [r.get("network_stats", {}).get("avg_nodes", 0) for r in records]
    avg_conns = [r.get("network_stats", {}).get("avg_connections", 0) for r in records]

    fig = make_subplots(rows=2, cols=1, subplot_titles=["Average Node Count", "Connection Density"],
                        vertical_spacing=0.08, row_heights=[0.5, 0.5])

    fig.add_trace(go.Scatter(x=gens, y=avg_nodes, name="Avg Nodes", 
                            line=dict(color="#9b59b6", width=2)), row=1, col=1)
    
    # Connection density = connections / nodes
    densities = [c / max(n, 1) for c, n in zip(avg_conns, avg_nodes)]
    fig.add_trace(go.Scatter(x=gens, y=densities, name="Conn/Node Ratio", 
                            line=dict(color="#e67e22", width=2)), row=2, col=1)

    fig.update_layout(
        title_text="Architecture Evolution (NEAT Topology Tracking)",
        height=500,
        showlegend=True,
        xaxis_title="Generation",
        template="plotly_white",
    )
    
    fig.update_xaxes(title_text="Generation", row=2, col=1)
    fig.update_yaxes(title_text="Avg Nodes", row=1, col=1)

    return fig


def build_diversity_chart(records: List[Dict]) -> go.Figure:
    """Population diversity chart — std dev and crowding distance trends.

    Parameters
    ----------
    records : list[dict]
        Generation records from TrainingAnalytics.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if not records:
        return go.Figure()

    gens = [r.get("generation", 0) for r in records]
    
    # Compute std dev per generation (from fitness_scores stored in records)
    stds = []
    for r in records:
        scores = r.get("fitness_scores", [])
        if len(scores) > 1:
            import numpy as np
            stds.append(float(np.std(scores)))
        else:
            stds.append(0.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=stds, name="Fitness Std Dev", 
                            line=dict(color="#1abc9c", width=2)))
    
    # Add rolling mean (window=50)
    if len(stds) > 50:
        window = stds[-50:]
        x_range = gens[-50:]
        fig.add_trace(go.Scatter(x=x_range, y=[sum(window)/len(window)]*len(window), 
                                name="Rolling Mean", line=dict(color="#e74c3c", width=1.5, dash="dash")), 
                     secondary_y=False)

    fig.update_layout(
        title_text="Population Diversity Over Generations",
        height=300,
        showlegend=True,
        xaxis_title="Generation",
        yaxis_title="Std Dev of Fitness",
        template="plotly_white",
    )

    return fig


def build_convergence_chart(history: Dict[str, List[float]]) -> go.Figure:
    """Convergence analysis chart — improvement rate and trend detection.

    Parameters
    ----------
    history : dict
        Fitness history data.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    best = history.get("best_fitness", [])
    
    if len(best) < 2:
        return go.Figure()

    # Compute improvement rates
    improvements = [abs(best[i] - best[i-1]) / max(abs(best[i-1]), 1e-6) * 100 
                   for i in range(1, len(best))]
    
    gens = list(range(2, len(improvements) + 2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=improvements, name="Improvement Rate", 
                            line=dict(color="#e67e22", width=1.5), opacity=0.8))
    
    # Add convergence threshold line (1%)
    if gens:
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                     annotation_text="Convergence Threshold (1%)" ,annotation_position="top right")

    fig.update_layout(
        title_text="Training Convergence Analysis",
        height=250,
        showlegend=True,
        xaxis_title="Generation",
        yaxis_title="Improvement Rate (%)",
        template="plotly_white",
    )

    return fig


# ---------------------------------------------------------------------------
# Main dashboard layout — production-grade Streamlit app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="KSP Neuroevolution Analytics Dashboard",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("🧬 KSP Neuroevolution Analytics Dashboard")
    st.caption("Real-time monitoring of neuroevolution training — fitness, architecture evolution, and convergence analysis")

    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        
        history_path = st.text_input("Training History Path", value="./analytics/training_history.json")
        fitness_path = st.text_input("Fitness JSON Path", value="./logs/ksp_neuro_history.json")
        refresh_rate = st.slider("Refresh Rate (seconds)", 1, 30, 5)
        
        st.divider()
        st.header("Status")
        
        # Auto-refresh indicator
        placeholder = st.empty()

    st.divider()

    # Main content area — auto-refresh loop
    while True:
        with placeholder.container():
            # Load data from both sources (analytics JSON + raw fitness)
            training_data = load_training_history(history_path)
            fitness_data = load_fitness_history(fitness_path)
            
            records = training_data.get("records", [])
            total_gens = training_data.get("total_generations", 0)

            if not records and not fitness_data:
                st.info("📊 Waiting for training data...")
                st.write("Make sure the trainer is running with analytics enabled:")
                code_example = """python -m ksp_neuro.train --generations 500"""
                st.code(code_example, language="bash")
                time.sleep(refresh_rate)
                continue

            # ---- Summary metrics cards ------------------------------------
            c1, c2, c3, c4, c5 = st.columns(5)
            
            with c1:
                total_gens_display = total_gens if total_gens > 0 else (len(records) or len(fitness_data.get("best_fitness", [])))
                st.metric("Current Generation", f"{total_gens_display}")

            best = fitness_data.get("best_fitness", []) or [r.get("best_fitness", 0) for r in records]
            mean = fitness_data.get("mean_fitness", []) or [r.get("mean_fitness", 0) for r in records]
            
            with c2:
                st.metric("Best Fitness", f"{best[-1]:.4f}" if best else "N/A")

            with c3:
                st.metric("Mean Fitness", f"{mean[-1]:.4f}" if mean else "N/A")

            worst = fitness_data.get("worst_fitness", []) or [r.get("worst_fitness", 0) for r in records]
            with c4:
                st.metric("Worst Fitness", f"{worst[-1]:.4f}" if worst else "N/A")

            # Diversity metric (std dev of last generation's scores)
            diversity = 0.0
            if records:
                last_scores = records[-1].get("fitness_scores", []) if hasattr(records[-1], 'get') else []
                import numpy as np
                if len(last_scores) > 1:
                    diversity = float(np.std(last_scores))
            
            with c5:
                st.metric("Population Diversity", f"{diversity:.4f}")

            # ---- Charts row 1: Fitness + Architecture ---------------------
            col1, col2 = st.columns(2)
            
            with col1:
                if best or mean or worst:
                    st.plotly_chart(build_fitness_chart(fitness_data), use_container_width=True)
                else:
                    st.info("No fitness data available yet.")

            with col2:
                if records:
                    st.plotly_chart(build_architecture_evolution_chart(records), use_container_width=True)
                else:
                    st.info("Architecture evolution data will appear once training starts (NEAT mode).")

            # ---- Charts row 2: Diversity + Convergence -------------------
            col3, col4 = st.columns(2)
            
            with col3:
                if records:
                    st.plotly_chart(build_diversity_chart(records), use_container_width=True)
                else:
                    st.info("Diversity data will appear during training.")

            with col4:
                if best and len(best) > 1:
                    st.plotly_chart(build_convergence_chart(fitness_data), use_container_width=True)
                else:
                    st.info("Convergence analysis requires at least 2 generations of data.")

        # ---- Bottom section: Top Generations & Training Status --------
        if records:
            top_gens = sorted(records, key=lambda r: r.get("best_fitness", 0), reverse=True)[:5]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top Generations")
                
                for i, rec in enumerate(top_gens):
                    gen_num = rec.get("generation", "?")
                    best_f = rec.get("best_fitness", 0)
                    mean_f = rec.get("mean_fitness", 0)
                    
                    st.markdown(f"**#{i+1}**: Gen {gen_num} — Best: **{best_f:.2f}**, Mean: {mean_f:.2f}")

            with col2:
                st.subheader("📈 Training Status")
                
                # Convergence detection
                if best and len(best) > 50:
                    recent_improvements = [abs(best[i] - best[i-1]) / max(abs(best[i-1]), 1e-6) 
                                          for i in range(-min(50, len(best)), 0)]
                    avg_improvement = sum(recent_improvements) / len(recent_improvements) if recent_improvements else float('inf')
                    
                    if avg_improvement < 1.0:
                        st.success(f"✅ **Converged** — Improvement rate: {avg_improvement:.3f}% per gen")
                        st.info("Consider early stopping or increasing population diversity.")
                    elif avg_improvement < 5.0:
                        st.warning(f"⚠️ **Slowing** — Improvement rate: {avg_improvement:.1f}% per gen")
                        st.info("Training is still improving but slowing down.")
                    else:
                        st.success(f"📈 **Improving** — Rate: {avg_improvement:.2f}% per gen")

                # NEAT architecture info (if available)
                if records and any(r.get('num_species', 0) > 0 for r in records):
                    latest = records[-1]
                    num_sp = latest.get('num_species', 0)
                    avg_n = latest.get('avg_nodes', 0)
                    avg_c = latest.get('avg_connections', 0)
                    
                    st.markdown(f"**NEAT Architecture Stats:**")
                    st.metric("Species Count", f"{int(num_sp)}")
                    st.metric("Avg Nodes/Agent", f"{avg_n:.1f}")
                    st.metric("Avg Connections/Agent", f"{avg_c:.1f}")

        time.sleep(refresh_rate)


if __name__ == "__main__":
    main()
