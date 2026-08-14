"""Visualization modules for monitoring NE training in real time.

Provides three visualization backends:
    1. ``TerminalPlotter`` — ASCII / ANSI progress bars (no GUI needed)
    2. ``MatplotlibPlotter`` — interactive matplotlib windows
    3. ``StreamlitDashboard`` — web-based dashboard (recommended for long runs)
"""

from __future__ import annotations

import abc
import time
from typing import Any, Dict, List


class BaseVizBackend(abc.ABC):
    """Abstract visualization backend interface."""

    @abc.abstractmethod
    def update_metrics(self, history: Dict[str, List[float]]) -> None:
        pass

    @abc.abstractmethod
    def log_episode(
        self, agent_id: int, reward: float, steps: int, info: Dict[str, Any]
    ) -> None:
        pass

    @abc.abstractmethod
    def show_best_agent(self, obs_history: List[List[float]], action_history: List[List[float]]) -> None:
        """Display the latest episode of the best agent."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Terminal plotter — ANSI progress bars + text metrics
# ---------------------------------------------------------------------------

class TerminalPlotter(BaseVizBackend):
    """Lightweight terminal-based visualization using ANSI escape codes.

    No GUI or web server needed — ideal for headless / SSH sessions.
    """

    def __init__(self, update_interval_s: float = 2.0):
        self._update_interval = update_interval_s
        self._last_update = 0.0
        self._best_fitness_history: List[float] = []
        self._mean_fitness_history: List[float] = []

    def _needs_update(self) -> bool:
        return time.time() - self._last_update >= self._update_interval

    def update_metrics(self, history: Dict[str, List[float]]) -> None:
        if not self._needs_update():
            return

        best = history.get("best_fitness", [])
        mean = history.get("mean_fitness", [])
        worst = history.get("worst_fitness", [])
        std = history.get("std_fitness", [])

        gen = len(best) - 1 if best else 0

        # ANSI color codes (works in most modern terminals)
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m"
        MAGENTA = "\033[95m"
        RESET = "\033[0m"

        bar_width = 40

        def _bar(value, max_val):
            if max_val == 0:
                return "[" + " " * bar_width + "]"
            filled = int(bar_width * min(abs(value) / abs(max_val), 1.0))
            filled = max(0, min(bar_width, filled))
            return "[" + "#" * filled + "-" * (bar_width - filled) + "]"

        if gen % 10 == 0 or gen < 5:  # print every 10 gens, first 5 always
            print("\n" + "=" * 70)
            print(f"  Generation {gen:>6d}  |  Time: {time.strftime('%H:%M:%S')}")
            print("-" * 70)

            if best:
                b = best[-1]
                m = mean[-1] if mean else 0
                w = worst[-1] if worst else 0
                s = std[-1] if std else 0

                ref = max(abs(b), abs(m), abs(w), 1e-6) * 1.1
                print(f"Best: {b:>12.4f}  {_bar(b, ref)}")
                print(f"Mean: {m:>12.4f}  {_bar(m, ref)}")
                print(f"Worst:{w:>12.4f}  {_bar(w, ref)}")
                print(f"Std:  {s:>12.4f}")

            self._last_update = time.time()

    def log_episode(
        self, agent_id: int, reward: float, steps: int, info: Dict[str, Any]
    ) -> None:
        pass  # suppressed in terminal mode for performance

    def show_best_agent(self, obs_history: List[List[float]], action_history: List[List[float]]) -> None:
        print(f"  Best agent episode — {len(obs_history)} steps")
        if obs_history and len(obs_history) > 0:
            last_obs = obs_history[-1]
            last_act = action_history[-1] if action_history else []
            keys = ["alt", "vel", "fpa", "pitch", "yaw", "throt", "fuel"]
            for k, o, a in zip(keys[:min(len(last_obs), len(last_act))], last_obs, last_act):
                print(f"    {k:6s}: obs={o:8.4f}  act={a:8.4f}")

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Matplotlib plotter — interactive window (optional dependency)
# ---------------------------------------------------------------------------

class MatplotlibPlotter(BaseVizBackend):
    """Interactive matplotlib-based visualization with rolling plots."""

    def __init__(self, update_interval_s: float = 2.0, history_size: int = 100):
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for safety
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        self._fig, self._axes = plt.subplots(2, 2, figsize=(14, 10))
        self._update_interval = update_interval_s
        self._history_size = history_size
        self._last_update = 0.0

        # fitness plot (top-left)
        ax_fitness = self._axes[0, 0]
        ax_fitness.set_title("Fitness Over Generations")
        ax_fitness.set_xlabel("Generation")
        ax_fitness.set_ylabel("Fitness")
        self._line_best, = ax_fitness.plot([], [], "g-", lw=2, label="Best")
        self._line_mean, = ax_fitness.plot([], [], "y--", lw=1.5, label="Mean")
        self._line_worst, = ax_fitness.plot([], [], "r:", lw=1.5, label="Worst")
        ax_fitness.legend()
        ax_fitness.grid(True, alpha=0.3)

        # std plot (top-right)
        ax_std = self._axes[0, 1]
        ax_std.set_title("Population Std Deviation")
        ax_std.set_xlabel("Generation")
        ax_std.set_ylabel("Std Dev")
        self._line_std, = ax_std.plot([], [], "b-", lw=2)
        ax_std.grid(True, alpha=0.3)

        # episode steps (bottom-left)
        ax_steps = self._axes[1, 0]
        ax_steps.set_title("Max Steps Survived")
        ax_steps.set_xlabel("Generation")
        ax_steps.set_ylabel("Steps")
        self._line_steps, = ax_steps.plot([], [], "m-", lw=2)
        ax_steps.grid(True, alpha=0.3)

        # rollout plot (bottom-right) — latest best agent episode
        ax_rollout = self._axes[1, 1]
        ax_rollout.set_title("Best Agent: Latest Episode")
        ax_rollout.set_xlabel("Step")
        ax_rollout.set_ylabel("Value")
        self._lines_rollout = {}

        plt.tight_layout()

    def update_metrics(self, history: Dict[str, List[float]]) -> None:
        if not self._needs_update():
            return

        import matplotlib
        matplotlib.use("Agg")  # ensure non-interactive

        best = history.get("best_fitness", [])[-self._history_size:]
        mean = history.get("mean_fitness", [])[-self._history_size:]
        worst = history.get("worst_fitness", [])[-self._history_size:]
        std = history.get("std_fitness", [])[-self._history_size:]
        steps = history.get("episode_counts", [])[-self._history_size:]

        gens = list(range(1, len(best) + 1))

        self._line_best.set_data(gens, best)
        self._line_mean.set_data(gens, mean)
        self._line_worst.set_data(gens, worst)
        self._line_std.set_data(gens, std)
        self._line_steps.set_data(gens, steps)

        # auto-scale
        for ax in self._axes.flat:
            ax.relim()
            ax.autoscale_view()

        self._fig.canvas.draw_idle()
        try:
            self._fig.canvas.flush_events()
        except Exception:
            pass  # non-interactive backend may not support this

        self._last_update = time.time()

    def log_episode(
        self, agent_id: int, reward: float, steps: int, info: Dict[str, Any]
    ) -> None:
        pass  # handled in show_best_agent

    def show_best_agent(self, obs_history: List[List[float]], action_history: List[List[float]]) -> None:
        if not obs_history:
            return

        import matplotlib
        matplotlib.use("Agg")

        ax_rollout = self._axes[1, 1]
        # clear previous
        for line in list(self._lines_rollout.values()):
            line.remove()
        self._lines_rollout.clear()

        steps_range = range(len(obs_history))
        obs_keys = ["alt_surface", "vel_magnitude", "throttle", "fuel_ratio"]

        colors = ["g", "b", "r", "m"]
        for key, color in zip(obs_keys, colors):
            vals = [o[idx] if idx < len(o) else 0 for o in obs_history]
            line, = ax_rollout.plot(steps_range, vals, color=color, alpha=0.7, label=key)
            self._lines_rollout[key] = line

        ax_rollout.legend()
        ax_rollout.relim()
        ax_rollout.autoscale_view()
        try:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        except Exception:
            pass

    def close(self) -> None:
        import matplotlib.pyplot as plt
        plt.close("all")

    def _needs_update(self) -> bool:
        return time.time() - self._last_update >= self._update_interval


# ---------------------------------------------------------------------------
# Streamlit dashboard — web-based (recommended for long training runs)
# ---------------------------------------------------------------------------

class StreamlitDashboard(BaseVizBackend):
    """Web-based real-time visualization via Streamlit.

    Usage:
        >>> viz = StreamlitDashboard(port=8501, history_size=200)
        >>> # In another terminal: streamlit run dashboard.py --server.port 8501
        >>> ... during training call viz.update_metrics(history) periodically
    """

    def __init__(self, port: int = 8501, history_size: int = 200):
        self._port = port
        self._history_size = history_size
        self._latest_history: Dict[str, List[float]] = {}
        self._best_episode_obs: List[List[float]] = []
        self._best_episode_act: List[List[float]] = []

    def update_metrics(self, history: Dict[str, List[float]]) -> None:
        """Update the dashboard with latest metrics.

        For live use, call this periodically during training. The dashboard
        will poll for updates via a shared data store (JSON file or memory).
        """
        self._latest_history = {k: v[-self._history_size:] for k, v in history.items()}

    def log_episode(
        self, agent_id: int, reward: float, steps: int, info: Dict[str, Any]
    ) -> None:
        pass  # stored alongside fitness data

    def show_best_agent(self, obs_history: List[List[float]], action_history: List[List[float]]) -> None:
        if obs_history:
            self._best_episode_obs = obs_history[-100:]  # last 100 steps
            self._best_episode_act = action_history[-100:] if action_history else []

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Factory — create the right backend based on config / environment
# ---------------------------------------------------------------------------

def get_viz_backend(
    mode: str = "terminal",
    update_interval_s: float = 2.0,
    dashboard_port: int = 8501,
    history_size: int = 100,
) -> BaseVizBackend:
    """Create a visualization backend by name.

    Parameters
    ----------
    mode : str
        ``"terminal"``, ``"matplotlib"``, or ``"streamlit"``.
    update_interval_s : float
        Seconds between metric updates.
    dashboard_port : int
        Port for Streamlit / matplotlib web interface.
    history_size : int
        Rolling window size for plots.

    Returns
    -------
    BaseVizBackend
    """
    if mode == "terminal":
        return TerminalPlotter(update_interval_s=update_interval_s)
    elif mode == "matplotlib":
        return MatplotlibPlotter(update_interval_s=update_interval_s, history_size=history_size)
    elif mode == "streamlit":
        return StreamlitDashboard(port=dashboard_port, history_size=history_size)
    else:
        raise ValueError(f"Unknown viz mode: {mode}. Use 'terminal', 'matplotlib', or 'streamlit'.")


__all__ = [
    "BaseVizBackend",
    "TerminalPlotter",
    "MatplotlibPlotter",
    "StreamlitDashboard",
    "get_viz_backend",
]
