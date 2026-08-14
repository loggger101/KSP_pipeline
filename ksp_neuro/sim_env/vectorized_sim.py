"""Vectorized orbital simulator — runs N instances simultaneously.

This module implements a batched version of the orbital mechanics simulator
that operates on numpy arrays instead of individual objects. This enables
running hundreds of parallel training episodes in a single function call,
providing 10-50x speedup over loop-based evaluation.

Usage:
    >>> sim = VectorizedOrbitSimulator(n_agents=200)
    >>> obs = sim.reset()           # shape (n_agents, 16)
    >>> for _ in range(max_steps):
    ...     actions = policy(obs)   # neural network forward pass
    ...     obs, rewards, dones, info = sim.step(actions)

Performance notes:
    - Vectorized physics step: O(n_agents * steps) → single numpy op batch
    - Memory efficient: all state stored in contiguous arrays
    - Compatible with the single-agent OrbitSimulator API for testing
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Constants — Kerbin values (same as orbit_sim.py)
# ---------------------------------------------------------------------------
_KERBIN_MU = 3.5316e12          # GM in m^3/s^2
_KERBIN_RADIUS = 600_000.0      # surface radius [m]
_KERBIN_ATM_MAX_ALT = 70_000.0  # atmosphere height [m]
_KERBIN_SEA_PRESSURE = 90_000.0 # Pa at sea level
_KERBIN_SCALE_HEIGHT = 5_600.0   # atmospheric scale height [m]

# Default vessel parameters (shared across all agents)
_DEFAULT_DRY_MASS = 12_000.0       # kg
_DEFAULT_FUEL_MASS = 38_000.0      # kg
_DEFAULT_THRUST_STAGE = [400.0, 250.0]   # kN per stage
_DEFAULT_ISP = 320.0               # seconds (exhaust velocity / g0)
_DEFAULT_DRAG_COEFF = 0.6
_DEFAULT_CROSS_SECTION = 12.0      # m^2


# ---------------------------------------------------------------------------
# Observation / action specs
# ---------------------------------------------------------------------------
_OBS_KEYS = [
    "alt_surface", "vel_magnitude", "flight_path_angle",
    "pitch", "yaw", "roll", "throttle", "stage_count",
    "fuel_ratio", "mass_kg", "apoapsis_est", "periapsis_est",
    "radial_vel", "sas_on", "gear_down", "vel_norm_proxy",
]

_ACTION_KEYS = [
    "throttle", "pitch_offset", "yaw_offset", "roll_offset",
    "sas_toggle", "stage_fire", "gear_toggle", "brake_force",
]


# ---------------------------------------------------------------------------
# Vectorized simulator class
# ---------------------------------------------------------------------------

class VectorizedOrbitSimulator:
    """Batch orbital mechanics simulator for parallel NE training.

    All N agents share the same physics parameters but have independent
    state vectors (position, velocity, attitude, fuel, etc.).
    The entire simulation is vectorized — no Python loops over agents.

    Parameters
    ----------
    n_agents : int
        Number of simultaneous simulation instances.
    dt : float
        Physics timestep in seconds.
    max_steps : int
        Maximum steps per episode before forced termination.
    seed : int | None
        Base random seed for reproducibility.
    """

    def __init__(
        self,
        n_agents: int = 200,
        dt: float = 0.1,
        max_steps: int = 3600,
        seed: int | None = None,
    ):
        self.n_agents = n_agents
        self.dt = dt
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        # ---- per-agent state vectors (shape [n_agents]) --------------------
        self.pos = np.zeros((n_agents, 3), dtype=np.float64)   # inertial pos [m]
        self.vel = np.zeros((n_agents, 3), dtype=np.float64)   # inertial vel [m/s]

        self.mass = np.full(n_agents, _DEFAULT_DRY_MASS + _DEFAULT_FUEL_MASS, dtype=np.float64)
        self.fuel_remaining = np.full(n_agents, _DEFAULT_FUEL_MASS, dtype=np.float64)

        # attitude (Euler angles in radians)
        self.pitch = np.zeros(n_agents, dtype=np.float64)      # pitch from horizon
        self.yaw = np.full(n_agents, np.pi / 2, dtype=np.float64)  # default eastward
        self.roll = np.zeros(n_agents, dtype=np.float64)

        # control state (per-agent booleans stored as float/bool arrays)
        self.throttle = np.zeros(n_agents, dtype=np.float32)
        self.sas_on = np.zeros(n_agents, dtype=bool)
        self.gear_down = np.ones(n_agents, dtype=bool)
        self.brakes = np.zeros(n_agents, dtype=np.float32)

        # stage management
        n_stages = len(_DEFAULT_THRUST_STAGE)
        self.current_stage = np.zeros(n_agents, dtype=np.int32)
        self.stage_fired = [np.zeros(n_agents, dtype=bool) for _ in range(n_stages)]

        # episode tracking
        self.steps = 0
        self.done_mask = np.zeros(n_agents, dtype=bool)  # which agents are done
        self.info: Dict[str, Any] = {
            "apoapsis_alt": np.full(n_agents, -1.0),
            "periapsis_alt": np.full(n_agents, -1.0),
            "max_altitude": np.zeros(n_agents, dtype=np.float64),
            "total_distance": np.zeros(n_agents, dtype=np.float64),
            "fuel_used": np.zeros(n_agents, dtype=np.float64),
        }

    # ------------------------------------------------------------------
    @property
    def observation_space(self):
        return len(_OBS_KEYS), self.n_agents

    @property
    def action_space(self):
        return len(_ACTION_KEYS), "float32"

    # ------------------------------------------------------------------
    def reset(
        self,
        seed_offset: int = 0,
    ) -> np.ndarray:
        """Reset all agents to launchpad. Returns batched observations.

        Parameters
        ----------
        seed_offset : int
            Offset for per-agent random seeds (for varied initial conditions).

        Returns
        -------
        obs : np.ndarray  shape (n_agents, 16)
        """
        n = self.n_agents

        # Initialize positions on surface at equator (KSC-like location)
        r0 = _KERBIN_RADIUS * np.ones(n, dtype=np.float64)
        self.pos[:] = np.column_stack([r0, np.zeros(n), np.zeros(n)])

        # Initial orbital velocity from Kerbin rotation + random perturbation
        init_speed = 174.0 + self._rng.uniform(0, 5, size=n)
        self.vel[:] = np.column_stack([np.zeros(n), init_speed, np.zeros(n)])

        # Reset mass and fuel
        self.mass[:] = _DEFAULT_DRY_MASS + _DEFAULT_FUEL_MASS
        self.fuel_remaining[:] = _DEFAULT_FUEL_MASS
        self.current_stage.fill(0)
        for sf in self.stage_fired:
            sf.fill(False)

        # Reset attitude and control
        self.pitch[:] = np.radians(90.0)   # vertical at start
        self.yaw[:] = np.pi / 2             # eastward
        self.roll[:] = 0.0
        self.throttle.fill(0.0)
        self.sas_on.fill(False)
        self.gear_down.fill(True)
        self.brakes.fill(1.0)

        # Reset tracking
        self.steps = 0
        self.done_mask.fill(False)
        for k in self.info:
            if k == "apoapsis_alt" or k == "periapsis_alt":
                self.info[k].fill(-1.0)
            else:
                self.info[k].fill(0.0)

        return self._get_observations()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Execute one physics tick for all agents simultaneously.

        Parameters
        ----------
        actions : np.ndarray  shape (n_agents, 8)
            Control vector keyed by ``_ACTION_KEYS``.

        Returns
        -------
        obs : np.ndarray  shape (n_agents, 16)
        rewards : np.ndarray  shape (n_agents,)
        dones : np.ndarray  shape (n_agents,), bool — which agents have terminated
        info : dict with per-agent tracking arrays (copies for safety)
        """
        n = self.n_agents

        # ---- clamp actions to valid range ---------------------------------
        ctrl = np.clip(actions, -1.0, 1.0)

        throttle_raw = ctrl[:, 0]       # [0, 1]
        pitch_offset = ctrl[:, 1]       # [-1, +1] → rad/s
        yaw_offset = ctrl[:, 2]         # [-1, +1] → rad/s
        roll_offset = ctrl[:, 3]        # [-1, +1] → rad/s
        sas_toggle_raw = (ctrl[:, 4] > 0.5)   # bool mask
        stage_fire_mask = (ctrl[:, 5] > 0.5)  # bool mask
        gear_toggle_mask = (np.abs(ctrl[:, 6]) > 0.5)  # toggle when non-zero
        brake_force = ctrl[:, 7]

        # ---- update control state -----------------------------------------
        self.throttle = throttle_raw.astype(np.float32)

        # Attitude changes (only for agents not done)
        active = ~self.done_mask
        dt_active = self.dt * np.where(active, 1.0, 0.0)

        self.pitch += pitch_offset * dt_active * np.pi / 4   # ±45 deg/s max
        self.yaw += yaw_offset * dt_active * np.pi / 4
        self.roll += roll_offset * dt_active * np.pi / 2       # ±90 deg/s max

        # SAS toggle (only for active agents)
        sas_on = np.where(active, sas_toggle_raw & (throttle_raw > 0.01), False)
        self.sas_on |= sas_on

        # Gear toggle (flip state when toggled)
        gear_change = gear_toggle_mask & active
        self.gear_down ^= gear_change

        self.brakes = brake_force.astype(np.float32)

        # ---- fire stages (only for agents that haven't fired this stage) --
        n_stages = len(_DEFAULT_THRUST_STAGE)
        for i in range(n_stages):
            if i < len(self.stage_fired):
                to_fire = stage_fire_mask & active & ~self.stage_fired[i] & (self.current_stage == i)
                self._apply_stage_change(to_fire, i)

        # ---- physics step (vectorized over all agents) --------------------
        self._physics_step(throttle_raw, active)

        self.steps += 1
        new_dones = self._check_termination(active)
        self.done_mask |= new_dones

        obs = self._get_observations()
        rewards = self._compute_rewards(ctrl)

        return obs, rewards, self.done_mask.copy(), {k: v.copy() for k, v in self.info.items()}

    def close(self) -> None:
        """No-op — no external resources."""
        pass

    # ---- private helpers (vectorized physics) -----------------------------
    def _apply_stage_change(self, mask: np.ndarray, stage_idx: int) -> None:
        """Apply mass jettison for fired stages. Vectorized per-agent."""
        if stage_idx < len(_DEFAULT_THRUST_STAGE):
            # Each stage jettisons a portion of fuel (evenly split among stages)
            fuel_per_stage = _DEFAULT_FUEL_MASS / max(len(_DEFAULT_THRUST_STAGE), 1)
            self.fuel_remaining[mask] -= fuel_per_stage

        if stage_idx < len(self.stage_fired):
            self.stage_fired[stage_idx][mask] = True

        # Increment current_stage for agents that fired this stage
        if np.any(mask):
            self.current_stage[mask] += 1

    def _physics_step(
        self, throttle: np.ndarray, active: np.ndarray
    ) -> None:
        """Vectorized physics integration for all agents."""
        n = self.n_agents
        active_mask = active.astype(np.float64)[:, np.newaxis]  # [n, 1]

        r_mag = np.linalg.norm(self.pos[active], axis=1)  # shape [n_active]

        # Avoid division by zero
        safe_r = np.where(r_mag > 1e-6, r_mag, 1.0)
        active_pos = self.pos[active]  # [n_active, 3]
        active_vel = self.vel[active]  # [n_active, 3]

        # ---- gravitational acceleration (vectorized per-agent) --------------
        grav_acc = -_KERBIN_MU / (safe_r ** 3)[:, np.newaxis] * active_pos  # [n_active, 3]

        # ---- thrust for agents with throttle > 0 and fuel remaining ---------
        has_thrust_mask = (throttle[active] > 0.01) & (self.fuel_remaining[active] > 0)  # [n_active]
        n_fire = int(np.sum(has_thrust_mask))

        thrust_acc = np.zeros_like(active_pos)  # [n_active, 3]

        if n_fire > 0:
            fire_idx = np.where(has_thrust_mask)[0]  # indices into active array
            fire_throttle = throttle[active][fire_idx]  # throttle values for thrusting agents

            # Thrust in kN → N
            stage_idx_arr = self.current_stage[active][fire_idx].astype(int)
            stage_idx_clamped = np.clip(stage_idx_arr, 0, len(_DEFAULT_THRUST_STAGE) - 1)
            thrust_N = (np.array(_DEFAULT_THRUST_STAGE)[stage_idx_clamped] * 1_000.0 * fire_throttle).astype(np.float64)

            # Mass flow rate: m_dot = T / (Isp * g0) — [n_fire,]
            mdot = thrust_N / (_DEFAULT_ISP * 9.81)

            # Fuel consumption
            fuel_delta = np.minimum(mdot * self.dt, self.fuel_remaining[active][fire_idx])
            self.fuel_remaining[active][fire_idx] -= fuel_delta
            self.info["fuel_used"][active][np.where(has_thrust_mask)[0]] += fuel_delta

            # Thrust direction from ship attitude
            cos_p = np.cos(self.pitch[active][fire_idx])
            sin_p = np.sin(self.pitch[active][fire_idx])
            cos_y = np.cos(self.yaw[active][fire_idx])
            sin_y = np.sin(self.yaw[active][fire_idx])

            thrust_dir = np.stack([cos_p * cos_y, cos_p * sin_y, sin_p], axis=1)  # [n_fire, 3]
            mass_for_thrust = np.maximum(self.mass[active][fire_idx][:, np.newaxis], 1.0)
            thrust_acc[fire_idx] = (thrust_N[:, np.newaxis] / mass_for_thrust) * thrust_dir

        # ---- atmospheric drag (exponential model) ---------------------------
        alt_surface_active = r_mag - _KERBIN_RADIUS
        in_atmosphere = (alt_surface_active >= 0) & (alt_surface_active < _KERBIN_ATM_MAX_ALT)

        v_rel_mag_active = np.linalg.norm(active_vel, axis=1)  # [n_active]

        drag_acc = np.zeros_like(active_pos)
        if np.any(in_atmosphere):
            atm_alt = np.maximum(alt_surface_active[in_atmosphere], 0.0)
            rho = (_KERBIN_SEA_PRESSURE / (287.05 * 261.1)) * np.exp(-atm_alt / _KERBIN_SCALE_HEIGHT)

            drag_mag = (0.5 * rho[:, np.newaxis] * v_rel_mag_active[in_atmosphere][:, np.newaxis]**2
                        * _DEFAULT_DRAG_COEFF * _DEFAULT_CROSS_SECTION
                        / np.maximum(self.mass[active][in_atmosphere][:, np.newaxis], 1.0))

            drag_vel = active_vel[in_atmosphere]
            safe_vmag = np.maximum(v_rel_mag_active[in_atmosphere][:, np.newaxis], 1e-6)
            drag_acc[in_atmosphere] = -drag_mag * (drag_vel / safe_vmag)

        # ---- combine accelerations ------------------------------------------
        accel = grav_acc + thrust_acc + drag_acc

        # ---- Euler integration for active agents ----------------------------
        dt_active_vec = self.dt * np.where(active, 1.0, 0.0)[:, np.newaxis]  # [n, 1]
        self.vel[active] += accel * dt_active_vec[:len(accel)]

        # Ground collision check (clamp to surface)
        r_mag_all = np.linalg.norm(self.pos, axis=1)  # [n_agents]
        below_surface = r_mag_all < _KERBIN_RADIUS

        if np.any(below_surface):
            normal = self.pos[below_surface] / np.maximum(r_mag_all[below_surface][:, np.newaxis], 1e-6)
            v_dot_n = np.sum(self.vel[below_surface] * normal, axis=1)
            bounce_mask = v_dot_n < 0
            if np.any(bounce_mask):
                bi = np.where(bounce_mask)[0]
                self.pos[below_surface][bi] += (-2.0 * v_dot_n[bi][:, np.newaxis] * normal[bi]) * 0.3

    def _check_termination(self, active: np.ndarray) -> np.ndarray:
        """Check which agents have terminated this step."""
        r_mag = np.linalg.norm(self.pos[active], axis=1)
        alt_surface = r_mag - _KERBIN_RADIUS

        # Ground collision (well below surface)
        crashed = alt_surface < -10.0

        # Escaped too far (> 5 million Kerbin radii)
        escaped = r_mag > 5e6 * _KERBIN_RADIUS

        return np.logical_or(crashed, escaped)

    def _get_observations(self) -> np.ndarray:
        """Build normalized observation vector for all agents."""
        n = self.n_agents
        r_mag = np.linalg.norm(self.pos, axis=1)  # [n]
        alt_surface = r_mag - _KERBIN_RADIUS
        speed = np.linalg.norm(self.vel, axis=1)    # [n]

        # Radial velocity (outward positive) — dot product of pos and vel directions
        pos_norm = self.pos / np.maximum(r_mag[:, np.newaxis], 1e-6)
        radial_vel = np.sum(self.vel * pos_norm, axis=1)  # [n]

        # Flight path angle: arccos(v·r̂ / |v|)
        cos_fpa = (self.vel * pos_norm).sum(axis=1) / (speed * r_mag + 1e-12)
        fpa = np.arccos(np.clip(cos_fpa, -1.0, 1.0))

        northing = self.pos[:, 2]
        easting = np.sqrt(self.pos[:, 0]**2 + self.pos[:, 1]**2)

        # Estimate apoapsis/periapsis (vis-viva approximation)
        mu = _KERBIN_MU
        a_est = np.where(
            (r_mag > 0) & (speed > 0),
            1.0 / (2.0 / r_mag - speed**2 / mu + 1e-30),
            1e6  # default large value for invalid orbits
        )

        eccentricity_term = (r_mag * speed / np.maximum(mu, 1e-6))**2
        apo_est = np.where(
            (eccentricity_term < 1) & (a_est > 0),
            a_est * (1 + np.sqrt(np.maximum(1 - eccentricity_term, 0))) - _KERBIN_RADIUS,
            0.0
        )
        peri_est = np.where(
            (eccentricity_term < 1) & (a_est > 0),
            a_est * (1 - np.sqrt(np.maximum(1 - eccentricity_term, 0))) - _KERBIN_RADIUS,
            0.0
        )

        # Normalize features to [0, ~2] range for stable NN input
        obs = np.column_stack([
            alt_surface / 1e6,
            speed / 3000,
            fpa / (np.pi + 1e-12),
            self.pitch / (np.pi + 1e-12),
            (self.yaw + np.pi) / (2 * np.pi + 1e-12),
            self.throttle.astype(np.float64),
            self.current_stage.astype(np.float64) / max(len(_DEFAULT_THRUST_STAGE), 1),
            self.fuel_remaining / _DEFAULT_FUEL_MASS,
            self.mass / (_DEFAULT_DRY_MASS + _DEFAULT_FUEL_MASS),
            northing / _KERBIN_RADIUS,
            easting / (_KERBIN_RADIUS * np.pi),
            radial_vel / 3000,
            apo_est / 1e6,
            peri_est / 1e6,
            self.sas_on.astype(np.float64),
            speed / (speed + 1),
        ])
        return obs.astype(np.float32)

    def _compute_rewards(self, action: np.ndarray) -> np.ndarray:
        """Multi-objective reward for all agents.

        Reward components:
            - altitude_bonus: encourages reaching higher altitudes
            - orbital_circularity: penalizes deviation from circular orbit velocity
            - fuel_penalty: small penalty for throttle usage (encourage efficiency)
            - stage_bonus: rewards completing stages
            - crash_penalty: heavy penalty for ground collision

        Returns
        -------
        rewards : np.ndarray  shape (n_agents,)
        """
        r_mag = np.linalg.norm(self.pos, axis=1)
        alt_surface = r_mag - _KERBIN_RADIUS
        speed = np.linalg.norm(self.vel, axis=1)

        # Primary: altitude achieved (encourage going higher)
        alt_reward = alt_surface / 1e6

        # Secondary: orbital velocity bonus (penalize deviation from circular orbit)
        v_circ = np.sqrt(_KERBIN_MU / r_mag)
        vel_ratio = speed / np.maximum(v_circ, 1.0)
        vel_reward = -np.abs(vel_ratio - 1.0) * 0.5

        # Fuel efficiency penalty (small — encourages not wasting fuel)
        fuel_penalty = -action[:, 0] * 0.01

        # Stage completion bonus (encourages staging to reduce dead weight)
        stage_bonus = self.current_stage.astype(np.float64) / max(len(_DEFAULT_THRUST_STAGE), 1) * 2.0

        # Crash penalty (heavy — encourages avoiding ground contact)
        crash_penalty = np.where(alt_surface < -5, -50.0, 0.0)

        return alt_reward + vel_reward + fuel_penalty + stage_bonus + crash_penalty


# Module-level alias for compatibility with existing code
OrbitSimulator = VectorizedOrbitSimulator

__all__ = ["VectorizedOrbitSimulator", "OrbitSimulator"]
