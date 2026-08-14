"""Orbital mechanics simulator — lightweight physics for parallel NE training.

This module implements a simplified but physically meaningful 3D orbital
simulator targeting Kerbin (KSP's Earth analog).  It is designed to be:

- **Fast**: pure NumPy, no external dependencies beyond numpy/scipy
- **Parallelizable**: identical API whether running one or N instances
- **Configurable**: gravity, atmosphere, and initial conditions via config

Physics model:
    - Point-mass central gravity (Newtonian)
    - Exponential atmospheric drag model
    - Simplified thrust vectoring (3 DOF attitude + throttle)
    - Stage-based mass changes
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants — Kerbin values (KSP approximations, SI units)
# ---------------------------------------------------------------------------
_KERBIN_MU = 3.5316e12          # GM in m^3/s^2 (μ = G * M)
_KERBIN_RADIUS = 600_000.0      # surface radius in meters
_KERBIN_ATM_MAX_ALT = 70_000.0  # atmosphere height in meters
_KERBIN_SEA_PRESSURE = 90_000.0 # Pa at sea level
_KERBIN_SCALE_HEIGHT = 5_600.0   # atmospheric scale height (m)


# ---------------------------------------------------------------------------
# Vessel parameters (default "basic rocket")
# ---------------------------------------------------------------------------
@dataclass
class _VesselParams:
    """Fixed properties of the default test vessel."""

    dry_mass_kg: float = 12_000.0       # structural mass
    fuel_mass_kg: float = 38_000.0      # propellant mass (initial)
    thrust_kn: float = 400.0            # max thrust in kN → N
    isp_s: float = 320.0                # specific impulse
    drag_coefficient: float = 0.6       # ballistic drag coefficient
    cross_section_m2: float = 12.0      # frontal area
    stages: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (38_000.0, 400.0),   # fuel mass, thrust for stage 1
            (0.0, 250.0),         # fairing/jettisoned mass, upper stage thrust
        ]
    )


# ---------------------------------------------------------------------------
# Observation / action specs
# ---------------------------------------------------------------------------
_OBS_KEYS = [
    "alt_surface",      # altitude above sea level [m]
    "vel_magnitude",    # speed magnitude [m/s]
    "flight_path_angle",  # angle between velocity and local horizon [-π, π]
    "pitch",            # ship pitch angle (0=horizontal) [rad]
    "yaw",              # ship yaw angle from north [rad]
    "throttle",         # current throttle setting [0, 1]
    "stage_count",      # remaining stages (integer → float)
    "fuel_ratio",       # fraction of fuel remaining [0, 1]
    "mass_total",       # total mass [kg] — normalized later
    "northing",         # distance north from launch site [m]
    "easting",          # distance east from launch site [m]
    "radial_vel",       # radial component of velocity [m/s] (outward +)
    "prograde_magnitude",  # prograde orbital speed approximation
    "apoapsis_est",     # estimated apoapsis altitude [m]
    "periapsis_est",    # estimated periapsis altitude [m]
    "sas_on",           # SAS status (0 or 1)
]

_ACTION_KEYS = [
    "throttle",         # throttle [0, 1]
    "pitch_offset",     # pitch change rate [-1, +1] → rad/s
    "yaw_offset",       # yaw change rate [-1, +1] → rad/s
    "roll_offset",      # roll rate [-1, +1] → rad/s
    "sas_toggle",       # SAS on/off (0 or 1)
    "stage_fire",       # fire next stage (0 or 1)
    "gear_toggle",      # landing gear (0 or 1)
    "brake_force",      # wheel brake force [0, 1]
]


# ---------------------------------------------------------------------------
# Simulator class
# ---------------------------------------------------------------------------
class OrbitSimulator:
    """Simplified orbital mechanics simulator for NE training.

    Each instance manages one vessel's state vector and physics step.
    Call :meth:`reset` to initialize, then :meth:`step` each tick.

    Parameters
    ----------
    config : Config | None
        Pipeline configuration object (from ``ksp_neuro.config.loader``).
    seed : int | None
        Random seed for reproducibility.
    initial_altitude_km : float
        Launch altitude above sea level in km (default 0 = surface).
    """

    def __init__(
        self,
        config: Any = None,
        seed: int | None = None,
        initial_altitude_km: float = 0.0,
    ):
        # defaults from module-level constants if no config
        if config is not None and hasattr(config, "sim_time_step"):
            self._dt = config.sim_time_step
        else:
            self._dt = 0.1

        self._rng = np.random.default_rng(seed)
        self._initial_alt_km = initial_altitude_km

        # vessel state (numpy arrays for speed in vectorized ops)
        self.pos = None       # [x, y, z] inertial frame, origin at Kerbin center [m]
        self.vel = None       # inertial velocity [m/s]
        self.mass = 0.0       # total mass [kg]
        self.fuel_remaining = 0.0  # remaining propellant [kg]

        # attitude (Euler angles: pitch, yaw, roll in radians)
        self.pitch = 0.0
        self.yaw = np.pi / 2  # start pointing east for launch
        self.roll = 0.0

        # control state
        self.throttle = 0.0
        self.sas_on = False
        self.gear_down = False
        self.brakes = 0.0
        self.current_stage = 0
        self.stage_fired = [False] * 2

        # tracking
        self.steps = 0
        self.done = False
        self.info: Dict[str, Any] = {}

        # vessel params (load from config or use defaults)
        if config is not None and hasattr(config, "sim_seed"):
            seed_val = config.sim_seed
        else:
            seed_val = 42
        self._vessel = _VesselParams()

        # max steps per episode (configurable for testing)
        self._max_steps = getattr(
            config, 'sim_max_steps', 36_000
        ) if hasattr(config, 'sim_max_steps') else 36_000

    # ------------------------------------------------------------------
    @property
    def dt(self) -> float:
        return self._dt

    @property
    def observation_space(self):
        return len(_OBS_KEYS), "float32"

    @property
    def action_space(self):
        return len(_ACTION_KEYS), "float32"

    # ------------------------------------------------------------------
    def reset(
        self,
        initial_altitude_km: float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Reset vessel to launchpad and return observation.

        Parameters
        ----------
        initial_altitude_km : optional
            Override default launch altitude.
        seed : optional
            Per-reset random seed for atmospheric turbulence.

        Returns
        -------
        obs : np.ndarray  shape (16,)
            Normalized observation vector keyed by ``_OBS_KEYS``.
        """
        alt_km = initial_altitude_km if initial_altitude_km is not None else self._initial_alt_km
        r0 = _KERBIN_RADIUS + alt_km * 1_000.0

        # launch from "KSC" position — equator, for simplicity
        self.pos = np.array([r0, 0.0, 0.0], dtype=np.float64)
        # initial orbital speed: Kerbin rotation ~174 m/s at equator;
        # give a small boost to simulate launch pad velocity
        init_speed = 174.0 + self._rng.uniform(0, 5)
        self.vel = np.array([0.0, init_speed, 0.0], dtype=np.float64)

        self.mass = self._vessel.dry_mass_kg + self._vessel.fuel_mass_kg
        self.fuel_remaining = self._vessel.fuel_mass_kg
        self.current_stage = 0
        self.stage_fired = [False] * len(self._vessel.stages)

        self.pitch = np.radians(90.0)   # vertical at start
        self.yaw = np.pi / 2            # east
        self.roll = 0.0
        self.throttle = 0.0
        self.sas_on = False
        self.gear_down = True
        self.brakes = 1.0

        self.steps = 0
        self.done = False
        self.info = {
            "apoapsis_alt": None,
            "periapsis_alt": None,
            "max_altitude": 0.0,
            "total_distance": 0.0,
            "fuel_used": 0.0,
            "stage_changes": [],
        }

        self._rng = np.random.default_rng(seed) if seed is not None else self._rng
        return self._get_observation()

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one physics tick.

        Parameters
        ----------
        action : np.ndarray  shape (8,)
            Clamped control vector keyed by ``_ACTION_KEYS``.

        Returns
        -------
        obs : np.ndarray
        reward : float
        done : bool
        info : dict
        """
        if self.done:
            raise RuntimeError("Cannot step a finished episode; call reset() first.")

        # clamp action to [-1, 1] range (some outputs are [0, 1])
        ctrl = np.clip(action, -1.0, 1.0)

        throttle_raw = float(ctrl[0])
        pitch_offset = float(ctrl[1])
        yaw_offset = float(ctrl[2])
        roll_offset = float(ctrl[3])
        sas_on = bool(int(round(ctrl[4])))
        stage_fire = bool(int(round(ctrl[5])))
        gear_toggle = bool(int(round(ctrl[6])))
        brake_force = float(ctrl[7])

        # ---- update control state ---------------------------------------
        self.throttle = throttle_raw  # already in [0,1] from clip
        self.pitch += pitch_offset * self._dt * np.pi / 4   # ±45 deg/s max
        self.yaw += yaw_offset * self._dt * np.pi / 4
        self.roll += roll_offset * self._dt * np.pi / 2

        if sas_on:
            self.sas_on = True
        elif ctrl[4] == 0 and self.throttle < 0.01:
            # only turn SAS off when not throttling (safety)
            self.sas_on = False

        if gear_toggle > 0.5:
            self.gear_down = not self.gear_down

        self.brakes = brake_force

        # stage fire (bounds-safe)
        if stage_fire and self.current_stage < len(self.stage_fired) and not self.stage_fired[self.current_stage]:
            self._fire_stage()
            self.info["stage_changes"].append(self.steps)

        # ---- physics step -----------------------------------------------
        self._physics_step(throttle_raw, pitch_offset, yaw_offset)

        self.steps += 1
        self.done = self._check_termination()
        obs = self._get_observation()
        reward = self._compute_reward(ctrl)

        return obs, reward, self.done, dict(self.info)

    # ------------------------------------------------------------------
    def close(self) -> None:
        """No-op for simulator (no external resources)."""
        pass

    # ---- private helpers --------------------------------------------------
    def _fire_stage(self):
        if self.current_stage >= len(self._vessel.stages):
            return
        fuel_lost, thrust_bonus = self._vessel.stages[self.current_stage]
        self.fuel_remaining -= max(fuel_lost, 0.0)
        self.mass += fuel_lost  # negative value jettisons mass
        self.stage_fired[self.current_stage] = True
        self.current_stage += 1

    def _physics_step(self, throttle: float, pitch_rate: float, yaw_rate: float):
        """Single RK2 (Heun) physics integration step."""
        # gravitational acceleration at current position
        r_mag = np.linalg.norm(self.pos)
        if r_mag < 1e-6:
            return  # safety clamp

        grav_acc = -_KERBIN_MU / (r_mag ** 3) * self.pos  # vector

        # thrust acceleration
        if throttle > 0.01 and self.fuel_remaining > 0:
            current_thrust_kn = 400.0 + (self.current_stage * 250.0)
            thrust_N = current_thrust_kn * 1_000.0 * throttle

            # mass flow rate: m_dot = T / (Isp * g0)
            mdot = thrust_N / (self._vessel.isp_s * 9.81)
            fuel_delta = min(mdot * self._dt, self.fuel_remaining)
            self.fuel_remaining -= fuel_delta
            self.info["fuel_used"] += fuel_delta

            # thrust direction from ship attitude
            cos_p = np.cos(self.pitch)
            sin_p = np.sin(self.pitch)
            cos_y = np.cos(self.yaw)
            sin_y = np.sin(self.yaw)

            # local frame: x=radial-out, y=tangential (prograde), z=north
            thrust_dir = np.array(
                [cos_p * cos_y, cos_p * sin_y, sin_p],
                dtype=np.float64,
            )
            thrust_acc = (thrust_N / max(self.mass, 1.0)) * thrust_dir

            # atmospheric drag
            alt_surface = r_mag - _KERBIN_RADIUS
            if alt_surface < 0:
                alt_surface = 0.0
            # atmospheric drag (exponential atmosphere model)
            if alt_surface < _KERBIN_ATM_MAX_ALT:
                rho = (_KERBIN_SEA_PRESSURE / (287.05 * 261.1)) * np.exp(-alt_surface / _KERBIN_SCALE_HEIGHT)
                v_rel_mag = float(np.linalg.norm(self.vel))
                drag_magnitude = (0.5 * rho * (v_rel_mag ** 2) * self._vessel.drag_coefficient
                                  * self._vessel.cross_section_m2 / max(self.mass, 1.0))
                if v_rel_mag > 0:
                    drag_acc = -drag_magnitude * self.vel / v_rel_mag
                else:
                    drag_acc = np.zeros(3)
            else:
                drag_acc = np.zeros(3)

            # total acceleration
            accel = grav_acc + thrust_acc + drag_acc
        else:
            accel = grav_acc  # coasting — gravity only

        # simple Euler integration (dt=0.1 is small enough for this sim)
        self.vel += accel * self._dt
        self.pos += self.vel * self._dt

        # ground collision check
        if r_mag < _KERBIN_RADIUS:
            self.pos = self.pos / r_mag * (_KERBIN_RADIUS + 0.1)
            # reflect velocity partially (bounce)
            normal = self.pos / np.linalg.norm(self.pos)
            v_dot_n = np.dot(self.vel, normal)
            if v_dot_n < 0:
                self.vel += (-2.0 * v_dot_n * normal) * 0.3  # damped bounce

    def _check_termination(self) -> bool:
        r_mag = np.linalg.norm(self.pos)
        alt_surface = r_mag - _KERBIN_RADIUS
        speed = np.linalg.norm(self.vel)

        # crashed / exploded
        if alt_surface < -10:  # well below surface
            return True

        # escaped too far (lost contact)
        if r_mag > 5_000_000 * _KERBIN_RADIUS:
            return True

        # max steps exceeded
        if self.steps >= self._max_steps:
            return True

        return False

    def _get_observation(self) -> np.ndarray:
        """Build normalized observation vector."""
        r_mag = np.linalg.norm(self.pos)
        alt_surface = max(r_mag - _KERBIN_RADIUS, 0.0)
        speed = float(np.linalg.norm(self.vel))
        radial_vel = float(np.dot(self.vel, self.pos / r_mag)) if r_mag > 0 else 0.0

        # flight path angle: angle between velocity and local horizontal plane
        cos_fpa = np.dot(self.vel, self.pos) / (speed * r_mag + 1e-12)
        fpa = float(np.arccos(np.clip(cos_fpa, -1, 1)))

        northing = self.pos[2] if len(self.pos) > 2 else 0.0
        easting = np.sqrt(self.pos[0] ** 2 + self.pos[1] ** 2)

        # estimate apoapsis/periapsis (vis-viva approximation)
        mu = _KERBIN_MU
        a = 1.0 / (2.0 / r_mag - speed * speed / mu) if r_mag > 0 and mu > 0 else float('inf')
        apo_est = max(a * (1 + np.sqrt(1 - (r_mag * speed / mu) ** 2)) - _KERBIN_RADIUS, 0) if a != float('inf') else 0.0
        peri_est = max(a * (1 - np.sqrt(1 - (r_mag * speed / mu) ** 2)) - _KERBIN_RADIUS, 0) if a != float('inf') else 0.0

        # normalize features to [0, 1] range for stable NN input
        obs = np.array([
            alt_surface / 1e6,           # altitude → [0, ~1.5]
            speed / 3000,               # velocity → [0, ~2]
            fpa / (np.pi + 1e-12),      # flight path angle
            self.pitch / (np.pi + 1e-12),
            (self.yaw + np.pi) / (2 * np.pi + 1e-12),
            self.throttle,
            float(self.current_stage) / len(getattr(self._vessel, 'stages', [(0,0)])),
            self.fuel_remaining / max(getattr(self._vessel, 'fuel_mass_kg', 38_000), 1e-6),
            self.mass / 52_000,         # mass normalized to initial
            northing / _KERBIN_RADIUS,
            easting / (_KERBIN_RADIUS * np.pi),
            radial_vel / 3000,
            apo_est / 1e6 if apo_est != float('inf') else 0.0,
            peri_est / 1e6 if peri_est != float('inf') else 0.0,
            1.0 if self.sas_on else 0.0,
            speed / (speed + 1),        # normalized speed proxy
        ], dtype=np.float32)

        return obs

    def _compute_reward(self, action: np.ndarray) -> float:
        """Reward function — distance-based for orbital insertion."""
        r_mag = np.linalg.norm(self.pos)
        alt_surface = max(r_mag - _KERBIN_RADIUS, 0.0)
        speed = float(np.linalg.norm(self.vel))

        # primary reward: altitude achieved (encourage going higher)
        alt_reward = alt_surface / 1e6  # normalize to ~[0, 2]

        # secondary: orbital velocity bonus (circular orbit at that altitude)
        v_circ = np.sqrt(_KERBIN_MU / max(r_mag, _KERBIN_RADIUS))
        vel_ratio = min(speed / max(v_circ, 1e-6), 2.0)
        vel_reward = -abs(vel_ratio - 1.0) * 0.5  # penalize deviation from circular

        # fuel efficiency penalty (small)
        fuel_penalty = -self.throttle * 0.01

        # stage completion bonus
        stage_bonus = self.current_stage * 2.0 / len(getattr(self._vessel, 'stages', [(0,0)]))

        crash_penalty = -50.0 if alt_surface < 0 else 0.0

        return float(alt_reward + vel_reward + fuel_penalty + stage_bonus + crash_penalty)


__all__ = ["OrbitSimulator", "BaseEnvironment"]
