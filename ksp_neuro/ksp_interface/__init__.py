"""kRPC-based live Kerbal Space Program game interface."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class KSPState:
    """Snapshot of vessel state from the kRPC server.

    Attributes
    ----------
    altitude : float  # meters above sea level
    velocity_magnitude : float  # m/s
    orbital_velocity : np.ndarray  # [m/s] in local frame
    apoapsis_alt : float  # meters
    periapsis_alt : float  # meters
    stage_count : int
    fuel_ratio : float  # [0, 1]
    mass_kg : float
    pitch_rad : float
    yaw_rad : float
    roll_rad : float
    sas_on : bool
    gear_down : bool
    throttle : float  # [0, 1]
    gravity : np.ndarray  # gravitational acceleration vector at vessel [m/s²]
    atmospheric_density : float  # kg/m³ (0.0 if outside atmosphere)
    max_altitude_reached : float  # meters
    total_distance_traveled : float  # meters
"""

    altitude: float = 0.0
    velocity_magnitude: float = 0.0
    orbital_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    apoapsis_alt: float = 0.0
    periapsis_alt: float = 0.0
    stage_count: int = 0
    fuel_ratio: float = 1.0
    mass_kg: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0
    roll_rad: float = 0.0
    sas_on: bool = False
    gear_down: bool = True
    throttle: float = 0.0
    gravity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    atmospheric_density: float = 0.0
    max_altitude_reached: float = 0.0
    total_distance_traveled: float = 0.0


class KSPInterface:
    """kRPC client wrapper for live Kerbal Space Program interaction.

    Connects to a running kRPC server (the KSP mod) and exposes
    vessel state reading + action sending as an OpenAI-gym-like env.

    Usage
    -----
    >>> ksp = KSPInterface(host="localhost", port=6767)
    >>> obs = ksp.reset()          # returns normalized observation np.ndarray
    >>> obs, reward, done, info = ksp.step(action_vector)
    >>> ksp.close()

    Parameters
    ----------
    host : str
        Hostname of the machine running the KSP kRPC server.
    port : int
        TCP port for the kRPC connection (default 6767).
    timeout_s : float
        Connection timeout in seconds.
    """

    def __init__(self, host: str = "localhost", port: int = 6767, timeout_s: float = 10.0):
        self._host = host
        self._port = port
        self._timeout_s = timeout_s
        self._krpc = None          # kRPC connection object (lazy)
        self._vessel = None        # vessel reference
        self._stream = None        # continuous state stream handle

        # observation/action specs
        self._obs_keys = [
            "alt_surface", "vel_magnitude", "flight_path_angle",
            "pitch", "yaw", "roll", "throttle", "stage_count",
            "fuel_ratio", "mass_kg", "apoapsis_alt", "periapsis_alt",
            "radial_vel", "sas_on", "gear_down", "brake_force",
        ]
        self._action_keys = [
            "throttle", "pitch_offset", "yaw_offset", "roll_offset",
            "sas_toggle", "stage_fire", "gear_toggle", "brake_force",
        ]

        # tracking state for reward computation
        self._prev_altitude = 0.0
        self._max_altitude = 0.0
        self._total_distance = 0.0
        self._steps = 0
        self._done = False

    @property
    def observation_space(self):
        return len(self._obs_keys), "float32"

    @property
    def action_space(self):
        return len(self._action_keys), "float32"

    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Establish TCP connection to the kRPC server."""
        try:
            import krpc  # pip install krpc-client (or the appropriate package)
        except ImportError:
            raise ImportError(
                "kRPC Python client not installed. Install via:\n"
                "    pip install krpc-client\n"
                "Or build from https://github.com/krpc/krpc"
            )

        self._krpc = krpc.connect(name="KSP_Neuroevolution", remote_service_name="ksp_neuro")
        self._vessel = self._krpc.space_center.active_vessel
        self._stream = None  # will be set in reset()

    def _ensure_connected(self):
        if self._krpc is None:
            raise RuntimeError("Not connected to kRPC server. Call connect() first.")

    # ------------------------------------------------------------------
    def reset(
        self,
        initial_altitude_km: float = 0.0,
        seed: int | None = None,
    ) -> np.ndarray:
        """Reset the vessel state and return observation.

        In live KSP this reads current vessel state from the game via kRPC.
        The *initial_altitude_km* parameter is informational only (cannot
        modify the loaded game state).

        Returns
        -------
        obs : np.ndarray  shape (16,)
            Normalized observation vector.
        """
        self._ensure_connected()
        self._prev_altitude = 0.0
        self._max_altitude = 0.0
        self._total_distance = 0.0
        self._steps = 0
        self._done = False

        return self._get_observation_from_krpc()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one tick of actions against the live game.

        Parameters
        ----------
        action : np.ndarray  shape (8,)
            Clamped control vector keyed by ``_action_keys``.

        Returns
        -------
        obs : np.ndarray
        reward : float
        done : bool
        info : dict
        """
        self._ensure_connected()
        ctrl = np.clip(action, -1.0, 1.0)

        # apply actions to the vessel via kRPC
        self._apply_actions(ctrl)

        # read updated state from game
        obs = self._get_observation_from_krpc()
        reward = self._compute_reward(ctrl)
        done = self._check_termination()

        return obs, reward, done, {
            "altitude": self._prev_altitude,
            "max_altitude": self._max_altitude,
            "distance": self._total_distance,
            "steps": self._steps,
        }

    def close(self) -> None:
        """Disconnect from the kRPC server."""
        if self._krpc is not None:
            try:
                self._krpc.close()
            except Exception:
                pass  # best-effort cleanup
            self._krpc = None
            self._vessel = None

    # ---- private helpers (live KSP state) ---------------------------------
    def _get_observation_from_krpc(self) -> np.ndarray:
        """Read vessel state from kRPC and build normalized observation."""
        if self._vessel is None:
            raise RuntimeError("No active vessel. Is a ship loaded?")

        v = self._vessel
        srf_vel = v.surface_velocity  # surface-relative velocity vector
        vel_mag = float(np.linalg.norm(srf_vel))

        altitude = v.altitude  # from sea level (m)
        apoapsis = v.orbit.apoapsis_altitude if v.orbit else 0.0
        periapsis = v.orbit.periapsis_altitude if v.orbit else 0.0

        # radial velocity (outward positive)
        pos_norm = v.pos - self._krpc.space_center.body.position
        r_mag = float(np.linalg.norm(pos_norm))
        radial_vel = float(np.dot(srf_vel, pos_norm / max(r_mag, 1e-6))) if r_mag > 0 else 0.0

        # flight path angle
        cos_fpa = np.dot(srf_vel, pos_norm) / (vel_mag * r_mag + 1e-12)
        fpa = float(np.arccos(np.clip(cos_fpa, -1, 1)))

        # atmospheric density at altitude
        body = self._krpc.space_center.body
        if v.orbit and hasattr(body, 'atmosphere') and body.atmosphere:
            atm_alt = v.orbit.altitude
            scale_h = getattr(body.atmosphere, 'scale_height', 5600.0)
            density = body.atmosphere.atmosphereDensity  # direct API call

        obs = np.array([
            altitude / 1e6,
            vel_mag / 3000,
            fpa / (np.pi + 1e-12),
            v.attitude.pitch / (np.pi + 1e-12),
            (v.attitude.yaw + np.pi) / (2 * np.pi + 1e-12),
            v.attitude.roll / (np.pi + 1e-12),
            v.control.throttle,
            float(v.available_stages),
            v.fuel_percentage / 100.0 if hasattr(v, 'fuel_percentage') else 0.0,
            v.mass / 52_000,
            apoapsis / 1e6,
            periapsis / 1e6,
            radial_vel / 3000,
            1.0 if v.control.sas else 0.0,
            1.0 if v.control.gear else 0.0,
            vel_mag / (vel_mag + 1),
        ], dtype=np.float32)

        # update tracking state
        self._prev_altitude = altitude
        self._max_altitude = max(self._max_altitude, altitude)
        dt = 0.1
        self._total_distance += vel_mag * dt
        self._steps += 1

        return obs

    def _apply_actions(self, ctrl: np.ndarray) -> None:
        """Send control actions to the live kRPC server."""
        if self._vessel is None:
            return

        # throttle (always settable via kRPC)
        self._vessel.control.throttle = float(ctrl[0])

        # pitch/yaw/roll — use RCS or SAS for attitude control
        pitch_rate = float(ctrl[1]) * np.pi / 4   # ±45 deg/s
        yaw_rate = float(ctrl[2]) * np.pi / 4
        roll_rate = float(ctrl[3]) * np.pi / 2

        if ctrl[4] > 0.5:  # SAS toggle on
            self._vessel.control.sas = True
            # point prograde for stability
            self._vessel.control.manual_control = {
                "pitch": pitch_rate != 0,
                "yaw": yaw_rate != 0,
                "roll": roll_rate != 0,
            }
        else:
            self._vessel.control.sas = False

        # stage fire
        if ctrl[5] > 0.5:
            try:
                self._vessel.control.next_stage()
            except Exception:
                pass

        # gear toggle
        if abs(float(ctrl[6])) > 0.5:
            self._vessel.control.gear = not self._vessel.control.gear

        # brakes
        brake_val = float(ctrl[7])
        for wheel in (getattr(self._vessel.wheels, 'wheels', [])):
            if hasattr(wheel, 'brake'):
                wheel.brake = brake_val

    def _compute_reward(self, action: np.ndarray) -> float:
        """Reward function — same logic as OrbitSimulator."""
        alt = self._prev_altitude
        vel_mag = float(np.linalg.norm(
            getattr(getattr(self._vessel, 'surface_velocity', None), '__len__', lambda: 0)() or [0]
        )) if self._vessel is not None else 0.0

        # simplified — in practice would read from kRPC state
        alt_reward = alt / 1e6
        vel_penalty = -abs(vel_mag - 2300) * 0.001  # target orbital velocity
        fuel_penalty = -action[0] * 0.01
        stage_bonus = float(getattr(self._vessel, 'available_stages', 0)) * 2.0 / 5.0

        return float(alt_reward + vel_penalty + fuel_penalty + stage_bonus)

    def _check_termination(self) -> bool:
        if self._vessel is None:
            return True

        alt = self._vessel.altitude
        r_mag = getattr(getattr(self._vessel, 'pos', None), '__len__', lambda: 0)() or [0]
        if len(r_mag) > 0 and np.linalg.norm(r_mag) < 600_000:
            return True

        # escaped too far
        if r_mag is not None and hasattr(self._vessel, 'orbit') and self._vessel.orbit is not None:
            if self._vessel.orbit.reference_body is not None:
                max_dist = 5_000_000 * 600_000
                if np.linalg.norm(
                    getattr(self._vessel, 'pos', None) or [0] -
                    (getattr(getattr(self._vessel.orbit, 'reference_body', None), 'position', None) or [0])
                ) > max_dist:
                    return True

        self._steps += 1
        if self._steps >= 36_000:
            return True

        return False


__all__ = ["KSPInterface", "KSPState"]
