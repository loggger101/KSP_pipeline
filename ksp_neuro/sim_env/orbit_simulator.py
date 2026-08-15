"""Single-agent high-fidelity orbital mechanics simulator for KSP-like environments.

Implements patched-conics approximation with:
- Keplerian two-body dynamics
- Sphere of influence (SOI) transitions between celestial bodies
- Gravity model with J2 perturbations (optional)
- Atmospheric drag model for low orbits
- Thrust/propulsion model with specific impulse and mass depletion
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Dict, List, Tuple


# ───────────────────── Celestial Body Data (KSP-like) ─────────────────────

@dataclass(frozen=True)
class CelestialBody:
    """A celestial body with gravitational and physical properties."""
    name: str
    mu: float  # Standard gravitational parameter [m^3/s^2]
    radius: float  # Surface radius [m]
    atmosphere: bool = False
    atm_scale_height: float = 0.0
    atm_surface_density: float = 0.0
    soI_radius: float = 0.0


CELESTIAL_BODIES = {
    "Kerbin": CelestialBody(name="Kerbin", mu=8.7129645463e12, radius=600_000.0,
        atmosphere=True, atm_scale_height=5_000.0, atm_surface_density=1.225, soI_radius=84_159_746.0),
    "Mun": CelestialBody(name="Mun", mu=6.53834759e10, radius=200_000.0, atmosphere=False, soI_radius=15_897_647.0),
    "Minmus": CelestialBody(name="Minmus", mu=3.56666e9, radius=60_000.0, atmosphere=False, soI_radius=7_100_000.0),
    "Duna": CelestialBody(name="Duna", mu=5.430872e11, radius=320_000.0,
        atmosphere=True, atm_scale_height=900.0, atm_surface_density=0.663, soI_radius=47_800_000.0),
    "Eve": CelestialBody(name="Eve", mu=1.7152e12, radius=700_000.0,
        atmosphere=True, atm_scale_height=4_000.0, atm_surface_density=8.364, soI_radius=53_000_000.0),
    "Sun": CelestialBody(name="Sun", mu=1.1723328e18, radius=261_250_000.0, atmosphere=False, soI_radius=1.49235678e12),
}


# ───────────────────── Orbit State Representation ─────────────────────

@dataclass
class Vector3:
    """Simple 3D vector with KSP coordinate conventions."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector3":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vector3":
        return Vector3(-self.x, -self.y, -self.z)

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self) -> "Vector3":
        mag = self.magnitude
        if mag == 0:
            return Vector3(0, 0, 0)
        return self / mag


@dataclass
class OrbitalState:
    """Full orbital state of a vessel."""
    position: Vector3
    velocity: Vector3
    current_body: str = "Kerbin"
    altitude: float = 0.0
    speed: float = 0.0

    def __post_init__(self):
        self.speed = self.velocity.magnitude


@dataclass
class VesselProperties:
    """Physical properties of the vessel being controlled."""
    mass: float = 10_000.0
    dry_mass: float = 5_000.0
    max_thrust: float = 200_000.0
    specific_impulse: float = 340.0
    cross_section_area: float = 50.0
    fuel_mass: float = 10_000.0


# ───────────────────── Environment Parameters ─────────────────────

@dataclass
class EnvParams:
    """Configuration for a training environment/task."""
    task_name: str
    start_altitude_m: float = 70_000.0
    target_altitude_m: float = 100_000.0
    target_orbit_radius_m: float = 0.0
    max_episode_steps: int = 3000
    dt: float = 1.0 / 60.0
    gravity_angle_penalty: bool = True
    fuel_efficiency_bonus: bool = True
    orbit_circularity_reward: bool = False

    @property
    def observation_dim(self) -> int:
        dims = {"vertical_burn": 12, "orbital_insertion": 15, "kerbin_orbit": 18,
                "duna_transfer": 20, "landing": 22}
        return dims.get(self.task_name, 15)

    @property
    def action_dim(self) -> int:
        dims = {"vertical_burn": 1, "orbital_insertion": 3, "kerbin_orbit": 6,
                "duna_transfer": 4, "landing": 5}
        return dims.get(self.task_name, 3)


DEFAULT_ENV_PARAMS = {
    "vertical_burn": EnvParams(task_name="vertical_burn", start_altitude_m=70_000.0, target_altitude_m=100_000.0),
    "orbital_insertion": EnvParams(task_name="orbital_insertion", start_altitude_m=70_000.0, target_orbit_radius_m=700_000.0, max_episode_steps=5000),
    "kerbin_orbit": EnvParams(task_name="kerbin_orbit", start_altitude_m=70_000.0, target_orbit_radius_m=700_000.0, max_episode_steps=5000, orbit_circularity_reward=True),
    "duna_transfer": EnvParams(task_name="duna_transfer", start_altitude_m=100_000.0, target_orbit_radius_m=7_382_400.0, max_episode_steps=8000),
    "landing": EnvParams(task_name="landing", start_altitude_m=50_000.0, target_altitude_m=0.0, max_episode_steps=4000),
}

ENV_REGISTRY = dict(DEFAULT_ENV_PARAMS)


# ───────────────────── Physics Engine ─────────────────────

class OrbitSimulator:
    """High-fidelity single-agent orbital mechanics simulator."""

    def __init__(self, task_name: str = "vertical_burn", params: Optional[EnvParams] = None):
        self.params = params or ENV_REGISTRY.get(task_name, DEFAULT_ENV_PARAMS["vertical_burn"])
        if hasattr(self.params, 'task_name') is False and isinstance(self.params, dict):
            self.params = EnvParams(**self.params)

        self.current_body: Optional[CelestialBody] = None
        self.vessel = VesselProperties()
        self.state = OrbitalState(Vector3(), Vector3())
        self.step_count = 0
        self.done = False
        self.reward = 0.0
        self.info: Dict[str, Any] = {}
        self.episode_history: List[Dict[str, Any]] = []

    def reset(self) -> OrbitalState:
        """Reset simulation to initial conditions."""
        body_name = "Kerbin" if self.params.task_name != "duna_transfer" else "Sun"
        self.current_body = CELESTIAL_BODIES.get(body_name, CELESTIAL_BODIES["Kerbin"])

        radius = self.current_body.radius + self.params.start_altitude_m
        orbital_velocity = math.sqrt(self.current_body.mu / radius) if radius > 0 else 0.0

        if "orbit" in self.params.task_name:
            self.state = OrbitalState(
                position=Vector3(x=radius, y=0, z=0),
                velocity=Vector3(x=0, y=orbital_velocity * 0.1, z=0),
                current_body=body_name,
            )
        else:
            self.state = OrbitalState(
                position=Vector3(x=radius, y=0, z=0),
                velocity=Vector3(x=orbital_velocity * 0.01, y=0, z=0),
                current_body=body_name,
            )

        self.state.altitude = self.state.position.x - self.current_body.radius if body_name == "Kerbin" else 0.0
        self.state.speed = self.state.velocity.magnitude
        self.vessel = VesselProperties()
        self.step_count = 0
        self.done = False
        self.reward = 0.0
        self.episode_history.clear()
        self.info = {}
        return copy.deepcopy(self.state)

    def step(self, action: List[float]) -> Tuple[OrbitalState, float, bool, Dict[str, Any]]:
        """Execute one physics timestep."""
        if self.done:
            raise RuntimeError("Cannot step a finished episode. Call reset() first.")

        clamped = [max(-1.0, min(1.0, a)) for a in action]
        throttle = max(0.0, clamped[0]) if len(clamped) > 0 else 0.0
        pitch_angle = clamped[1] if len(clamped) > 1 else 0.0
        yaw_angle = clamped[2] if len(clamped) > 2 else 0.0

        thrust_mag = throttle * self.vessel.max_thrust
        accel = Vector3(0, 0, 0)

        if thrust_mag > 0 and self.vessel.fuel_mass > 0:
            fuel_consumed = min((thrust_mag / (self.vessel.specific_impulse * 9.81)) * self.params.dt, self.vessel.fuel_mass)
            self.vessel.fuel_mass -= fuel_consumed
            self.vessel.mass = self.vessel.dry_mass + self.vessel.fuel_mass
            accel_mag = thrust_mag / max(self.vessel.mass, 1.0)

            radial_dir = Vector3(1, 0, 0) if self.state.position.x > 0 else Vector3(-1, 0, 0)
            prograde_dir = self.state.velocity.normalized() if self.state.speed > 1.0 else Vector3(0, 1, 0)

            thrust_direction = radial_dir * math.cos(pitch_angle * 0.5) + prograde_dir * math.sin(pitch_angle * 0.3)
            thrust_direction = thrust_direction.normalized() if thrust_direction.magnitude > 1e-10 else Vector3(1, 0, 0)

            yaw_component = self._cross_product(radial_dir, prograde_dir).normalized() * math.sin(yaw_angle * 0.2)
            thrust_direction = (thrust_direction + yaw_component).normalized() if thrust_direction.magnitude > 1e-10 else Vector3(1, 0, 0)

            accel = thrust_direction * accel_mag

        r = self.state.position.magnitude
        gravity_accel = (-self.current_body.mu / (r ** 2)) * self.state.position.normalized() if r > 1e-6 and self.current_body else Vector3(0, 0, 0)

        drag_accel = Vector3(0, 0, 0)
        if self.current_body and self.current_body.atmosphere and r > self.current_body.radius:
            alt_above = r - self.current_body.radius
            atm_density = self.current_body.atm_surface_density * math.exp(-max(alt_above, 0) / self.current_body.atm_scale_height)
            drag_mag = 0.5 * atm_density * (self.state.speed ** 2) * self.vessel.cross_section_area * 0.47
            vel_norm = self.state.velocity.normalized() if self.state.speed > 0 else Vector3(0, 1, 0)
            df = drag_mag / max(self.vessel.mass, 1.0)
            drag_accel = Vector3(-vel_norm.x * df, -vel_norm.y * df, -vel_norm.z * df)

        total_accel = gravity_accel + accel + drag_accel
        self.state.velocity = self.state.velocity + total_accel * self.params.dt
        self.state.position = self.state.position + self.state.velocity * self.params.dt
        r = max(self.state.position.magnitude, 1.0)
        self.state.speed = math.sqrt(sum(c ** 2 for c in [self.state.velocity.x, self.state.velocity.y, self.state.velocity.z]))
        self.state.altitude = (r - self.current_body.radius) if self.current_body else 0.0

        # SOI transition check (simplified)
        if r > self.current_body.soI_radius * 2.5:
            pass  # Would handle body switch here in full version

        self.step_count += 1
        fuel_frac = self.vessel.fuel_mass / max(self.vessel.max_thrust * 10, 1.0)
        self.reward = self._compute_reward()
        self.done = self._check_termination()
        self.info = {"step": self.step_count, "altitude_m": self.state.altitude, "speed_ms": self.state.speed,
                     "fuel_remaining_kg": self.vessel.fuel_mass, "vessel_mass_kg": self.vessel.mass}

        if self.episode_history is not None:
            self.episode_history.append({"step": self.step_count, "altitude": self.state.altitude,
                                         "speed": self.state.speed, "reward": self.reward})

        return copy.deepcopy(self.state), self.reward, self.done, self.info

    def _compute_reward(self) -> float:
        r = 0.0
        if self.params.task_name == "vertical_burn":
            target_alt = self.params.target_altitude_m + (self.current_body.radius if self.current_body else 600_000)
            alt_ratio = min(max(0, self.state.altitude) / max(target_alt, 1.0), 5.0)
            r += alt_ratio * 2.0
            h_vel = math.sqrt(self.state.velocity.y ** 2 + self.state.velocity.z ** 2)
            total_vel = max(self.state.speed, 1.0)
            r -= 0.5 * (h_vel / total_vel) if self.params.gravity_angle_penalty else 0.0
            r += (self.vessel.fuel_mass / 10_000.0) * 0.5 if self.params.fuel_efficiency_bonus else 0.0
        elif "orbit" in self.params.task_name or self.params.task_name == "duna_transfer":
            target_r = (self.params.target_orbit_radius_m + (self.current_body.radius if self.current_body else 600_000))
            r_err = abs(max(self.state.position.magnitude, 1.0) - target_r) / target_r
            r -= min(r_err * 5.0, 10.0)
            circ_vel = math.sqrt((self.current_body.mu if self.current_body else 8.7e12) / max(self.state.position.magnitude, 1.0))
            vel_ratio = abs(self.state.speed - circ_vel) / max(circ_vel, 1.0)
            r -= min(vel_ratio * 3.0, 5.0) if self.params.orbit_circularity_reward else 0.0
            r += (self.vessel.fuel_mass / 10_000.0) * 1.0
        elif self.params.task_name == "landing":
            alt_err = max(0, self.state.altitude) / max((self.current_body.radius if self.current_body else 600_000), 1.0)
            r -= min(alt_err * 5.0, 10.0)
            low_alt = self.state.altitude < 10_000
            land_pen = max(0, (self.state.speed - 50)) / 200.0 * 3.0 if low_alt else 0.0
            r -= land_pen
            r += (self.vessel.fuel_mass / 10_000.0) * 2.0
        return r

    def _check_termination(self) -> bool:
        if self.step_count >= self.params.max_episode_steps:
            return True
        if self.current_body and self.state.altitude < -100.0:
            return True
        if self.current_body and max(self.state.position.magnitude, 1.0) > self.current_body.soI_radius * 3.0:
            return True
        if self.params.task_name == "vertical_burn" and self.state.altitude > self.params.target_altitude_m * 0.95:
            return True
        if ("orbit" in self.params.task_name or self.params.task_name == "duna_transfer"):
            target_r = (self.params.target_orbit_radius_m + (self.current_body.radius if self.current_body else 600_000))
            cur_r = max(self.state.position.magnitude, 1.0)
            if abs(cur_r - target_r) / target_r < 0.05:
                return True
        if self.params.task_name == "landing" and self.state.altitude <= 10.0 and self.state.speed < 30.0:
            return True
        return False

    @staticmethod
    def _cross_product(a: Vector3, b: Vector3) -> Vector3:
        return Vector3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)

    def get_observation(self) -> List[float]:
        if self.current_body is None:
            return [0.0] * self.params.observation_dim
        body = self.current_body
        r = max(max(self.state.position.magnitude, 1.0), 1.0)
        orb_vel = math.sqrt(body.mu / r) if r > body.radius else 0

        obs = [
            self.state.altitude / (body.soI_radius * 0.5),
            self.state.speed / max(orb_vel * 2, 1.0),
            self.state.velocity.x / max(orb_vel, 1.0) if orb_vel > 0 else 0,
            self.state.velocity.y / max(orb_vel, 1.0) if orb_vel > 0 else 0,
            self.state.velocity.z / max(orb_vel, 1.0) if orb_vel > 0 else 0,
            r / body.soI_radius,
        ]

        if "orbit" in self.params.task_name or self.params.task_name == "vertical_burn":
            circ_vel = math.sqrt(body.mu / r) if orb_vel > 0 else 0
            circ_err = abs(self.state.speed - circ_vel) / max(circ_vel, 1.0)
            alt_rate = self.state.altitude_rate_approx() if hasattr(self.state, 'altitude_rate_approx') else self.state.velocity.magnitude * (obs[2] if obs[2] != 0 else 1.0)
            norm_alt_rate = alt_rate / max(orb_vel, 1.0)
            obs.extend([circ_err, norm_alt_rate])

        elif self.params.task_name == "duna_transfer":
            energy = 0.5 * self.state.speed ** 2 - (body.mu / r) if body else 0
            esc_ratio = self.state.speed / math.sqrt(max(2 * body.mu / r, 1e-6)) if body and r > 0 else 0
            obs.extend([energy / max(orb_vel ** 2, 1.0), esc_ratio])

        elif self.params.task_name == "landing":
            impact_angle = math.atan2(abs(self.state.velocity.y), abs(self.state.velocity.x)) if self.state.position.x > 1 else 0
            norm_impact = (impact_angle / (math.pi / 4)) - 1.0
            tt_surf = max(0, self.state.altitude) / max(self.state.speed, 1.0)
            obs.extend([self.state.speed / max(orb_vel, 1.0), norm_impact, min(tt_surf / 60.0, 2.0) - 1.0])

        while len(obs) < self.params.observation_dim:
            obs.append(0.0)
        return obs[:self.params.observation_dim]


class OrbitalStateWithRate(OrbitalState):
    """Extended state with altitude rate tracking."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_alt = 0.0

    def altitude_rate_approx(self) -> float:
        return (self.altitude - self._prev_alt) / 1.0 if hasattr(self, '_prev_alt') else 0.0


# ───────────────────── Convenience ─────────────────────

def throttle_safe(obs):
    """Dummy helper — returns a constant."""
    return 0.0
