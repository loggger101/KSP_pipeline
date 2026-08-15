"""Vectorized batched orbital simulator — runs N parallel episodes simultaneously."""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CelestialBody:
    name: str; mu: float; radius: float; atmosphere: bool = False
    atm_scale_height: float = 0.0; atm_surface_density: float = 0.0; soI_radius: float = 0.0


BODY_DATA = {
    "Kerbin": np.array([8.7129645463e12, 600_000.0, 1.0, 5_000.0, 1.225, 84_159_746.0]),
    "Mun": np.array([6.53834759e10, 200_000.0, 0.0, 0.0, 0.0, 15_897_647.0]),
    "Minmus": np.array([3.56666e9, 60_000.0, 0.0, 0.0, 0.0, 7_100_000.0]),
    "Duna": np.array([5.430872e11, 320_000.0, 1.0, 900.0, 0.663, 47_800_000.0]),
    "Eve": np.array([1.7152e12, 700_000.0, 1.0, 4_000.0, 8.364, 53_000_000.0]),
    "Sun": np.array([1.1723328e18, 261_250_000.0, 0.0, 0.0, 0.0, 1.49235678e12]),
}

TASK_CONFIGS = {
    "vertical_burn": {"obs_dim": 12, "act_dim": 1, "start_altitude": 70_000.0, "target_altitude": 100_000.0, "max_steps": 3000},
    "orbital_insertion": {"obs_dim": 15, "act_dim": 3, "start_altitude": 70_000.0, "target_radius": 700_000.0, "max_steps": 5000},
    "kerbin_orbit": {"obs_dim": 18, "act_dim": 6, "start_altitude": 70_000.0, "target_radius": 700_000.0, "max_steps": 5000},
    "duna_transfer": {"obs_dim": 20, "act_dim": 4, "start_altitude": 100_000.0, "target_radius": 7_382_400.0, "max_steps": 8000},
    "landing": {"obs_dim": 22, "act_dim": 5, "start_altitude": 50_000.0, "target_radius": 600_000.0, "max_steps": 4000},
}


@dataclass
class BatchState:
    n_agents: int
    position: np.ndarray = field(init=False)
    velocity: np.ndarray = field(init=False)
    altitude: np.ndarray = field(init=False)
    speed: np.ndarray = field(init=False)
    radius: np.ndarray = field(init=False)
    fuel_mass: np.ndarray = field(init=False)
    vessel_mass: np.ndarray = field(init=False)
    done_mask: np.ndarray = field(init=False)
    dry_mass: float = 5000.0; max_thrust: float = 200_000.0; specific_impulse: float = 340.0
    cross_section_area: float = 50.0

    def __post_init__(self):
        n = self.n_agents
        ones = np.ones(n, dtype=np.float64)
        zeros = np.zeros(n, dtype=np.float64)
        init_r = 670_000.0
        self.position = np.column_stack([ones * init_r, zeros, zeros])
        self.velocity = np.column_stack([zeros, ones * 1e-3, zeros])
        self.altitude = np.full(n, 70_000.0)
        self.speed = np.zeros(n, dtype=np.float64)
        self.radius = np.full(n, init_r)
        self.fuel_mass = np.full(n, 10_000.0)
        self.vessel_mass = np.full(n, 15_000.0)
        self.done_mask = np.zeros(n, dtype=bool)


class VectorizedOrbitSimulator:
    """Batched orbital simulator running N parallel episodes simultaneously."""

    def __init__(self, task_name: str = "vertical_burn", n_agents: int = 100, dt: float = 1.0 / 60.0):
        self.task_name = task_name; self.n_agents = n_agents; self.dt = dt
        self.config = TASK_CONFIGS.get(task_name, TASK_CONFIGS["vertical_burn"])
        self.mu = BODY_DATA["Kerbin"][0]; self.body_radius = BODY_DATA["Kerbin"][1]
        self.soI = 84_159_746.0; self.atmosphere = True; self.atm_scale_height = 5_000.0
        self.atm_surface_density = 1.225
        self.state: Optional[BatchState] = None; self.step_count = 0
        self.max_steps = self.config["max_steps"]
        self.episode_rewards: List[float] = []; self.episode_info: List[Dict[str, Any]] = []

    def reset(self) -> np.ndarray:
        start_alt = self.config["start_altitude"]; init_r = self.body_radius + start_alt
        orb_vel = math.sqrt(self.mu / init_r) if init_r > 0 else 0.0
        self.state = BatchState(n_agents=self.n_agents)
        self.state.position[:, 0] = np.full(self.n_agents, init_r)
        self.state.velocity[:, 1] = np.full(self.n_agents, orb_vel * 0.1) if "orbit" in self.task_name else np.full(self.n_agents, orb_vel * 0.01)
        r = np.sqrt(np.sum(self.state.position ** 2, axis=1))
        self.state.radius = r; self.state.altitude = r - self.body_radius
        self.state.speed = np.sqrt(np.sum(self.state.velocity ** 2, axis=1))
        self.step_count = 0; self.episode_rewards.clear(); self.episode_info.clear()
        return self._get_observation_batch()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        if self.state is None: raise RuntimeError("Call reset() first.")
        n = self.n_agents; actions = np.clip(actions, -1.0, 1.0)
        throttle = np.clip(actions[:, 0], 0.0, 1.0); pitch = actions[:, 1] if actions.shape[1] > 1 else np.zeros(n)
        yaw = actions[:, 2] if actions.shape[1] > 2 else np.zeros(n)

        fuel_rate = (throttle * self.state.max_thrust) / max(self.state.specific_impulse * 9.81, 0.01)
        fuel_consumed = fuel_rate * self.dt; old_fuel = self.state.fuel_mass.copy()
        self.state.fuel_mass = np.maximum(self.state.fuel_mass - fuel_consumed, 0.0)

        thrust_accel_mag = throttle * self.state.max_thrust / np.maximum(self.state.vessel_mass, 1.0)
        r_vec = self.state.position[:, 0]
        thrust_dir_x = np.where(r_vec > 0, 1.0, -1.0)
        cos_p = np.cos(pitch * 0.5); sin_p = np.sin(pitch * 0.3)
        vel_norm_y = self.state.velocity[:, 1] / np.maximum(self.state.speed, 1.0)

        accel_x = thrust_accel_mag * (thrust_dir_x * cos_p + vel_norm_y * sin_p)
        accel_y = thrust_accel_mag * (vel_norm_y * cos_p + sin_p * 0.3)
        accel_z = thrust_accel_mag * yaw * 0.2

        r_mag = np.sqrt(np.sum(self.state.position ** 2, axis=1)); r_mag = np.maximum(r_mag, 1.0)
        grav_f = -self.mu / (r_mag ** 3)
        total_ax = grav_f * self.state.position[:, 0] + accel_x; total_ay = accel_y; total_az = accel_z

        if self.atmosphere:
            alt_above = r_mag - self.body_radius
            atm_dens = self.atm_surface_density * np.exp(-np.maximum(alt_above, 0) / self.atm_scale_height)
            drag_c = 0.5 * atm_dens * (self.state.speed ** 2) * self.state.cross_section_area * 0.47 / np.maximum(self.state.vessel_mass, 1.0)
            vn = np.maximum(self.state.speed, 1e-6)
            total_ax -= drag_c * (self.state.velocity[:, 0] / vn); total_ay -= drag_c * (self.state.velocity[:, 1] / vn); total_az -= drag_c * (self.state.velocity[:, 2] / vn)

        self.state.velocity[:, 0] += total_ax * self.dt; self.state.velocity[:, 1] += total_ay * self.dt
        self.state.position[:, 0] += self.state.velocity[:, 0] * self.dt; self.state.position[:, 1] += self.state.velocity[:, 1] * self.dt
        r_mag = np.sqrt(np.sum(self.state.position ** 2, axis=1))
        self.state.radius = r_mag; self.state.altitude = r_mag - self.body_radius
        self.state.speed = np.sqrt(np.sum(self.state.velocity ** 2, axis=1))
        self.state.vessel_mass = self.state.dry_mass + self.state.fuel_mass

        rewards = self._compute_rewards(throttle)
        dones = np.zeros(n, dtype=bool)
        if self.task_name == "vertical_burn": dones |= self.state.altitude > self.config["target_altitude"] * 0.95
        elif "orbit" in self.task_name or self.task_name == "duna_transfer":
            target_r = self.config.get("target_radius", 700_000.0) + self.body_radius; dones |= np.abs(r_mag - target_r) / target_r < 0.05
        elif self.task_name == "landing": dones |= (self.state.altitude <= 10.0) & (self.state.speed < 30.0)
        crash_mask = self.state.altitude < -100.0
        max_step_mask = np.full(n, self.step_count >= self.max_steps)
        dones |= crash_mask | max_step_mask
        newly_done = ~self.state.done_mask & dones; self.state.done_mask |= dones

        fuel_frac = self.state.fuel_mass / 10_000.0
        info = {"altitude_m": self.state.altitude.copy(), "speed_ms": self.state.speed.copy(),
                "fuel_remaining_kg": self.state.fuel_mass.copy(), "vessel_mass_kg": self.state.vessel_mass.copy(), "throttle_avg": throttle.mean()}

        if np.any(newly_done):
            for idx in np.where(newly_done)[0]:
                hist = [h["reward"] for h in getattr(self, 'history_per_agent', {}).get(int(idx), [{"reward": 0.0}])]
                self.episode_rewards.append(float(np.mean(hist))); self.episode_info.append({"agent_idx": int(idx), "steps": self.step_count + 1})

        self.step_count += 1; return self._get_observation_batch(), rewards, dones, info

    def _compute_rewards(self, throttle):
        n = self.n_agents; rewards = np.zeros(n)
        if self.task_name == "vertical_burn":
            alt_ratio = np.minimum(np.maximum(0, self.state.altitude) / (self.config["target_altitude"] + self.body_radius), 5.0)
            rewards += alt_ratio * 2.0; h_vel = np.sqrt(self.state.velocity[:, 1]**2 + self.state.velocity[:, 2]**2)
            tv_arr = np.maximum(self.state.speed, 1.0)
            rewards -= 0.5 * (h_vel / tv_arr); rewards += (self.state.fuel_mass / 10_000.0) * 0.5
        elif "orbit" in self.task_name or self.task_name == "duna_transfer":
            target_r = self.config.get("target_radius", 700_000.0) + self.body_radius; ae = np.abs(self.state.radius - target_r) / target_r
            rewards -= np.minimum(ae * 5.0, 10.0); ec = np.sqrt(self.mu / np.maximum(self.state.radius, 1.0))
            vr = np.abs(self.state.speed - ec) / np.maximum(ec, 1.0); rewards -= np.minimum(vr * 3.0, 5.0); rewards += (self.state.fuel_mass / 10_000.0) * 1.0
        elif self.task_name == "landing":
            ae = np.maximum(0, self.state.altitude) / np.maximum(self.body_radius, 1.0); rewards -= np.minimum(ae * 5.0, 10.0)
            lp = np.where(self.state.altitude < 10_000, np.maximum(0, (self.state.speed - 50)) / 200.0 * 3.0, 0.0); rewards -= lp; rewards += (self.state.fuel_mass / 10_000.0) * 2.0
        return rewards

    def _get_observation_batch(self) -> np.ndarray:
        if self.state is None or self.n_agents == 0: return np.zeros((max(self.config["obs_dim"], 1),))
        n = self.n_agents; r = np.maximum(self.state.radius, 1.0); ov = np.sqrt(self.mu / r)
        obs = [self.state.altitude / (self.soI * 0.5), self.state.speed / np.maximum(ov * 2, 1.0),
               self.state.velocity[:, 0] / np.maximum(ov, 1.0), self.state.velocity[:, 1] / np.maximum(ov, 1.0),
               self.state.velocity[:, 2] / np.maximum(ov, 1.0), r / self.soI]
        if "orbit" in self.task_name or self.task_name == "vertical_burn":
            ec = np.sqrt(self.mu / r); obs.extend([np.abs(self.state.speed - ec) / np.maximum(ec, 1.0),
                (self.state.speed * self.state.velocity[:, 0] / np.maximum(ov, 1.0)) / np.maximum(ov, 1.0)])
        elif self.task_name == "duna_transfer":
            obs.extend([(0.5 * self.state.speed**2 - self.mu/r) / np.maximum(ov**2, 1.0), self.state.speed / np.sqrt(np.maximum(2*self.mu/r, 1e-6))])
        elif self.task_name == "landing":
            obs.extend([self.state.speed/np.maximum(ov, 1.0), (np.arctan2(np.abs(self.state.velocity[:, 1]), np.maximum(np.abs(self.state.velocity[:, 0]), 1e-6))/(np.pi/4))-1.0,
                (np.minimum(np.maximum(0,self.state.altitude)/np.maximum(self.state.speed,1.0)/60., 2.) - 1.0)])
        batch = np.column_stack(obs)
        while batch.shape[1] < self.config["obs_dim"]: batch = np.pad(batch, ((0,0),(0,1)), constant_values=0.0)
        return batch[:, :self.config["obs_dim"]].astype(np.float32)

    def get_population_stats(self) -> Dict[str, float]:
        if not self.state: return {}
        active = ~self.state.done_mask; na = int(np.sum(active))
        stats = {"step": self.step_count, "n_done": int(np.sum(self.state.done_mask)), "n_active": na}
        if na > 0:
            stats.update({"mean_altitude": float(self.state.altitude[active].mean()), "max_altitude": float(self.state.altitude[active].max()),
                          "min_altitude": float(self.state.altitude[active].min()), "mean_speed": float(self.state.speed[active].mean()),
                          "fuel_frac_mean": float(self.state.fuel_mass[active].mean() / 10_000.0)})
        if self.episode_rewards: stats.update({"best_reward": float(max(self.episode_rewards)), "worst_reward": float(min(self.episode_rewards)),
                                                "mean_reward": float(np.mean(self.episode_rewards)), "std_reward": float(np.std(self.episode_rewards))})
        return stats
