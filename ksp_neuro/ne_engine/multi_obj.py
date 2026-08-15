"""NSGA-II multi-objective optimization engine with Pareto-front tracking."""

from __future__ import annotations
import math; import copy; import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Objective:
    name: str; minimize: bool = False

@dataclass
class AgentFitness:
    genome_id: int = 0; objectives: Dict[str, float] = field(default_factory=dict)
    rank: int = -1; crowding_distance: float = 0.0; species_fitness: float = 0.0
    @property
    def raw_fitness(self): return -sum(self.objectives.values())
    @property
    def is_pareto_optimal(self): return self.rank == 0
    def to_dict(self): return {"genome_id":self.genome_id,"objectives":dict(self.objectives),"rank":self.rank}


class NSGAIIEvaluator:
    def __init__(self, objectives=None):
        self.objectives = objectives or [Objective("altitude",minimize=False), Objective("delta_v_efficiency",minimize=True),
            Objective("maneuver_accuracy",minimize=True), Objective("stability_score",minimize=False)]
        self.pareto_front_history = []; self.current_pareto_front = set()

    def evaluate_fitness(self, genomes, episode_rewards, episode_info=None):
        fitnesses = []
        for i, genome in enumerate(genomes):
            gid = getattr(genome,"genome_id",i); rewards = episode_rewards.get(gid,0.0)
            info = episode_info.get(gid,{}) if episode_info else {}
            objectives = self._compute_objectives(rewards, info)
            fitnesses.append(AgentFitness(genome_id=gid, objectives=objectives))
        self._non_dominated_sort(fitnesses); self._crowding_distance(fitnesses)
        pf_ids = {f.genome_id for f in fitnesses if f.rank == 0}
        self.pareto_front_history.append(list(pf_ids)); self.current_pareto_front = pf_ids
        return fitnesses

    def _compute_objectives(self, total_reward, info):
        final_alt = info.get("final_altitude",info.get("altitude_m",0))
        if isinstance(final_alt,np.ndarray): final_alt=float(np.max(final_alt))
        fuel_used = info.get("fuel_remaining_kg",10_000.0)
        if isinstance(fuel_used,np.ndarray): fuel_used=float(np.max(fuel_used))
        efficiency = (10_000 - max(0, 10_000-fuel_used))/max(10_000,1)
        speed = info.get("final_speed",info.get("speed_ms",0))
        if isinstance(speed,np.ndarray): speed=float(np.max(speed))
        target_vel=7660.0; acc_err=abs(speed-target_vel)/target_vel if target_vel>0 else 1.0
        throttle_avg = info.get("throttle_avg",0.5)
        if isinstance(throttle_avg,np.ndarray): throttle_avg=float(np.mean(throttle_avg))
        stability = 1.0 - abs(throttle_avg-0.5)*2.0
        return {"altitude":max(0,float(final_alt))/1e6, "delta_v_efficiency":1.0-efficiency,
                "maneuver_accuracy":min(acc_err,5.0), "stability_score":max(0,min(1.0,stability))}

    def _dominates(self, a, b):
        """Check if solution a dominates solution b."""
        for i, obj in enumerate(self.objectives):
            val_a = a[i]; val_b = b[i]
            if obj.minimize:
                if val_a >= val_b: return False
            else:
                if val_a <= val_b: return False
        return True

    def _non_dominated_sort(self, fitnesses):
        n = len(fitnesses)
        obj_list = [[f.objectives.get(o.name,0) for o in self.objectives] for f in fitnesses]
        
        # Fast non-dominated sort (NSGA-II style)
        dominated_count = [0]*n; dominated_set = [[] for _ in range(n)]
        
        for p in range(n):
            for q in range(p+1, n):
                if self._dominates(obj_list[p], obj_list[q]):
                    dominated_set[p].append(q); dominated_count[q] += 1
                elif self._dominates(obj_list[q], obj_list[p]):
                    dominated_set[q].append(p); dominated_count[p] += 1
        
        # Find first front (rank 0)
        current_front = [i for i in range(n) if dominated_count[i] == 0]
        rank = 0
        
        while current_front:
            for idx in current_front: fitnesses[idx].rank = rank
            next_front = []
            for p in current_front:
                for q in dominated_set[p]:
                    dominated_count[q] -= 1
                    if dominated_count[q] == 0: next_front.append(q)
            current_front = next_front; rank += 1

    def _crowding_distance(self, fitnesses):
        fronts = {}
        for f in fitnesses: fronts.setdefault(f.rank,[]).append(f)
        n_obj = len(self.objectives)
        
        for rank, members in list(fronts.items())[:5]:
            if len(members) <= 2:
                for m in members: m.crowding_distance = float('inf')
                continue
            
            dists = [0.0] * len(members)
            
            for oi, obj in enumerate(self.objectives):
                sm = sorted(members, key=lambda x: x.objectives.get(obj.name, 0))
                rng = sm[-1].objectives.get(obj.name, 0) - sm[0].objectives.get(obj.name, 0)
                if rng == 0: continue
                
                dists[members.index(sm[0])] = float('inf')
                dists[members.index(sm[-1])] = float('inf')
                
                for i in range(1, len(sm) - 1):
                    idx_in_members = members.index(sm[i])
                    dists[idx_in_members] += (sm[i+1].objectives.get(obj.name, 0) - sm[i-1].objectives.get(obj.name, 0)) / rng
            
            for i, m in enumerate(members): m.crowding_distance = dists[i]

    def select_parents(self, fitnesses, n_parents=2):
        p1 = self._tournament(fitnesses)
        p2 = self._tournament(fitnesses)
        while p2 == p1 and len(fitnesses) > 1: p2 = self._tournament(fitnesses)
        return p1, p2

    def _tournament(self, fitnesses, size=2):
        cands = np.random.choice(len(fitnesses), size=min(size, len(fitnesses)), replace=False)
        best = int(cands[0])
        for idx in cands[1:]:
            if fitnesses[idx].rank < fitnesses[best].rank: best = int(idx)
            elif fitnesses[idx].rank == fitnesses[best].rank and fitnesses[idx].crowding_distance > fitnesses[best].crowding_distance: best = int(idx)
        return best

    def compute_adaptive_fitness(self, fitnesses, species_map):
        sp_avg = {}; total = 0
        for sid, mids in species_map.items():
            avg = sum(fitnesses[i].raw_fitness for i in mids) / max(len(mids), 1)
            sp_avg[sid] = float(avg); total += abs(avg) + 1e-6
        for sid, mids in species_map.items():
            share = (abs(sp_avg.get(sid, 0)) + 1e-6) / total
            for i in mids: fitnesses[i].species_fitness = share * len(fitnesses)

    def get_generation_summary(self):
        n = len(self.current_pareto_front); total = sum(len(f) for f in self.pareto_front_history)
        return {"pareto_front_size": n, "total_pareto_genomes_seen": total}
