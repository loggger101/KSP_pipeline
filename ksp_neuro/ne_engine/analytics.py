"""Training analytics engine — learning curves, convergence detection, diversity metrics."""

from __future__ import annotations
import json; import math; import os; import time
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class GenerationRecord:
    generation: int = 0
    timestamp: float = 0.0
    mean_fitness: float = 0.0
    std_fitness: float = 0.0
    best_fitness: float = 0.0
    worst_fitness: float = 0.0
    pareto_front_size: int = 0
    total_agents_evaluated: int = 0
    species_count: int = 0
    fitness_diversity: float = 0.0
    improvement_rate: float = 0.0
    stagnation_generations: int = 0
    eval_time_sec: float = 0.0
    total_elapsed_sec: float = 0.0


class TrainingAnalytics:
    def __init__(self, output_dir="data/analytics"): self.output_dir=output_dir; os.makedirs(output_dir,exist_ok=True); self.records=[]; self.fitness_history=[]

    def record_generation(self, generation, fitnesses, pareto_size=0, species_count=0, genome_complexities=None, eval_time=0.0):
        if not fitnesses: return GenerationRecord(generation=generation)
        sf = sorted(fitnesses); n=len(sf); mean=sum(sf)/n; var=sum((f-mean)**2 for f in sf)/max(n,1)
        imp_rate = (sf[-1]-self.records[-1].best_fitness)/abs(self.records[-1].best_fitness) if self.records else 0.0
        rec = GenerationRecord(generation=generation,timestamp=time.time(),mean_fitness=mean,std_fitness=math.sqrt(var),
            best_fitness=sf[-1],worst_fitness=sf[0],pareto_front_size=pareto_size,total_agents_evaluated=n,species_count=species_count,
            fitness_diversity=math.sqrt(var)/max(abs(mean),1e-6),improvement_rate=imp_rate,eval_time_sec=eval_time)
        self.records.append(rec); self.fitness_history.append(sf[-1])
        if self.records and imp_rate < 0.01: rec.stagnation_generations = (self.records[-2].stagnation_generations+1) if len(self.records)>1 else 1
        self._save_record(rec)
        return rec

    def get_learning_curve(self, window_size=50):
        h = list(self.fitness_history); alpha=2.0/(window_size+1); sm=[h[0]]
        for i in range(1,len(h)): sm.append(alpha*h[i]+(1-alpha)*sm[-1])
        return [{"generation":g,"raw_fitness":r,"smoothed_fitness":s} for g,r,s in zip(range(len(sm)),h,sm)]

    def get_diversity_metrics(self):
        if not self.records: return {}
        sc=[r.species_count for r in self.records]; ps=[r.pareto_front_size for r in self.records]
        return {"current_species_count":sc[-1],"mean_species_count":sum(sc)/len(sc),"current_pareto_size":ps[-1],"mean_pareto_size":sum(ps)/len(ps)}

    def get_best_agents(self, n=5):
        sr = sorted(self.records,key=lambda r:r.best_fitness,reverse=True)
        return [{"generation":r.generation,"best_fitness":r.best_fitness,"mean_fitness":r.mean_fitness} for r in sr[:n]]

    def get_summary(self):
        if not self.records:
            return {"status": "no_data"}
        last_rec = self.records[-1]; first_rec = self.records[0]
        tg = len(self.records); te = last_rec.timestamp - first_rec.timestamp if tg > 1 else 0
        return {"status": "training", "total_generations": tg,
                "best_fitness_overall": max(r.best_fitness for r in self.records),
                "current_best": last_rec.best_fitness, "time_elapsed_sec": te}

    def save_full_analytics(self):
        lc = self.get_learning_curve()[-500:]; best=self.get_best_agents(10); summary=self.get_summary(); div=self.get_diversity_metrics()
        out={"summary":summary,"learning_curve":lc,"diversity_metrics":div,"best_agents":best}; path=os.path.join(self.output_dir,"training_analytics.json")
        with open(path,"w") as f: json.dump(out,f,indent=2); return path

    def _save_record(self, rec): pass  # Simplified for now
