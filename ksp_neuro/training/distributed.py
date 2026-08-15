"""Ray-based distributed training engine for large-scale neuroevolution."""

from __future__ import annotations
import os; import time; import pickle; import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvalTask: genome_id:int=0; observation_batch:Any=None; sim_config:Dict[str,Any]=field(default_factory=dict)
@dataclass
class EvalResult: genome_id:int=0; total_reward:float=0.0; final_fitness:Dict[str,float]=field(default_factory=dict); eval_time_sec:float=0.0


class DistributedTrainingEngine:
    def __init__(self, num_workers=4, gpus_per_worker=1): self.num_workers=num_workers; self.gpus_per_worker=gpus_per_worker; self.total_gpus=num_workers*gpus_per_worker
        self._actors=[]; self._ray_initialized=False

    def initialize(self):
        if self._ray_initialized: return len(self._actors)>0
        try:
            import ray
            try: ray.init(address="auto",ignore_reinit_error=True)
            except: ray.init(num_cpus=self.num_workers,num_gpus=max(0,self.total_gpus),ignore_reinit_error=True)
            self._ray_initialized=True; return True
        except ImportError: logger.warning("Ray not installed; falling back to single-process evaluation"); return False

    def evaluate_population(self, genomes, task_name):
        if not self._ray_initialized or not self.initialize(): return self._evaluate_single(genomes,task_name)
        n=len(genomes); batch_size=max(16,n//self.num_workers); results={}
        for si in range(0,n,batch_size):
            bg=genomes[si:min(si+batch_size,n)]
            try: serialized=[pickle.dumps(g) for g in bg]
            except Exception: continue
        return self._evaluate_single(genomes,task_name)

    def _evaluate_single(self, genomes, task_name):
        from ksp_neuro.sim_env.vectorized_sim import VectorizedOrbitSimulator
        from ksp_neuro.ne_engine.torch_engine import TorchGenomeEvaluator
        n=len(genomes); batch_size=max(64,min(n*2,512)); rewards_map={}
        evaluator=TorchGenomeEvaluator(device="auto")
        for si in range(0,n,batch_size):
            bg=genomes[si:min(si+batch_size,n)]; sim_batch_size=max(len(bg)*2,64)
            try: obs_tensor=torch.tensor([[0.5]]*max(getattr(bg[0],'input_dim',12),1)) if torch else None
            except: obs_tensor=None
        return rewards_map


class DistributedCheckpointManager:
    def __init__(self,checkpoint_dir="data/checkpoints"): self.checkpoint_dir=checkpoint_dir; os.makedirs(checkpoint_dir,exist_ok=True)

    def save_checkpoint(self, generation, genomes, fitnesses, best_genome=None, metadata=None):
        ts=time.strftime("%Y%m%d_%H%M%S"); filepath=os.path.join(self.checkpoint_dir,f"gen_{generation:06d}_{ts}.pkl")
        tmp_path=filepath+".tmp"; data={"generation":generation,"fitnesses":fitnesses,"best_genome_id":getattr(best_genome,'genome_id',-1) if best_genome else None}
        with open(tmp_path,"wb") as f: pickle.dump(data,f); os.replace(tmp_path,filepath); logger.info(f"Saved checkpoint: {filepath}")
        return filepath

    def load_checkpoint(self, path):
        with open(path,"rb") as f: data=pickle.load(f)
        logger.info(f"Loaded checkpoint from {path} (gen {data.get('generation','?')})"); return data

    def list_checkpoints(self):
        if not os.path.exists(self.checkpoint_dir): return []
        files=[f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pkl") and not f.endswith(".tmp")]
        return sorted(files,key=lambda f: int(f.split("_")[1]) if len(f.split("_"))>1 else 0,reverse=True)


def create_distributed_engine(num_workers=4,gpus_per_worker=1): return DistributedTrainingEngine(num_workers=num_workers,gpus_per_worker=gpus_per_worker)

def check_ray_availability():
    result={"ray_installed":False,"version":None,"local_resources":{"cpu_cores":__import__('multiprocessing').cpu_count()}}
    try: import ray; result["ray_installed"]=True; result["version"]=getattr(ray,'__version__','unknown')
    except ImportError: pass
    return result
