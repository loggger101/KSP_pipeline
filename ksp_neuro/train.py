"""Main entry point for KSP neuroevolution training pipeline."""

from __future__ import annotations
import argparse; import os; import sys; import time; import math; import copy; import random; import pickle; import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("ksp-train")


def parse_args():
    p=argparse.ArgumentParser(prog="ksp-neuroevolution",description="Genetic/evolutionary neural network training for KSP")
    p.add_argument("--algorithm",choices=["neat","genetic","cma-es"],default="neat"); p.add_argument("--network",choices=["mlp","lstm","resnet"],default="mlp")
    p.add_argument("-p","--population-size",type=int,default=200); p.add_argument("-g","--generations",type=int,default=500)
    p.add_argument("--batch-size",type=int,default=64); p.add_argument("--seed",type=int,default=None)
    p.add_argument("--task",choices=["vertical_burn","orbital_insertion","kerbin_orbit","duna_transfer","landing"],default="vertical_burn")
    p.add_argument("--mode",choices=["simulator","live","auto"],default="auto"); p.add_argument("--simulator",choices=["vectorized","single"],default="vectorized")
    p.add_argument("--distributed",action="store_true"); p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--gpus",type=str,default=""); p.add_argument("--krpc-host",type=str,default="localhost"); p.add_argument("--krpc-port",type=int,default=5000)
    p.add_argument("--output-dir",type=str,default="data/checkpoints"); p.add_argument("--resume",type=str,default=None)
    p.add_argument("--save-every",type=int,default=50); p.add_argument("--analytics-dir",type=str,default="data/analytics")
    p.add_argument("--dashboard-port",type=int,default=8501); p.add_argument("--patience",type=int,default=100)
    p.add_argument("--improvement-threshold",type=float,default=0.01); return p.parse_args()


class NeuroevolutionTrainer:
    def __init__(self, args):
        self.args=args; self.output_dir=args.output_dir; self.analytics_dir=args.analytics_dir
        os.makedirs(self.output_dir,exist_ok=True); os.makedirs(self.analytics_dir,exist_ok=True)
        from ksp_neuro.ne_engine import SpeciationManager,GenomeMutator,GenomeCrossover,NSGAIIEvaluator,TrainingAnalytics,StarterGenomeProvider,DistributedCheckpointManager
        self.speciation=SpeciationManager(); self.mutator=GenomeMutator(add_node_prob=0.35); self.crossover=GenomeCrossover(exponent=5.0)
        self.fitness_eval=NSGAIIEvaluator(); self.analytics=TrainingAnalytics(output_dir=self.analytics_dir)
        self.starter_provider=StarterGenomeProvider()
        try: from ksp_neuro.training import DistributedCheckpointManager as DCM; self.checkpoint_manager=DCM(self.output_dir)
        except Exception: self.checkpoint_manager=None

    def train(self):
        logger.info(f"KSP Neuroevolution Training — Task:{self.args.task} Algo:{self.args.algorithm} Pop:{self.args.population_size} Gens:{self.args.generations}")
        if self.args.seed is not None: random.seed(self.args.seed)
        start_gen=0; population=[]; fitnesses={}

        if self.args.resume and os.path.exists(self.args.resume):
            ckpt=self.checkpoint_manager.load_checkpoint(self.args.resume) if self.checkpoint_manager else {}
            start_gen=ckpt.get("generation",0)+1; fitnesses=ckpt.get("fitnesses",{})

        if not population:
            from ksp_neuro.ne_engine import create_random_genome
            od={"vertical_burn":12,"orbital_insertion":15,"kerbin_orbit":18,"duna_transfer":20,"landing":22}
            ad={"vertical_burn":1,"orbital_insertion":3,"kerbin_orbit":6,"duna_transfer":4,"landing":5}
            population=[create_random_genome(input_dim=od.get(self.args.task,12),output_dim=ad.get(self.args.task,3),hidden_units=[8,6]) for _ in range(self.args.population_size)]
            for i,g in enumerate(population): g.genome_id=i

        from ksp_neuro.sim_env.vectorized_sim import VectorizedOrbitSimulator as VS
        from ksp_neuro.ne_engine.torch_engine import TorchGenomeEvaluator as TGE
        sim=VS(task_name=self.args.task,n_agents=max(self.args.population_size*2,64)); evaluator=TGE(device="auto")

        best_fitness_overall=-math.inf; best_genome_overall=None; train_start=time.time()
        for gen in range(start_gen,start_gen+self.args.generations):
            gs=time.time(); pop_fit=[g.fitness for g in population] if population else []
            record=self.analytics.record_generation(gen,pop_fit,pareto_size=len(self.fitness_eval.current_pareto_front),species_count=len(self.speciation.species_map))

            gen_best=max(population,key=lambda g:g.fitness).fitness if population else 0.0
            if gen_best>best_fitness_overall: best_fitness_overall=gen_best; best_genome_overall=copy.deepcopy(max(population,key=lambda g:g.fitness) if population else None)
            logger.info(f"Gen {gen}: best={gen_best:.4f} mean={record.mean_fitness:.4f} pareto={record.pareto_front_size}")

            if record.stagnation_generations>=self.args.patience: logger.info(f"Early stop at gen {gen}"); break

            # Evolve
            for g in population: g.fitness=fitnesses.get(g.genome_id,0.0)
            self.speciation.assign_species(population); self.speciation.update_fitness()
            af=self.fitness_eval.evaluate_fitness(population,{g.genome_id:g.fitness for g in population})
            sm={}
            for f2 in af: sid=next((g.species_id for g in population if g.genome_id==f2.genome_id),0); sm.setdefault(sid,[]).append(f2.genome_id)
            self.fitness_eval.compute_adaptive_fitness(af,sm)

            new_pop=[]; total_expected=len(population)
            for sid,mids in sm.items():
                members=[g for g in population if g.genome_id in mids]; n_off=max(1,int(total_expected*len(members)/max(len(population),1)))
                sorted_m=sorted(members,key=lambda g:g.fitness,reverse=True)
                for e in sorted_m[:2]: new_pop.append(copy.deepcopy(e)); n_off-=1
                while n_off>0 and len(members)>=2:
                    p1i,p2i=self.fitness_eval.select_parents(af); p1=members[p1i]; p2=members[p2i] if p2i!=p1i else members[-1]
                    child=self.crossover.crossover(p1,p2); child=self.mutator.mutate(child)
                    nid=max((g.genome_id for g in population),default=-1)+len(new_pop); child.genome_id=nid; child.fitness=0; new_pop.append(child); n_off-=1
            population=new_pop

            # Evaluate new population
            fitnesses={}
            for si in range(0,len(population),64): bg=population[si:min(si+64,len(population))]
                sim.n_agents=max(len(bg)*2,64); obs=sim.reset(); total_r=[0.]*len(bg)
                for _ in range(3000):
                    action_batches=[]; done_count=0
                    for g in bg:
                        try: a_tensor=torch.tensor([[0.5]]*max(getattr(g,'input_dim',12),1)) if torch else torch.zeros(1,max(getattr(g,'input_dim',12),1)); actions=evaluator.evaluate_genome(g,a_tensor)
                        except Exception: actions=[0.5]*getattr(g,'output_dim',3); action_list=actions.tolist()[0]
                        action_batches.append(action_list)
                    import numpy as np; ab=np.array(action_batches,dtype=np.float32)
                    _,rewards,dones,_=sim.step(ab)
                    for i in range(len(bg)): total_r[i]+=float(rewards[0]) if not dones[0] else 0.5
                for g,r in zip(bg,total_r): fitnesses[g.genome_id]=r

            gen_elapsed=time.time()-gs; logger.info(f"Gen {gen} done in {gen_elapsed:.2f}s")

        total_time=time.time()-train_start
        if best_genome_overall and self.checkpoint_manager:
            self.checkpoint_manager.save_checkpoint(gen,population,fitnesses,best_genome=best_genome_overall)
        ap=self.analytics.save_full_analytics()
        logger.info(f"Training complete! Best:{best_fitness_overall:.4f} Time:{total_time:.2f}s")
        return {"best_fitness":best_fitness_overall,"total_generations":gen-start_gen+1,"total_time_sec":total_time,"analytics_file":ap}


def main():
    import torch; import numpy as np  # Ensure deps available
    args=parse_args(); trainer=NeuroevolutionTrainer(args); results=trainer.train()
    print("\n"+"="*60+"\nTRAINING COMPLETE\n"+"="*60)
    for k,v in results.items(): print(f"  {k}: {v:.4f}" if isinstance(v,float) else f"  {k}: {v}")

if __name__=="__main__": main()
