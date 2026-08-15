"""Pre-trained starter agent definitions and NEAT integration helpers."""

from __future__ import annotations
import os; import pickle; import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ksp_neuro.ne_engine.neat_evaluator import (NEATGenome, GenomeNode, GenomeConnection, create_random_genome, create_starter_genome, serialize_weights, deserialize_weights)


WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

@dataclass
class StarterAgent:
    name: str
    task_name: str
    description: str
    network_type: str = "mlp"
    hidden_dims: Optional[List[int]] = None
    weight_file: Optional[str] = None
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [8, 6]
    @property
    def has_pretrained_weights(self): return self.weight_file is not None and os.path.exists(os.path.join(WEIGHTS_DIR, self.weight_file))
    def get_genome(self):
        if self.has_pretrained_weights and self.weight_file:
            path = os.path.join(WEIGHTS_DIR, self.weight_file)
            try:
                with open(path,"rb") as f: data=pickle.load(f); nodes,conns=deserialize_weights(data)
                g=NEATGenome(genome_id=0); g.nodes=nodes; g.connections=conns; g.input_dim=len([n for n in nodes if n.type=="input"]); g.output_dim=len([n for n in nodes if n.type=="output"])
                return g
            except: pass
        return self._generate_random_genome()
    def _generate_random_genome(self):
        od={"vertical_burn":12,"orbital_insertion":15,"kerbin_orbit":18,"duna_transfer":20,"landing":22}
        ad={"vertical_burn":1,"orbital_insertion":3,"kerbin_orbit":6,"duna_transfer":4,"landing":5}
        return create_random_genome(input_dim=od.get(self.task_name,12),output_dim=ad.get(self.task_name,3),hidden_units=self.hidden_dims)

STARTER_AGENTS: Dict[str, StarterAgent] = {
    "vertical_burn": StarterAgent(name="Vertical Burn Agent",task_name="vertical_burn",description="Basic vertical ascent policy.",network_type="mlp",hidden_dims=[8,6],weight_file="vertical_burn.pkl"),
    "orbital_insertion": StarterAgent(name="Orbital Insertion Agent",task_name="orbital_insertion",description="Reaches circular orbit around Kerbin at ~100km.",network_type="mlp",hidden_dims=[10,8],weight_file="orbital_insertion.pkl"),
    "kerbin_orbit": StarterAgent(name="Kerbin Orbit Stabilization Agent",task_name="kerbin_orbit",description="Achieves and maintains stable circular orbit around Kerbin.",network_type="mlp",hidden_dims=[12,10,8],weight_file="kerbin_orbit.pkl"),
    "duna_transfer": StarterAgent(name="Duna Transfer Agent",task_name="duna_transfer",description="Performs Hohmann transfer to Duna intercept.",network_type="mlp",hidden_dims=[12,10],weight_file="duna_transfer.pkl"),
    "landing": StarterAgent(name="Powered Landing Agent",task_name="landing",description="Executes powered descent landing on any body's surface.",network_type="lstm",hidden_dims=[16,2],weight_file="landing.pkl"),
}

class StarterGenomeProvider:
    def __init__(self): self.agents=dict(STARTER_AGENTS)
    def get_starter_genome(self, task_name="vertical_burn", preset="mlp"):
        if task_name in self.agents: return self.agents[task_name].get_genome()
        return create_starter_genome(task_name, preset)
    def get_all_starter_names(self): return list(self.agents.keys())
    def ensure_weights_exist(self):
        for name,agent in self.agents.items():
            if not agent.has_pretrained_weights: generate_starter_weights(name)

def generate_starter_weights(agent_name="vertical_burn"):
    agent=STARTER_AGENTS.get(agent_name)
    if not agent: return None
    genome=create_random_genome(input_dim=12,output_dim=1,hidden_units=[8,6])
    data=serialize_weights(genome); path=os.path.join(WEIGHTS_DIR,agent.weight_file or f"{agent_name}_weights.pkl")
    with open(path,"wb") as f: pickle.dump(data,f)
    return path

def get_starter_genome(task_name="vertical_burn",preset="mlp"): return StarterGenomeProvider().get_starter_genome(task_name,preset)
def ensure_all_weights_exist(): StarterGenomeProvider().ensure_weights_exist()
