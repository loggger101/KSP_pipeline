"""NEAT genome representation, speciation, crossover, and mutation operations."""

from __future__ import annotations
import math, random, copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GenomeNode:
    key: int; type: str = "hidden"; activation: str = "tanh"; enabled: bool = True; innovation_id: Optional[int] = None

@dataclass
class GenomeConnection:
    in_node: int; out_node: int; weight: float = 0.0; enabled: bool = True; innovation_id: Optional[int] = None
    @property
    def key(self) -> Tuple[int, int]: return (self.in_node, self.out_node)

@dataclass
class NEATGenome:
    genome_id: int = 0; fitness: float = 0.0; avg_fitness: float = 0.0
    nodes: List[GenomeNode] = field(default_factory=list); connections: List[GenomeConnection] = field(default_factory=list)
    species_id: Optional[int] = None; compatibility_distance: float = math.inf
    input_dim: int = 0; output_dim: int = 0

    @property
    def hidden_count(self): return sum(1 for n in self.nodes if n.type == "hidden" and n.enabled)
    @property
    def enabled_connections(self): return [c for c in self.connections if c.enabled]
    @property
    def complexity_cost(self): return 0.01 * len(self.nodes) + 0.025 * len(self.enabled_connections) ** 2

    def topological_order(self) -> List[int]:
        ec = [c for c in self.connections if c.enabled]
        if not ec: return [n.key for n in self.nodes]
        keys = {n.key for n in self.nodes}; indeg = {k: 0 for k in keys}; adj = {k: [] for k in keys}
        for c in ec:
            if c.in_node in keys and c.out_node in keys:
                indeg[c.out_node] += 1; adj[c.in_node].append(c.out_node)
        q = [k for k, d in indeg.items() if d == 0]; order = []
        while q:
            nd = q.pop(0); order.append(nd)
            for nb in adj[nd]:
                indeg[nb] -= 1
                if indeg[nb] == 0: q.append(nb)
        return order + [k for k in keys if k not in order]

    def to_dict(self):
        return {"genome_id": self.genome_id, "fitness": self.fitness, "nodes": [{"key": n.key, "type": n.type, "activation": n.activation} for n in self.nodes],
                "connections": [{"in_node": c.in_node, "out_node": c.out_node, "weight": c.weight, "innovation_id": c.innovation_id} for c in self.connections],
                "species_id": self.species_id, "input_dim": self.input_dim, "output_dim": self.output_dim}

    @classmethod
    def from_dict(cls, d):
        nodes = [GenomeNode(key=n["key"], type=n.get("type","hidden"), activation=n.get("activation","tanh")) for n in d["nodes"]]
        conns = []
        for c in d.get("connections", []):
            ii = None
            if isinstance(c.get("innovation_id"), str):
                try: ii = int(c["innovation_id"])
                except (ValueError, TypeError): pass
            elif c.get("innovation_id") is not None: ii = int(c["innovation_id"])
            conns.append(GenomeConnection(in_node=c["in_node"], out_node=c["out_node"], weight=c.get("weight", 0.0), innovation_id=ii))
        g = cls(genome_id=d["genome_id"]); g.nodes = nodes; g.connections = conns
        g.species_id = d.get("species_id"); g.input_dim = len([n for n in nodes if n.type == "input"])
        g.output_dim = len([n for n in nodes if n.type == "output"])
        return g


@dataclass
class Species:
    species_id: int; representative: Optional[NEATGenome] = None; members: List[NEATGenome] = field(default_factory=list)
    avg_fitness: float = 0.0; generation_born: int = 0; stagnation_generations: int = 0

class SpeciationManager:
    def __init__(self, compatibility_threshold=3.0):
        self.threshold = compatibility_threshold
        self.species_counter = 0
        self.species_map = {}
        self.genomes_by_species = {}
    def assign_species(self, genomes):
        for genome in genomes:
            sid = self._find_best(genome)
            if sid is not None and sid in self.species_map:
                genome.species_id = sid; self.species_map[sid].members.append(genome); self.species_map[sid].representative = genome
                self.genomes_by_species.setdefault(sid, []).append(genome)
            else:
                self.species_counter += 1; s = Species(species_id=self.species_counter, representative=genome)
                self.species_map[self.species_counter] = s; self.genomes_by_species.setdefault(self.species_counter, [])
                genome.species_id = self.species_counter

    def _find_best(self, genome):
        best_dist, best_id = math.inf, None
        for sid, sp in self.species_map.items():
            if sp.representative is None: continue
            dist = self._compat(genome, sp.representative)
            if dist < best_dist and dist <= self.threshold: best_dist = dist; best_id = sid
        return best_id

    def _compat(self, g1, g2):
        c1 = {c.innovation_id: c.weight for c in g1.connections if hasattr(c,'innovation_id') and c.innovation_id is not None}
        c2 = {c.innovation_id: c.weight for c in g2.connections if hasattr(c,'innovation_id') and c.innovation_id is not None}
        all_ids = set(c1) | set(c2); matching = set(c1) & set(c2)
        excess = len(all_ids - matching); wd = sum(abs(c1[k] - c2[k]) for k in matching) / max(len(matching), 1)
        n = max(len(g1.connections), len(g2.connections))
        return (1.0 * excess / max(n, 1)) + (0.5 * wd) if n > 0 else 0.0

    def update_fitness(self):
        for sid, sp in self.species_map.items():
            if sp.members: sp.avg_fitness = sum(m.fitness for m in sp.members) / len(sp.members)

    def get_fittest_genome(self):
        best_g, best_f = None, -math.inf
        for sp in self.species_map.values():
            if sp.representative and sp.representative.fitness > best_f: best_f = sp.representative.fitness; best_g = sp.representative
        return best_g


class GenomeMutator:
    def __init__(self, mutate_weight_prob=0.9, weight_mutation_rate=0.8, weight_mutation_std=1.0, add_node_prob=0.35, add_conn_prob=0.25):
        self.mutate_weight_prob = mutate_weight_prob; self.weight_mutation_rate = weight_mutation_rate; self.weight_mutation_std = weight_mutation_std
        self.add_node_prob = add_node_prob; self.add_conn_prob = add_conn_prob

    def mutate(self, genome):
        mutant = copy.deepcopy(genome)
        if random.random() < self.mutate_weight_prob:
            for conn in mutant.connections:
                if hasattr(conn,'enabled') and conn.enabled and random.random() < self.weight_mutation_rate: conn.weight += random.gauss(0, self.weight_mutation_std)
        if random.random() < self.add_node_prob: self._add_node(mutant)
        elif random.random() < self.add_conn_prob: self._add_conn(mutant)
        return mutant

    def _add_node(self, genome):
        ec = [c for c in genome.connections if hasattr(c,'enabled') and c.enabled]
        if not ec: return
        chosen = random.choice(ec); nk = max((n.key for n in genome.nodes), default=0) + 1
        genome.nodes.append(GenomeNode(key=nk, type="hidden", activation="tanh"))
        chosen.enabled = False
        genome.connections.append(GenomeConnection(in_node=chosen.in_node, out_node=nk, weight=1.0))
        genome.connections.append(GenomeConnection(in_node=nk, out_node=chosen.out_node, weight=chosen.weight))

    def _add_conn(self, genome):
        en = [n for n in genome.nodes if hasattr(n,'enabled') and n.enabled]
        if len(en) < 2: return
        for _ in range(50):
            i, o = random.choice(en), random.choice(en)
            if i.key == o.key or any(c.in_node == i.key and c.out_node == o.key for c in genome.connections): continue
            genome.connections.append(GenomeConnection(in_node=i.key, out_node=o.key, weight=random.gauss(0, 1.0)))
            return


class GenomeCrossover:
    def __init__(self, exponent=5.0): self.exponent = exponent

    def crossover(self, p1, p2):
        if p1.species_id != p2.species_id: raise ValueError("Same species required")
        child = NEATGenome(genome_id=-1); child.input_dim = max(p1.input_dim, p2.input_dim); child.output_dim = max(p1.output_dim, p2.output_dim)
        f1, f2 = max(p1.fitness, 1e-6), max(p2.fitness, 1e-6)
        w1 = f1 ** self.exponent / (f1 ** self.exponent + f2 ** self.exponent)
        c1m = {c.innovation_id: c for c in p1.connections if hasattr(c,'innovation_id') and c.innovation_id is not None}
        c2m = {c.innovation_id: c for c in p2.connections if hasattr(c,'innovation_id') and c.innovation_id is not None}
        all_ids = set(c1m) | set(c2m); matching = set(c1m) & set(c2m)
        for mid in matching: child.connections.append(copy.deepcopy(random.choice([c1m[mid], c2m[mid]]) if random.random() < w1 else [c2m[mid]]))
        fitter_c = p1 if p1.fitness >= p2.fitness else p2
        for eid in all_ids - matching:
            found = [c for c in fitter_c.connections if hasattr(c,'innovation_id') and c.innovation_id == eid]
            if found: child.connections.append(copy.deepcopy(found[0]))
        n1 = {n.key: n for n in p1.nodes}; n2 = {n.key: n for n in p2.nodes}
        for k in set(n1) | set(n2): child.nodes.append(copy.deepcopy(random.choice([n1[k], n2[k]]) if random.random() < w1 else [n2[k]]))
        return child


def create_random_genome(input_dim, output_dim, hidden_units=None, max_connections=50):
    genome = NEATGenome(genome_id=0, input_dim=input_dim, output_dim=output_dim)
    for i in range(input_dim): genome.nodes.append(GenomeNode(key=i, type="input"))
    bias_key = input_dim; genome.nodes.append(GenomeNode(key=bias_key, type="bias")); genome.bias_nodes = [bias_key]
    ni = input_dim + 1
    if hidden_units is None: hidden_units = [8]
    for sz in hidden_units:
        for _ in range(sz): genome.nodes.append(GenomeNode(key=ni, type="hidden", activation="tanh")); ni += 1
    out_start = ni
    for i in range(output_dim): genome.nodes.append(GenomeNode(key=out_start + i, type="output"))

    inp_n = [n for n in genome.nodes if n.type == "input"] + [genome.nodes[bias_key]]
    hid_n = [n for n in genome.nodes if n.type == "hidden"]; out_n = [n for n in genome.nodes if n.type == "output"]
    for i_node in inp_n:
        for h in hid_n:
            if random.random() < 0.7: genome.connections.append(GenomeConnection(in_node=i_node.key, out_node=h.key, weight=random.gauss(0,1), innovation_id=(i_node.key, h.key)))
    for h in hid_n:
        for o in out_n:
            if random.random() < 0.5: genome.connections.append(GenomeConnection(in_node=h.key, out_node=o.key, weight=random.gauss(0,1), innovation_id=(h.key, o.key)))
    return genome


def create_starter_genome(task_name, preset="mlp"):
    obs_d = {"vertical_burn": 12, "orbital_insertion": 15, "kerbin_orbit": 18, "duna_transfer": 20, "landing": 22}
    act_d = {"vertical_burn": 1, "orbital_insertion": 3, "kerbin_orbit": 6, "duna_transfer": 4, "landing": 5}
    return create_random_genome(input_dim=obs_d.get(task_name, 12), output_dim=act_d.get(task_name, 3), hidden_units=[8, 6])


def serialize_weights(genome):
    wm = {}
    for conn in genome.connections:
        if hasattr(conn,'innovation_id') and conn.innovation_id is not None:
            key_str = f"{conn.in_node},{conn.out_node}"
            wm[key_str] = conn.weight
    return {"weights": wm, "nodes": [{"key": n.key, "type": n.type, "activation": n.activation} for n in genome.nodes],
            "connections": [{"in_node": c.in_node, "out_node": c.out_node, "innovation_id": str(c.innovation_id) if isinstance(c.innovation_id, int) else None} for c in genome.connections]}


def deserialize_weights(data):
    if not data.get("nodes"): return [], []
    nodes = [GenomeNode(key=n["key"], type=n.get("type","hidden"), activation=n.get("activation","tanh")) for n in data["nodes"]]
    wm_int = {}; wm_conn = {}
    for k, v in data.get("weights", {}).items():
        try: wm_int[int(k)] = v
        except (ValueError, TypeError): pass
        parts = str(k).split(",")
        if len(parts) == 2:
            try: wm_conn[(int(parts[0]), int(parts[1]))] = v
            except (ValueError, TypeError): pass
    conns = []
    for c in data["connections"]:
        ii = None; raw = c.get("innovation_id")
        if isinstance(raw, str):
            try: ii = int(raw)
            except (ValueError, TypeError): pass
        elif raw is not None: ii = int(raw)
        conn = GenomeConnection(in_node=c["in_node"], out_node=c["out_node"], innovation_id=ii, weight=c.get("weight", 0.0))
        if ii and ii in wm_int: conn.weight = wm_int[ii]
        elif (c["in_node"], c["out_node"]) in wm_conn: conn.weight = wm_conn[(c["in_node"], c["out_node"])]
        conns.append(conn)
    return nodes, conns
