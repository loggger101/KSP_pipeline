"""PyTorch GPU-accelerated neural network evaluation engine for neuroevolution."""

from __future__ import annotations
import math, copy
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch; import torch.nn as nn; import torch.nn.functional as F
except ImportError:
    torch = None; nn = None; F = None


ACTIVATION_MAP = {"tanh": "Tanh", "sigmoid": "Sigmoid", "relu": "ReLU", "leaky_relu": "LeakyReLU", "linear": "Identity"}

def get_activation(name):
    if not F: raise ImportError("torch required")
    a = ACTIVATION_MAP.get(name)
    if name == "constant": return lambda x: torch.ones_like(x)
    return getattr(nn, a)()


class TorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activations=None, bias=True):
        super().__init__()
        layers = []; prev = input_dim
        acts = activations or ["tanh"] * len(hidden_dims)
        for hs, an in zip(hidden_dims, acts):
            layers.append(nn.Linear(prev, hs, bias=bias))
            if an != "linear": layers.append(get_activation(an)); prev = hs
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x): return self.network(x)


class TorchLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h_init=None):
        if not torch: raise ImportError("torch required")
        if x.dim() == 2: x = x.unsqueeze(1)
        out, (h_n, c_n) = self.lstm(x, h_init)
        return self.fc(out[:, -1, :]), {"h": h_n, "c": c_n}


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.shortcut = nn.Identity()

    def forward(self, x): return x + self.net(x)


class TorchResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_blocks=3, output_dim=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.resnet = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x): return self.output(F.tanh(self.input_proj(x)) + self.resnet(F.tanh(self.input_proj(x))))


class TorchGraphBuilder:
    """Builds PyTorch modules dynamically from NEAT genome structures."""
    def build_from_genome(self, genome):
        if not torch: raise ImportError("torch required")
        nodes_by_key = {n.key: n for n in genome.nodes}
        input_keys = sorted([n.key for n in genome.nodes if n.type == "input"])
        output_keys = sorted([n.key for n in genome.nodes if n.type == "output"])
        hidden_nodes = [nodes_by_key[k] for k in sorted(nodes_by_key.keys()) if nodes_by_key[k].type == "hidden" and getattr(nodes_by_key[k], 'enabled', True)]

        modules = {}
        for node in input_keys: modules[node] = nn.Identity()
        all_input_count = len(input_keys) + len(hidden_nodes)
        for node in hidden_nodes: modules[node.key] = nn.Sequential(nn.Linear(all_input_count, 1), get_activation(node.activation))
        for i, node in enumerate([n for n in genome.nodes if n.type == "output"]):
            modules[node.key] = nn.Sequential(nn.Linear(all_input_count, 1), get_activation(node.activation))

        return DynamicNeatModule(modules=modules, connections=[c for c in genome.connections if getattr(c,'enabled',False)], input_keys=input_keys, output_keys=output_keys)


class DynamicNeatModule(nn.Module):
    def __init__(self, modules, connections, input_keys, output_keys):
        super().__init__()
        self.modules = nn.ModuleDict({str(k): v for k, v in modules.items()})
        self.connections = connections; self.input_keys = sorted(input_keys); self.output_keys = sorted(output_keys)

    def forward(self, x):
        if not torch: raise ImportError("torch required")
        bs = x.size(0) if x.dim() == 2 else x.shape[0]
        activations = {}
        if x.dim() == 1: x = x.unsqueeze(0)
        for key in self.input_keys: idx = self.input_keys.index(key); activations[key] = x[:, idx:idx+1] if x.size(1) > idx else torch.zeros(bs, 1, device=x.device)

        processed = set(self.input_keys); max_iter = len(self.connections)*2 + 10; iteration = 0
        while len(processed) < len(set(getattr(c,'out_node',None) for c in self.connections if getattr(c,'enabled',False))) and iteration < max_iter:
            iteration += 1; progressed = False
            for conn in self.connections:
                if not getattr(conn,'enabled',False) or getattr(conn,'innovation_id',None) is None: continue
                ik, ok = conn.in_node, conn.out_node
                if ik not in activations or ok in processed: continue
                w = float(getattr(conn,'weight',1.0))
                out_mod = self.modules.get(str(ok))
                if out_mod is None: continue
                try:
                    act_out = out_mod(activations[ik] * w)
                    activations[ok] = act_out + (activations[ok] if ok in activations else torch.zeros_like(act_out))
                    processed.add(ok); progressed = True
                except Exception: pass
            if not progressed: break

        outputs = []
        for key in self.output_keys:
            ot = activations.get(key, torch.zeros(bs, 1))
            if ot.dim() == 1: ot = ot.unsqueeze(0)
            outputs.append(ot)
        return torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]


class TorchGenomeEvaluator:
    def __init__(self, device="auto", use_half_precision=False):
        if not torch: raise ImportError("torch required")
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else (device or "cpu"))
        self.use_half = use_half_precision and self.device.type == "cuda"
        self.builder = TorchGraphBuilder(); self._module_cache = {}

    def evaluate_genome(self, genome, observation_batch):
        if isinstance(observation_batch, torch.Tensor): obs = observation_batch.to(self.device)
        else: obs = torch.tensor(observation_batch, dtype=torch.float32, device=self.device)
        if self.use_half: obs = obs.half()
        module = self._get_or_build_module(genome)
        with torch.no_grad(): output = module(obs)
        return output.cpu() if self.device.type == "cuda" else output

    def _get_or_build_module(self, genome):
        gid = getattr(genome,'genome_id',-1)
        if gid in self._module_cache: return self._module_cache[gid]
        module = self.builder.build_from_genome(genome)
        if self.use_half: module = module.half()
        module = module.to(self.device); module.eval(); self._module_cache[gid] = module; return module


def create_network(network_type, input_dim, hidden_dims, output_dim, device="auto"):
    if not torch: raise ImportError("torch required")
    if network_type == "mlp": return TorchMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    elif network_type == "lstm":
        h = hidden_dims[0] if hidden_dims else 32; nl = hidden_dims[1] if len(hidden_dims)>1 else 1
        return TorchLSTM(input_dim=input_dim, hidden_dim=h, output_dim=output_dim, num_layers=nl)
    elif network_type == "resnet":
        hd = hidden_dims[0] if hidden_dims else 64; nb = hidden_dims[1] if len(hidden_dims)>1 else 3
        return TorchResNet(input_dim=input_dim, hidden_dim=hd, num_blocks=nb, output_dim=output_dim)


def get_gpu_info():
    if not torch: return {"available": False}
    info = {"available": torch.cuda.is_available(), "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0, "devices": []}
    if info["available"]:
        for i in range(info["device_count"]):
            p = torch.cuda.get_device_properties(i); info["devices"].append({"index":i,"name":p.name,"memory_total_gb":round(p.total_memory/1e9,2)})
    return info

def get_available_device(device="auto"):
    if not torch: return None
    return torch.device("cuda" if device == "auto" and torch.cuda.is_available() else (device or "cpu"))
