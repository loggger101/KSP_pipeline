"""Neuroevolution engine — NEAT genome management, PyTorch evaluation, multi-objective optimization."""

from ksp_neuro.ne_engine import neat_evaluator as _neat
from ksp_neuro.ne_engine import torch_engine as _torch
from ksp_neuro.ne_engine import multi_obj as _multi
from ksp_neuro.ne_engine import analytics as _analytics
from ksp_neuro.ne_engine import starter_agents as _starter

# Re-export key symbols
GenomeNode = _neat.GenomeNode; GenomeConnection = _neat.GenomeConnection
NEATGenome = _neat.NEATGenome; Species = _neat.Species; SpeciationManager = _neat.SpeciationManager
GenomeMutator = _neat.GenomeMutator; GenomeCrossover = _neat.GenomeCrossover
create_random_genome = _neat.create_random_genome; create_starter_genome = _neat.create_starter_genome
serialize_weights = _neat.serialize_weights; deserialize_weights = _neat.deserialize_weights

TorchMLP = _torch.TorchMLP; TorchLSTM = _torch.TorchLSTM; TorchResNet = _torch.TorchResNet
ResBlock = _torch.ResBlock; DynamicNeatModule = _torch.DynamicNeatModule
TorchGraphBuilder = _torch.TorchGraphBuilder; TorchGenomeEvaluator = _torch.TorchGenomeEvaluator
create_network = _torch.create_network; get_gpu_info = _torch.get_gpu_info
get_available_device = _torch.get_available_device

Objective = _multi.Objective; AgentFitness = _multi.AgentFitness; NSGAIIEvaluator = _multi.NSGAIIEvaluator
GenerationRecord = _analytics.GenerationRecord; TrainingAnalytics = _analytics.TrainingAnalytics
StarterAgent = _starter.StarterAgent; STARTER_AGENTS = _starter.STARTER_AGENTS
StarterGenomeProvider = _starter.StarterGenomeProvider
get_starter_genome = _starter.get_starter_genome; ensure_all_weights_exist = _starter.ensure_all_weights_exist

__all__ = [
    "GenomeNode", "GenomeConnection", "NEATGenome", "Species", "SpeciationManager",
    "GenomeMutator", "GenomeCrossover", "create_random_genome", "create_starter_genome",
    "serialize_weights", "deserialize_weights", "TorchMLP", "TorchLSTM", "TorchResNet",
    "ResBlock", "DynamicNeatModule", "TorchGraphBuilder", "TorchGenomeEvaluator",
    "create_network", "get_gpu_info", "get_available_device", "Objective", "AgentFitness",
    "NSGAIIEvaluator", "GenerationRecord", "TrainingAnalytics", "StarterAgent", "STARTER_AGENTS",
    "StarterGenomeProvider", "get_starter_genome", "ensure_all_weights_exist",
]
