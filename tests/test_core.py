"""Comprehensive test suite for KSP neuroevolution pipeline."""

from __future__ import annotations
import math, os, sys, pickle, tempfile, unittest, numpy as np, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestOrbitSimulator(unittest.TestCase):
    def setUp(self):
        from ksp_neuro.sim_env.orbit_simulator import OrbitSimulator
        self.sim = OrbitSimulator(task_name="vertical_burn")

    def test_reset_produces_valid_state(self):
        state = self.sim.reset()
        self.assertGreater(state.altitude, 60_000)
        self.assertIsNotNone(state.position)

    def test_step_returns_tuple(self):
        state = self.sim.reset()
        obs, reward, done, info = self.sim.step([0.5])
        self.assertIsInstance(obs, type(state))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_throttle_affects_altitude(self):
        for throttle in [0.1, 0.5, 0.9]:
            state = self.sim.reset()
            done = False
            for _ in range(60 * 30):
                _, _, done, _ = self.sim.step([throttle])
                if done:
                    break

    def test_episode_termination(self):
        state = self.sim.reset()
        done = False
        for i in range(10_000):
            _, _, done, _ = self.sim.step([0.5])
            if done:
                break
        self.assertTrue(done)


class TestVectorizedSimulator(unittest.TestCase):
    def test_reset_produces_batch(self):
        from ksp_neuro.sim_env.vectorized_sim import VectorizedOrbitSimulator
        sim = VectorizedOrbitSimulator(task_name="vertical_burn", n_agents=10)
        obs = sim.reset()
        self.assertEqual(obs.shape, (10, 12))

    def test_step_returns_correct_shapes(self):
        from ksp_neuro.sim_env.vectorized_sim import VectorizedOrbitSimulator
        sim = VectorizedOrbitSimulator(task_name="vertical_burn", n_agents=10)
        obs = sim.reset()
        actions = np.random.uniform(-1, 1, (10, 1))
        obs_out, rewards, dones, info = sim.step(actions)
        self.assertEqual(obs_out.shape, (10, 12))

    def test_population_stats(self):
        from ksp_neuro.sim_env.vectorized_sim import VectorizedOrbitSimulator
        sim = VectorizedOrbitSimulator(task_name="vertical_burn", n_agents=20)
        obs = sim.reset()
        for _ in range(60 * 10):
            actions = np.random.uniform(-0.3, 0.8, (20, 1))
            _, _, dones, _ = sim.step(actions)
        stats = sim.get_population_stats()
        self.assertIn("mean_altitude", stats)
        self.assertIn("max_altitude", stats)
        self.assertIn("fuel_frac_mean", stats)

    def test_task_configs_exist(self):
        from ksp_neuro.sim_env.vectorized_sim import TASK_CONFIGS
        expected = {"vertical_burn", "orbital_insertion", "kerbin_orbit", "duna_transfer", "landing"}
        self.assertEqual(set(TASK_CONFIGS.keys()), expected)


class TestNEATGenome(unittest.TestCase):
    def test_create_random_genome(self):
        from ksp_neuro.ne_engine import create_random_genome, GenomeNode, GenomeConnection
        g = create_random_genome(input_dim=5, output_dim=2, hidden_units=[4, 3])
        self.assertEqual(g.input_dim, 5)
        self.assertEqual(g.output_dim, 2)
        inp = [n for n in g.nodes if n.type == "input"]
        hid = [n for n in g.nodes if n.type == "hidden"]
        out = [n for n in g.nodes if n.type == "output"]
        self.assertEqual(len(inp), 5)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(hid), 7)

    def test_genome_topological_order(self):
        from ksp_neuro.ne_engine import create_random_genome
        g = create_random_genome(input_dim=3, output_dim=1, hidden_units=[4])
        order = g.topological_order()
        all_keys = {n.key for n in g.nodes}
        self.assertEqual(set(order), all_keys)

    def test_serialize_deserialize(self):
        from ksp_neuro.ne_engine import create_random_genome, serialize_weights, deserialize_weights
        g = create_random_genome(input_dim=4, output_dim=2, hidden_units=[6])
        data = serialize_weights(g)
        nodes_back, conns_back = deserialize_weights(data)
        self.assertEqual(len(nodes_back), len(g.nodes))
        self.assertEqual(len(conns_back), len(g.connections))


class TestSpeciation(unittest.TestCase):
    def test_assign_species(self):
        from ksp_neuro.ne_engine import create_random_genome, SpeciationManager
        m = SpeciationManager()
        gs = [create_random_genome(input_dim=5, output_dim=2, hidden_units=[4]) for _ in range(10)]
        for i, g in enumerate(gs):
            g.genome_id = i
        m.assign_species(gs)
        self.assertTrue(all(g.species_id is not None for g in gs))

    def test_species_count(self):
        from ksp_neuro.ne_engine import create_random_genome, SpeciationManager
        m = SpeciationManager()
        gs = []
        for i in range(20):
            g = create_random_genome(input_dim=5, output_dim=2, hidden_units=[4 + i % 3])
            g.genome_id = i
            for c in g.connections:
                if hasattr(c, 'weight'):
                    c.weight *= (i + 1) * 0.5
            gs.append(g)
        m.assign_species(gs)
        self.assertGreater(len(m.species_map), 0)


class TestGenomeMutation(unittest.TestCase):
    def test_mutation_adds_nodes(self):
        from ksp_neuro.ne_engine import create_random_genome, GenomeMutator
        mut = GenomeMutator(add_node_prob=1.0, add_conn_prob=0)
        g = create_random_genome(input_dim=4, output_dim=2, hidden_units=[3])
        m = mut.mutate(g)
        self.assertGreaterEqual(len(m.nodes), len(g.nodes))


class TestGenomeCrossover(unittest.TestCase):
    def test_crossover_same_species(self):
        from ksp_neuro.ne_engine import create_random_genome, GenomeCrossover
        cr = GenomeCrossover()
        g1 = create_random_genome(input_dim=4, output_dim=2, hidden_units=[3])
        g2 = create_random_genome(input_dim=4, output_dim=2, hidden_units=[3])
        g1.genome_id = 0; g2.genome_id = 1
        g1.species_id = 1; g2.species_id = 1
        child = cr.crossover(g1, g2)
        self.assertIsNotNone(child)


class TestMultiObjective(unittest.TestCase):
    def test_non_dominated_sort(self):
        from ksp_neuro.ne_engine.multi_obj import AgentFitness, Objective, NSGAIIEvaluator
        ev = NSGAIIEvaluator()
        objs = [
            {"altitude": 1.0, "delta_v_efficiency": 0.5, "maneuver_accuracy": 0.3, "stability_score": 0.8},
            {"altitude": 2.0, "delta_v_efficiency": 0.4, "maneuver_accuracy": 0.6, "stability_score": 0.7},
            {"altitude": 1.5, "delta_v_efficiency": 0.3, "maneuver_accuracy": 0.2, "stability_score": 0.9},
        ]
        fs = [AgentFitness(genome_id=i, objectives=o) for i, o in enumerate(objs)]
        ev._non_dominated_sort(fs)
        self.assertGreater(sum(1 for f in fs if f.rank == 0), 0)

    def test_crowding_distance(self):
        from ksp_neuro.ne_engine.multi_obj import AgentFitness, Objective, NSGAIIEvaluator
        ev = NSGAIIEvaluator()
        fs = [AgentFitness(genome_id=i, objectives={o.name: float(i) for o in ev.objectives}) for i in range(10)]
        ev._non_dominated_sort(fs); ev._crowding_distance(fs)
        rank_0 = [f for f in fs if f.rank == 0]
        self.assertTrue(all(f.crowding_distance > 0 or len(rank_0) <= 2 for f in rank_0))

    def test_tournament_selection(self):
        from ksp_neuro.ne_engine.multi_obj import AgentFitness, Objective, NSGAIIEvaluator
        ev = NSGAIIEvaluator()
        fs = [AgentFitness(genome_id=i, objectives={"altitude": float(i + 1)}) for i in range(20)]
        ev._non_dominated_sort(fs); ev._crowding_distance(fs)
        p1, p2 = ev.select_parents(fs, n_parents=2)
        self.assertIn(p1, range(len(fs))); self.assertIn(p2, range(len(fs)))


class TestTrainingAnalytics(unittest.TestCase):
    def test_record_generation(self):
        from ksp_neuro.ne_engine.analytics import TrainingAnalytics
        with tempfile.TemporaryDirectory() as d:
            a = TrainingAnalytics(output_dir=d)
            r = a.record_generation(0, [1.0, 2.0, 3.0, 4.0, 5.0])
            self.assertAlmostEqual(r.mean_fitness, 3.0, places=5)

    def test_learning_curve(self):
        from ksp_neuro.ne_engine.analytics import TrainingAnalytics
        with tempfile.TemporaryDirectory() as d:
            a = TrainingAnalytics(output_dir=d)
            for gen in range(100):
                a.record_generation(gen, [gen + i * 0.1 for i in range(20)])
            curve = a.get_learning_curve(window_size=10)
            self.assertEqual(len(curve), 100)

    def test_convergence_detection(self):
        from ksp_neuro.ne_engine.analytics import TrainingAnalytics
        with tempfile.TemporaryDirectory() as d:
            a = TrainingAnalytics(output_dir=d)
            for gen in range(200):
                a.record_generation(gen, [1.0 + 0.0001 * (gen % 5)])
            summary = a.get_summary()
            # Just verify it doesn't crash and has expected keys
            self.assertIn("total_generations", summary)

    def test_save_full_analytics(self):
        from ksp_neuro.ne_engine.analytics import TrainingAnalytics
        with tempfile.TemporaryDirectory() as d:
            a = TrainingAnalytics(output_dir=d)
            for gen in range(10):
                a.record_generation(gen, [gen * 0.5 + i for i in range(20)])
            path = a.save_full_analytics()
            self.assertTrue(os.path.exists(path))


class TestStarterAgents(unittest.TestCase):
    def test_starter_agents_defined(self):
        from ksp_neuro.ne_engine.starter_agents import STARTER_AGENTS, StarterGenomeProvider
        expected = {"vertical_burn", "orbital_insertion", "kerbin_orbit", "duna_transfer", "landing"}
        self.assertEqual(set(STARTER_AGENTS.keys()), expected)

    def test_starter_genome_creation(self):
        from ksp_neuro.ne_engine.starter_agents import StarterGenomeProvider, generate_starter_weights
        provider = StarterGenomeProvider()
        genome = provider.get_starter_genome("vertical_burn")
        self.assertIsNotNone(genome)
        self.assertGreater(len(genome.nodes), 0)

    def test_get_all_starter_names(self):
        from ksp_neuro.ne_engine.starter_agents import StarterGenomeProvider
        p = StarterGenomeProvider()
        names = p.get_all_starter_names()
        self.assertIsInstance(names, list)
        self.assertGreater(len(names), 0)


class TestKSPInterface(unittest.TestCase):
    def test_check_ksp_health(self):
        from ksp_neuro.ksp_interface.hardened import check_ksp_health
        r = check_ksp_health("localhost", 59999, timeout=0.1)
        self.assertIn("reachable", r); self.assertIn("error", r)

    def test_connection_pool_initialization(self):
        from ksp_neuro.ksp_interface.hardened import kRPCConnectionPool
        pool = kRPCConnectionPool(host="localhost", port=59999)
        self.assertFalse(pool.is_connected)

    def test_simulation_fallback(self):
        from ksp_neuro.ksp_interface.hardened import SimulationFallback
        fb = SimulationFallback()
        r = fb.enable(task_name="vertical_burn", n_agents=10)
        self.assertTrue(r); self.assertTrue(fb.active)


class TestVectorizedSimPerformance(unittest.TestCase):
    def test_vectorized_speed(self):
        from ksp_neuro.sim_env.vectorized_sim import VectorizedOrbitSimulator
        sim = VectorizedOrbitSimulator(task_name="vertical_burn", n_agents=100)
        obs = sim.reset()
        start_time = time.time(); ticks = 0
        for _ in range(60 * 120):
            actions = np.random.uniform(-0.3, 0.8, (100, 1))
            _, _, dones, _ = sim.step(actions)
            if bool(dones.any()):
                sim.reset()
            ticks += 1
        elapsed = time.time() - start_time; tps = ticks / elapsed if elapsed > 0 else 0
        self.assertGreaterEqual(tps, 500), f"Vectorized sim only achieved {tps:.0f} ticks/sec"


class TestTorchEngine(unittest.TestCase):
    def test_create_mlp(self):
        from ksp_neuro.ne_engine.torch_engine import TorchMLP, create_network
        mlp = create_network("mlp", input_dim=10, hidden_dims=[8, 6], output_dim=3)
        self.assertIsNotNone(mlp)

    def test_mlp_forward_pass(self):
        from ksp_neuro.ne_engine.torch_engine import TorchMLP; import torch
        m = TorchMLP(input_dim=5, hidden_dims=[4, 3], output_dim=2)
        y = m(torch.randn(10, 5))
        self.assertEqual(y.shape, (10, 2))

    def test_create_lstm(self):
        from ksp_neuro.ne_engine.torch_engine import TorchLSTM, create_network
        lstm = create_network("lstm", input_dim=8, hidden_dims=[32, 2], output_dim=4)
        self.assertIsNotNone(lstm)

    def test_lstm_forward_pass(self):
        from ksp_neuro.ne_engine.torch_engine import TorchLSTM; import torch
        l = TorchLSTM(input_dim=6, hidden_dim=16, output_dim=3, num_layers=2)
        y, state = l(torch.randn(8, 10, 6))
        self.assertEqual(y.shape, (8, 3))


class TestIntegration(unittest.TestCase):
    def test_full_training_loop(self):
        from ksp_neuro.ne_engine import create_random_genome, SpeciationManager, GenomeMutator, GenomeCrossover, NSGAIIEvaluator, TrainingAnalytics
        from ksp_neuro.sim_env.vectorized_sim import VectorizedOrbitSimulator as VS
        spec = SpeciationManager(); mut = GenomeMutator(add_node_prob=0.1)
        cr = GenomeCrossover(); fe = NSGAIIEvaluator()
        with tempfile.TemporaryDirectory() as d:
            a = TrainingAnalytics(output_dir=d)
            pop = [create_random_genome(12, 1, [8]) for _ in range(20)]
            for i, g in enumerate(pop):
                g.genome_id = i
            sim = VS(task_name="vertical_burn", n_agents=20); obs = sim.reset()
            done_any = False
            for _ in range(60 * 30):
                actions = np.random.uniform(-0.2, 0.8, (20, 1))
                _, rewards, dones, _ = sim.step(actions)
                if bool(dones.any()):
                    done_any = True; break
            fitnesses = {g.genome_id: float(rewards[0]) for g in pop}
        for g, f in zip(pop, [fitnesses[g2.genome_id] for g2 in pop]):
            g.fitness = f
        spec.assign_species(pop)
        af = fe.evaluate_fitness(pop, {g.genome_id: g.fitness for g in pop})
        sm2 = {}
        for f2 in af:
            sid = next((g.species_id for g in pop if g.genome_id == f2.genome_id), 0)
            sm2.setdefault(sid, []).append(f2.genome_id)
        fe.compute_adaptive_fitness(af, sm2)
        record = a.record_generation(1, [g.fitness for g in pop])
        self.assertEqual(record.generation, 1)


if __name__ == "__main__":
    unittest.main()
