import os
import sys
import unittest
import numpy as np


APP_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(APP_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

class TestANNarchySetup(unittest.TestCase):
    def setUp(self):
        # List of required files to check
        self.required_files = [
            "app/Get_Drifty.py",
            "app/Utilities.py",
            "app/plotting_widget.py"
        ]

    def test_required_files_exist(self):
        for file in self.required_files:
            with self.subTest(file=file):
                self.assertTrue(os.path.exists(file), f"Missing required file: {file}")

    def test_annarchy_import_and_compile(self):
        try:
            import ANNarchy as ann
        except ImportError:
            self.skipTest("ANNarchy package is not installed.")

        # Try compiling a minimal ANNarchy model
        try:
            import ANNarchy as ann
            from ANNarchy import Neuron, Population, setup, compile
            setup()
            NeuronModel = Neuron(
                parameters=dict(
                    Iext = 1.0
                ), 
                equations="dr/dt +r = pos(sum(exc) + Iext)")
            net = ann.Network()
            pop = net.create(name="pop", geometry=1, neuron=NeuronModel)
            net.compile()
        except Exception as e:
            self.fail(f"ANNarchy compilation failed: {e}")


class TestCurrentSimulationCode(unittest.TestCase):
    def test_input_intervals_follow_recorded_external_drive(self):
        from plotting_widget import active_input_intervals

        input_history = np.zeros((12, 3))
        input_history[2:5, :] = 1.0
        input_history[8:10, 1] = 0.5

        self.assertEqual(active_input_intervals(input_history), [(2, 5), (8, 10)])

    def test_simulation_modules_are_import_safe(self):
        import GetDrift_FF
        import HPC_CTX_drift
        import TwoRegion_np
        import ThreeRegion

        self.assertTrue(callable(GetDrift_FF.main))
        self.assertTrue(callable(HPC_CTX_drift.main))
        self.assertTrue(callable(TwoRegion_np.main))
        self.assertTrue(callable(ThreeRegion.main))

    def test_getdrift_ff_excitability_turnover_recruits_current_cohort(self):
        from GetDrift_FF import twolayer_FF, relu

        np.random.seed(123)
        n_days = 3
        neurons_per_day = 4
        n_neurons = n_days * neurons_per_day
        model = twolayer_FF(
            n_inp=n_neurons,
            n_neurons=n_neurons,
            n_cont=2,
            baseline_e=np.zeros(n_neurons),
            tau=10.0,
            dt=1.0,
            act=relu,
            lr=0.0,
            decay_r=0.0,
            I0=0.0,
            I1=0.0,
            I2=0.0,
        )
        model.act_threshold[:] = 0.0
        model.cont_exc[:] = 0.0
        model.act_threshold_cont[:] = 0.0

        zero_input = np.zeros(n_neurons)
        zero_context = np.zeros(2)
        base_e = np.zeros(n_neurons)

        for day in range(n_days):
            start = day * neurons_per_day
            stop = start + neurons_per_day
            model.excitability = base_e.copy()
            model.excitability[start:stop] += 5.0

            for _ in range(60):
                rates = model.step(zero_input, zero_context)

            main_rates = rates[model.n_cont:]
            cohort = main_rates[start:stop]
            outside = np.concatenate([main_rates[:start], main_rates[stop:]])

            with self.subTest(day=day):
                self.assertGreater(cohort.mean(), 4.0)
                self.assertGreater(cohort.mean(), outside.mean() + 1.0)

    def test_pca_cosine_similarity_to_day0_shape_and_identity(self):
        from GetDrift_FF import pca_transform_activity, cosine_similarity_to_day0

        activity = np.array([
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
        ])
        scores, components, mean_activity, explained = pca_transform_activity(activity, n_components=2)
        cosine = cosine_similarity_to_day0(scores)

        self.assertEqual(scores.shape, (2, 3, 2))
        self.assertEqual(components.shape, (2, 3))
        self.assertEqual(mean_activity.shape, (3,))
        self.assertEqual(explained.shape, (2,))
        np.testing.assert_allclose(cosine[:, 0], np.ones(2), atol=1e-12)
        np.testing.assert_allclose(cosine[:, 1], np.ones(2), atol=1e-12)


if __name__ == "__main__":
    import unittest
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestANNarchySetup))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestCurrentSimulationCode))
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n All OK\n")
    else:
        print("\n Some tests failed.\n")
