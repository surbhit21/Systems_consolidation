import os
import unittest

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
            self.fail("ANNarchy package is not installed or failed to import.")

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

if __name__ == "__main__":
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestANNarchySetup)
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n All OK\n")
    else:
        print("\n Some tests failed.\n")