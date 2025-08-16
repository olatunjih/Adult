import unittest
import os
import shutil
from src.model_factory.factory import ModelFactory
from src.inference.engine import InferenceEngine

class TestInferenceEngine(unittest.TestCase):

    def setUp(self):
        self.model_factory = ModelFactory()
        self.model_dir = "test_inference_models"
        # Ensure the directory is clean before each test
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
        self.inference_engine = InferenceEngine(self.model_factory, model_dir=self.model_dir)

    def test_generate_response_with_no_model(self):
        """
        Tests that a new model is trained and saved if one doesn't exist.
        """
        prompt = "This is a test prompt."
        # The first call should train and save a model.
        response = self.inference_engine.generate_response(prompt)

        self.assertTrue(os.path.exists(self.inference_engine.model_path))
        self.assertIn("Predicted class:", response)

    def test_generate_response_with_existing_model(self):
        """
        Tests that an existing model is loaded correctly.
        """
        # First, generate a response to ensure a model is created and saved.
        self.inference_engine.generate_response("Initial prompt to create model.")

        # Now, create a new inference engine instance that should load the model.
        # This simulates restarting the application.
        new_engine = InferenceEngine(self.model_factory, model_dir=self.model_dir)
        prompt = "This is a second test prompt."
        response = new_engine.generate_response(prompt)

        # Check that the response is in the expected format.
        self.assertIn("Predicted class:", response)
        # We can also check if the loading message was printed (requires capturing stdout)
        # For simplicity, we just check the functionality.

    def tearDown(self):
        # Clean up the model directory after tests.
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)

if __name__ == '__main__':
    unittest.main()
