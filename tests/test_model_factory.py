import unittest
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.model_factory.factory import ModelFactory, SimpleTextModel
from src.data_structures.core import Task, TaskContext, ModelSpec

class TestModelFactory(unittest.TestCase):

    def setUp(self):
        self.factory = ModelFactory()
        self.model_dir = "test_models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "test_model.pth")

    def test_design_model_architecture(self):
        task_context = TaskContext(task_type="text_classification", user_role="user", multimodal_type="text")
        task = Task(id="test_task", input="some text", context=task_context)
        spec = self.factory.design_model_architecture(task)
        self.assertIsInstance(spec, ModelSpec)
        self.assertEqual(spec.task_domain, "text_classification")

    def test_build_model(self):
        spec = ModelSpec(
            model_id="test_model",
            architecture_type="transformer",
            num_layers=2,
            hidden_size=128,
            multimodal_types=["text"],
            task_domain="text_classification",
            version="1.0.0"
        )
        model = self.factory.build_model(spec)
        self.assertIsInstance(model, SimpleTextModel)

    def test_save_and_load_model(self):
        spec = ModelSpec(
            model_id="test_model",
            architecture_type="transformer",
            num_layers=2,
            hidden_size=128,
            multimodal_types=["text"],
            task_domain="text_classification",
            version="1.0.0"
        )
        model = self.factory.build_model(spec)
        self.factory.save_model(model, self.model_path)
        self.assertTrue(os.path.exists(self.model_path))

        loaded_model = self.factory.load_model(spec, self.model_path)
        self.assertIsInstance(loaded_model, SimpleTextModel)
        # A simple check to ensure the models have the same structure
        self.assertEqual(str(model.state_dict().keys()), str(loaded_model.state_dict().keys()))


    def test_train_model(self):
        spec = ModelSpec(
            model_id="test_model",
            architecture_type="transformer",
            num_layers=2,
            hidden_size=128,
            multimodal_types=["text"],
            task_domain="text_classification",
            version="1.0.0"
        )
        model = self.factory.build_model(spec)

        # Create some dummy data
        # Vocab size is 10000 in the model, so our inputs should be in that range
        inputs = torch.randint(0, 10000, (100, 10)) # 100 samples, 10 tokens each
        labels = torch.randint(0, 2, (100,)) # 100 labels, 2 classes
        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(dataset, batch_size=10)

        # The train_model method prints to stdout, we can capture that if needed
        # For now, just run it and make sure it doesn't crash.
        try:
            self.factory.train_model(model, dataloader, epochs=1)
        except Exception as e:
            self.fail(f"train_model failed with exception: {e}")


    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.model_dir):
            os.rmdir(self.model_dir)

if __name__ == '__main__':
    unittest.main()
