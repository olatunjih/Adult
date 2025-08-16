import torch
from src.model_factory.factory import ModelFactory
from src.data_structures.core import ModelSpec, Task, TaskContext
import os

class InferenceEngine:
    """
    An engine for generating responses using a trained model.
    """
    def __init__(self, model_factory: ModelFactory, model_dir="trained_models"):
        """
        Initializes the inference engine.

        Args:
            model_factory: An instance of ModelFactory.
            model_dir: The directory to save/load models from.
        """
        self.model_factory = model_factory
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "inference_model.pth")
        self.model = None
        self.spec = None

    def _ensure_model_is_loaded(self):
        """
        Ensures that a model is loaded and ready for inference.
        If no model is loaded, it trains a new one.
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                print(f"Loading model from {self.model_path}")
                # We need a spec to build the model before loading state_dict
                self.spec = self.model_factory.design_model_architecture(
                    Task(id="dummy_task", input=None, context=TaskContext(task_type="text_classification", user_role="system", multimodal_type="text"))
                )
                self.model = self.model_factory.load_model(self.spec, self.model_path)
            else:
                print("No trained model found. Training a new one...")
                self.spec = self.model_factory.design_model_architecture(
                    Task(id="training_task", input=None, context=TaskContext(task_type="text_classification", user_role="system", multimodal_type="text"))
                )
                self.model = self.model_factory.build_model(self.spec)

                # Create dummy data for training
                from torch.utils.data import DataLoader, TensorDataset
                inputs = torch.randint(0, 10000, (200, 10)) # More data for better training
                labels = torch.randint(0, 2, (200,))
                dataloader = DataLoader(TensorDataset(inputs, labels), batch_size=10)

                self.model_factory.train_model(self.model, dataloader, epochs=3)
                self.model_factory.save_model(self.model, self.model_path)

            self.model.eval()

    def preprocess_prompt(self, prompt: str) -> torch.Tensor:
        """
        Preprocesses a raw text prompt into a tensor.
        This is a simplified version. A real implementation would use a fitted tokenizer.
        """
        # Placeholder for tokenization. This should align with the model's vocabulary.
        # For this example, we create a dummy tensor of indices.
        # Assuming a vocab size of 10000 and sequence length of 10.
        tokens = [ord(c) % 10000 for c in prompt]
        if len(tokens) > 10:
            tokens = tokens[:10]
        else:
            tokens = tokens + [0] * (10 - len(tokens))  # Padding

        return torch.tensor([tokens], dtype=torch.long)

    def generate_response(self, prompt: str) -> str:
        """
        Generates a response to a given prompt.
        """
        self._ensure_model_is_loaded()

        # Preprocess the prompt
        input_tensor = self.preprocess_prompt(prompt)

        # Generate a response
        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted_class = torch.max(output, 1)

            # The response is the predicted class index.
            response = f"Predicted class: {predicted_class.item()}"
            return response
