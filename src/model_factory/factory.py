import torch
import torch.nn as nn
from typing import Dict, Any, List
from src.data_structures.core import ModelSpec, Task
from src.config import settings

class SimpleTextModel(nn.Module):
    """
    A very simple model for demonstration purposes.
    This model is intended for text classification.
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(SimpleTextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # We take the output of the last time step
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out

class ModelFactory:
    """
    A factory for creating, training, and managing models.
    """
    def __init__(self):
        self.model_architecture = settings.MODEL_ARCHITECTURE
        self.num_layers = settings.NUM_LAYERS
        self.hidden_size = settings.HIDDEN_SIZE

    def design_model_architecture(self, task: Task) -> ModelSpec:
        """
        Designs a model architecture based on the task.
        This is a placeholder and returns a predefined ModelSpec.
        """
        # In a real system, this would be a complex process involving analysis of the task.
        # Here, we just return a spec based on the global config.
        return ModelSpec(
            model_id="simple_text_model_v1",
            architecture_type=self.model_architecture,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            multimodal_types=["text"],
            task_domain="text_classification",
            version="1.0.0"
        )

    def build_model(self, spec: ModelSpec) -> nn.Module:
        """
        Builds a model from a ModelSpec.
        """
        # This is a simplified implementation that only supports a "transformer" architecture
        # which we map to our SimpleTextModel for this example.
        if spec.architecture_type == "transformer":
            # These would typically come from a vocabulary analysis or be part of the spec
            vocab_size = 10000
            embed_size = 128
            num_classes = 2 # e.g., for binary sentiment analysis
            return SimpleTextModel(vocab_size, embed_size, spec.hidden_size, num_classes)
        else:
            raise ValueError(f"Unsupported model architecture: {spec.architecture_type}")

    def train_model(self, model: nn.Module, dataloader, epochs=3):
        """
        A simplified training loop.
        Assumes dataloader provides batches of (inputs, labels).
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        print("Starting model training...")
        for epoch in range(epochs):
            for i, (inputs, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if (i+1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        print("Training finished.")

    def save_model(self, model: nn.Module, path: str):
        """
        Saves the model's state dictionary to a file.
        """
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, spec: ModelSpec, path: str) -> nn.Module:
        """
        Loads a model's state dictionary from a file.
        """
        model = self.build_model(spec)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
