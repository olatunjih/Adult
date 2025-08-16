from typing import List, Dict, Any
import os

try:
    from PIL import Image
except ImportError:
    print("Warning: Pillow library not found. Image processing will not be available.")
    Image = None

from src.config import settings

class DataPipeline:
    """
    A simplified data pipeline for processing text and image data.
    """
    def __init__(self):
        """
        Initializes the data pipeline with settings from the config.
        """
        self.image_target_size = settings.IMAGE_TARGET_SIZE

    def load_data(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Loads data from a list of sources.
        Each source is a dictionary with 'modality' and 'path'.
        """
        data = []
        for source in sources:
            modality = source.get("modality")
            path = source.get("path")

            if not modality or not path:
                print(f"WARNING: Skipping source due to missing 'modality' or 'path': {source}")
                continue

            if not os.path.exists(path):
                print(f"WARNING: Skipping source because path does not exist: {path}")
                continue

            if modality == "text":
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                data.append({"modality": "text", "content": content, "path": path})
            elif modality == "image":
                if Image is None:
                    print(f"WARNING: Skipping image source because Pillow is not installed: {path}")
                    continue
                try:
                    with Image.open(path) as img:
                        image = img.copy()
                    data.append({"modality": "image", "content": image, "path": path})
                except Exception as e:
                    print(f"WARNING: Failed to load image from {path}: {e}")
            else:
                print(f"WARNING: Skipping source with unsupported modality: {modality}")
        return data

    def preprocess(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocesses the loaded data.
        """
        processed_data = []
        for item in data:
            modality = item.get("modality")
            content = item.get("content")

            if modality == "text":
                # Simple tokenization by splitting on whitespace
                tokens = content.split()
                processed_data.append({"modality": "text", "content": tokens, "path": item["path"]})
            elif modality == "image":
                if Image is None:
                    continue
                # Simple preprocessing: resize and convert to RGB
                try:
                    image = content.resize(self.image_target_size).convert("RGB")
                    processed_data.append({"modality": "image", "content": image, "path": item["path"]})
                finally:
                    content.close()
        return processed_data

    def batch_data(self, data: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
        """
        Batches the preprocessed data.
        """
        if not data:
            return []
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def run(self, sources: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
        """
        Runs the full data pipeline.
        """
        loaded_data = self.load_data(sources)
        preprocessed_data = self.preprocess(loaded_data)
        batched_data = self.batch_data(preprocessed_data, batch_size)
        return batched_data
