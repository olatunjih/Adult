import unittest
import os
from src.data_pipeline.pipeline import DataPipeline
from src.config import settings

# Helper to check if Pillow is installed
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.pipeline = DataPipeline()
        self.test_data_dir = "test_data"
        self.text_file_path = os.path.join(self.test_data_dir, "text", "sample.txt")
        self.image_file_path = os.path.join(self.test_data_dir, "images", "dummy_cat.jpg")

        # Create dummy directories
        os.makedirs(os.path.dirname(self.text_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.image_file_path), exist_ok=True)

        # Create a dummy text file
        with open(self.text_file_path, "w") as f:
            f.write("This is a test.")

        # Create a dummy image file if Pillow is available
        if PIL_AVAILABLE:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(self.image_file_path)

    def test_load_data(self):
        """Test loading of text and image data."""
        sources = [
            {"modality": "text", "path": self.text_file_path},
            {"modality": "image", "path": self.image_file_path if PIL_AVAILABLE else "nonexistent.jpg"},
        ]

        # Filter out image source if Pillow is not available
        if not PIL_AVAILABLE:
            sources = [s for s in sources if s['modality'] == 'text']

        data = self.pipeline.load_data(sources)

        if PIL_AVAILABLE:
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["modality"], "text")
            self.assertEqual(data[1]["modality"], "image")
        else:
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["modality"], "text")

    @unittest.skipIf(not PIL_AVAILABLE, "Pillow not installed, skipping image-related tests")
    def test_preprocess_data(self):
        """Test preprocessing of data."""
        sources = [
            {"modality": "text", "path": self.text_file_path},
            {"modality": "image", "path": self.image_file_path},
        ]
        loaded_data = self.pipeline.load_data(sources)
        processed_data = self.pipeline.preprocess(loaded_data)
        self.assertEqual(len(processed_data), 2)

        # Check text tokenization
        self.assertEqual(processed_data[0]["modality"], "text")
        self.assertEqual(processed_data[0]["content"], ["This", "is", "a", "test."])

        # Check image resizing
        self.assertEqual(processed_data[1]["modality"], "image")
        self.assertEqual(processed_data[1]["content"].size, settings.IMAGE_TARGET_SIZE)

    def test_batch_data(self):
        """Test batching of data."""
        data = [{"id": i} for i in range(10)]
        batched_data = self.pipeline.batch_data(data, batch_size=3)
        self.assertEqual(len(batched_data), 4)
        self.assertEqual(len(batched_data[0]), 3)
        self.assertEqual(len(batched_data[3]), 1)

    @unittest.skipIf(not PIL_AVAILABLE, "Pillow not installed, skipping image-related tests")
    def test_run_pipeline(self):
        """Test the full data pipeline run."""
        sources = [
            {"modality": "text", "path": self.text_file_path},
            {"modality": "image", "path": self.image_file_path},
        ]
        batched_data = self.pipeline.run(sources, batch_size=1)
        self.assertEqual(len(batched_data), 2)
        self.assertEqual(len(batched_data[0]), 1)
        self.assertEqual(batched_data[0][0]["modality"], "text")
        self.assertEqual(batched_data[1][0]["modality"], "image")

    def tearDown(self):
        """Clean up test environment."""
        # Clean up dummy files
        if os.path.exists(self.text_file_path):
            os.remove(self.text_file_path)
        if os.path.exists(self.image_file_path):
            os.remove(self.image_file_path)
        if os.path.exists(self.test_data_dir):
            # Clean up created directories if they are empty
            try:
                os.rmdir(os.path.join(self.test_data_dir, "text"))
                os.rmdir(os.path.join(self.test_data_dir, "images"))
                os.rmdir(self.test_data_dir)
            except OSError:
                pass


if __name__ == '__main__':
    unittest.main()
