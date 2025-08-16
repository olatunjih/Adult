from PIL import Image
import os

def generate_dummy_image(path="data/images/dummy_cat.jpg"):
    """
    Generates a dummy image for testing.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        img = Image.new('RGB', (100, 100), color = 'red')
        img.save(path)
        print(f"Dummy image saved to {path}")
    except NameError:
        print("Pillow is not installed. Cannot generate dummy image.")
    except Exception as e:
        print(f"Failed to generate dummy image: {e}")

if __name__ == "__main__":
    generate_dummy_image()
