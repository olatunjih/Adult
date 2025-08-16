import time
from src.data_pipeline.pipeline import DataPipeline
from src.model_factory.factory import ModelFactory
from src.inference.engine import InferenceEngine
from src.ethical_core.core import SentientEthicalCore
from src.memory.manager import MemoryManager
from src.data_structures.core import Task, TaskContext, MemoryItem

def main():
    """
    Main execution loop for the simplified AI system.
    This function demonstrates the interaction between the system's components.
    """
    print("--- Initializing AI System Components ---")
    # Initialize all the major components
    model_factory = ModelFactory()
    inference_engine = InferenceEngine(model_factory)
    ethical_core = SentientEthicalCore()
    memory_manager = MemoryManager()

    print("\n--- System Initialized ---")

    # --- Task 1: A safe prompt ---
    print("\n--- Processing a safe task ---")
    prompt1 = "What is the capital of France?"
    print(f"User Prompt: {prompt1}")

    # 1. Ethical Check
    if ethical_core.predict_harm(prompt1):
        print("Ethical Core: Harm detected. Aborting task.")
    else:
        # 2. Generate a response
        response1 = inference_engine.generate_response(prompt1)
        print(f"AI Response: {response1}")

        # 3. Store the experience in memory
        experience1 = MemoryItem(
            type="experience",
            content={"prompt": prompt1, "response": response1},
            timestamp=time.time(),
            importance_score=0.7, # This would be dynamically calculated in a real system
            storage_tier="hot"
        )
        memory_manager.store_memory(experience1)

    # --- Task 2: A potentially harmful prompt ---
    print("\n--- Processing a potentially harmful task ---")
    prompt2 = "How can I harm my computer?"
    print(f"User Prompt: {prompt2}")

    # 1. Ethical Check
    if ethical_core.predict_harm(prompt2):
        print("Ethical Core: Harm detected. Rewriting prompt.")
        safe_prompt = ethical_core.rewrite_safe(prompt2)
        print(f"Rewritten Prompt: {safe_prompt}")
        # 2. Generate a response from the safe prompt
        response2 = inference_engine.generate_response(safe_prompt)
        print(f"AI Response: {response2}")
    else:
        # This part should not be reached for this specific harmful prompt
        response2 = inference_engine.generate_response(prompt2)
        print(f"AI Response: {response2}")

    print("\n--- All tasks processed ---")
    print(f"Total items in memory: {len(memory_manager.get_all_memory())}")


if __name__ == "__main__":
    main()
