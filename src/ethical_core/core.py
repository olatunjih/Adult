import re
from src.data_structures.core import Task

class SentientEthicalCore:
    """
    A simplified ethical core for harm prevention.
    This is a placeholder for the advanced concepts in the blueprint.
    """
    def __init__(self):
        """
        Initializes the ethical core with a simple list of harmful keywords.
        """
        self.harmful_keywords = ["harm", "kill", "destroy", "hate", "attack"]

    def predict_harm(self, task_input: str) -> bool:
        """
        Predicts if a task input is potentially harmful.
        This is a very basic keyword-based check.

        Args:
            task_input: The input string to check.

        Returns:
            True if harmful content is detected, False otherwise.
        """
        if not isinstance(task_input, str):
            return False

        prompt = task_input.lower()
        for keyword in self.harmful_keywords:
            if keyword in prompt:
                print(f"Ethical Core: Detected harmful keyword '{keyword}'.")
                return True
        return False

    def rewrite_safe(self, task_input: str) -> str:
        """
        Rewrites a task input to be safer by censoring harmful keywords.

        Args:
            task_input: The input string to rewrite.

        Returns:
            A rewritten, safer version of the input string.
        """
        if not isinstance(task_input, str):
            return task_input

        safe_prompt = task_input
        for keyword in self.harmful_keywords:
            # Using case-insensitive replacement
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            safe_prompt = pattern.sub("[censored]", safe_prompt)

        if safe_prompt != task_input:
            print("Ethical Core: Rewrote prompt to be safe.")
        return safe_prompt
