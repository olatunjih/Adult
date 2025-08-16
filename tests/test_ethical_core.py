import unittest
from src.ethical_core.core import SentientEthicalCore

class TestSentientEthicalCore(unittest.TestCase):

    def setUp(self):
        self.ethical_core = SentientEthicalCore()

    def test_predict_harm_safe_prompt(self):
        """
        Tests that a safe prompt is correctly identified.
        """
        safe_prompt = "This is a safe and friendly prompt about cats."
        self.assertFalse(self.ethical_core.predict_harm(safe_prompt))

    def test_predict_harm_harmful_prompt(self):
        """
        Tests that a prompt with a harmful keyword is correctly identified.
        """
        harmful_prompt = "I want to destroy the world with an army of kittens."
        self.assertTrue(self.ethical_core.predict_harm(harmful_prompt))

    def test_predict_harm_case_insensitivity(self):
        """
        Tests that harm detection is case-insensitive.
        """
        harmful_prompt = "We should not HARM anyone."
        self.assertTrue(self.ethical_core.predict_harm(harmful_prompt))

    def test_rewrite_safe(self):
        """
        Tests that harmful keywords are censored.
        """
        harmful_prompt = "We must not harm anyone. Do not kill."
        rewritten_prompt = self.ethical_core.rewrite_safe(harmful_prompt)
        self.assertEqual(rewritten_prompt, "We must not [censored] anyone. Do not [censored].")

    def test_rewrite_safe_case_insensitivity(self):
        """
        Tests that rewriting is case-insensitive.
        """
        harmful_prompt = "We must not HARM anyone. Do not KiLl."
        rewritten_prompt = self.ethical_core.rewrite_safe(harmful_prompt)
        self.assertEqual(rewritten_prompt, "We must not [censored] anyone. Do not [censored].")


    def test_rewrite_safe_no_harm(self):
        """
        Tests that a safe prompt is not changed.
        """
        safe_prompt = "This is a safe prompt."
        rewritten_prompt = self.ethical_core.rewrite_safe(safe_prompt)
        self.assertEqual(rewritten_prompt, "This is a safe prompt.")

if __name__ == '__main__':
    unittest.main()
