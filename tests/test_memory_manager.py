import unittest
import time
from src.memory.manager import MemoryManager
from src.data_structures.core import MemoryItem

class TestMemoryManager(unittest.TestCase):

    def setUp(self):
        self.memory_manager = MemoryManager()

    def test_store_and_retrieve_memory(self):
        """
        Tests basic storage and retrieval of memory items.
        """
        item1 = MemoryItem(type="experience", content="Good experience", timestamp=time.time(), importance_score=0.8, storage_tier="hot")
        item2 = MemoryItem(type="experience", content="Bad experience", timestamp=time.time(), importance_score=0.3, storage_tier="warm")
        item3 = MemoryItem(type="skill", content="New skill learned", timestamp=time.time(), importance_score=0.9, storage_tier="hot")

        self.memory_manager.store_memory(item1)
        self.memory_manager.store_memory(item2)
        self.memory_manager.store_memory(item3)

        self.assertEqual(len(self.memory_manager.get_all_memory()), 3)

        retrieved_experiences = self.memory_manager.retrieve_memory(item_type="experience")
        self.assertEqual(len(retrieved_experiences), 2)
        # Check that they are sorted by importance
        self.assertEqual(retrieved_experiences[0].content, "Good experience")

        retrieved_skills = self.memory_manager.retrieve_memory(item_type="skill")
        self.assertEqual(len(retrieved_skills), 1)
        self.assertEqual(retrieved_skills[0].content, "New skill learned")

    def test_retrieve_with_limit(self):
        """
        Tests that the retrieval limit is respected.
        """
        for i in range(20):
            item = MemoryItem(type="log", content=f"Log entry {i}", timestamp=time.time(), importance_score=i/20.0, storage_tier="cold")
            self.memory_manager.store_memory(item)

        retrieved_logs = self.memory_manager.retrieve_memory(item_type="log", limit=5)
        self.assertEqual(len(retrieved_logs), 5)
        # Check that the most important items are returned
        self.assertEqual(retrieved_logs[0].content, "Log entry 19")


    def test_retrieve_non_existent_type(self):
        """
        Tests that retrieving a non-existent type returns an empty list.
        """
        retrieved_items = self.memory_manager.retrieve_memory(item_type="non_existent")
        self.assertEqual(len(retrieved_items), 0)

    def tearDown(self):
        """
        Clears the memory after each test.
        """
        self.memory_manager.clear_memory()

if __name__ == '__main__':
    unittest.main()
