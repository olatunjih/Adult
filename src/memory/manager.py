from typing import List
from src.data_structures.core import MemoryItem

class MemoryManager:
    """
    A simple in-memory manager for storing and retrieving MemoryItems.
    This serves as a placeholder for the more complex Fractal Memory system.
    """
    def __init__(self):
        """
        Initializes the MemoryManager with an empty list to store memories.
        """
        self._memory: List[MemoryItem] = []

    def store_memory(self, item: MemoryItem):
        """
        Stores a MemoryItem in the in-memory list.

        Args:
            item: The MemoryItem to store.
        """
        self._memory.append(item)
        print(f"MemoryManager: Stored item of type '{item.type}' with importance {item.importance_score}.")

    def retrieve_memory(self, item_type: str, limit: int = 10) -> List[MemoryItem]:
        """
        Retrieves MemoryItems of a specific type from memory, sorted by importance.

        Args:
            item_type: The type of MemoryItem to retrieve.
            limit: The maximum number of items to retrieve.

        Returns:
            A list of matching MemoryItems.
        """
        # Filter by type and sort by importance score in descending order
        retrieved_items = sorted(
            [item for item in self._memory if item.type == item_type],
            key=lambda x: x.importance_score,
            reverse=True
        )
        print(f"MemoryManager: Retrieved {len(retrieved_items)} items of type '{item_type}'.")
        return retrieved_items[:limit]

    def get_all_memory(self) -> List[MemoryItem]:
        """
        Returns all items currently in memory. Useful for debugging.
        """
        return self._memory

    def clear_memory(self):
        """
        Clears all items from memory. Useful for testing.
        """
        self._memory = []
