from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class TaskContext:
    """
    Provides context for a given task.
    """
    task_type: str
    user_role: str
    multimodal_type: str
    hardware_config: Optional[Dict[str, Any]] = None

@dataclass
class Task:
    """
    Represents a task to be processed by the AI system.
    """
    id: str
    input: Any  # Represents a Tensor in the blueprint
    context: TaskContext
    output: Optional[Any] = None  # Represents a Tensor in the blueprint
    plan: Optional[Any] = None  # Represents a Plan in the blueprint
    fused_features: Optional[Any] = None  # Represents a Tensor in the blueprint

@dataclass
class ModelSpec:
    """
    Specification for a model to be created by the ModelFactory.
    """
    model_id: str
    architecture_type: str
    num_layers: int
    hidden_size: int
    multimodal_types: List[str]
    task_domain: str
    version: str
    performance_metrics: Optional[Dict[str, float]] = None

@dataclass
class MemoryItem:
    """
    Represents an item stored in the AI's memory.
    """
    type: str
    content: Any  # Represents a Tensor in the blueprint
    timestamp: float
    importance_score: float
    storage_tier: str
