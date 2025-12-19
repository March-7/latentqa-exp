from typing import List, Optional
from dataclasses import dataclass


@dataclass
class swanlab_config:
    project: str = "latentqa"
    workspace: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    mode: Optional[str] = None
    dir: Optional[str] = None