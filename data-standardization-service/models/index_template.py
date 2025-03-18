from pydantic_settings import BaseSettings
from typing import Dict, Any

class IndexSettings(BaseSettings):
    number_of_shards: str
    number_of_replicas: str
    routing_allocation_include_tier_preference: str

class IndexMappings(BaseSettings):
    dynamic: bool
    properties: Dict[str, Any]

class Template(BaseSettings):
    settings: IndexSettings
    mappings: IndexMappings
