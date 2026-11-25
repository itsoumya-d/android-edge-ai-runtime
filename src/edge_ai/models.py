from dataclasses import dataclass
from enum import Enum
from typing import Optional

class QuantizationTier(str, Enum):
    FP16 = "FP16"
    INT8 = "INT8"
    INT4 = "INT4"

@dataclass
class DeviceStatus:
    battery_pct: float
    temperature_c: float
    is_charging: bool
    available_ram_mb: int

@dataclass
class InferenceTask:
    task_id: str
    model_name: str
    prompt: str
    max_tokens: int = 128
    allow_cloud_fallback: bool = False

@dataclass
class InferenceResult:
    task_id: str
    output_text: str
    chosen_tier: QuantizationTier
    tokens_generated: int
    latency_ms: float
    energy_used_mwh: float
    privacy_verified: bool

class ThermalEmergencyException(Exception):
    def __init__(self, message: str, current_temp: float):
        super().__init__(message)
        self.message = message
        self.current_temp = current_temp

class PrivacyViolationException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
