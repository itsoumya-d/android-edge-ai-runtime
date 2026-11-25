import re
from .models import PrivacyViolationException

DEVICE_IDENTIFIER_PATTERNS = [
    r"\b(?:[0-9]{15}|[0-9]{17})\b", # IMEI pattern
    r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b", # MAC address
    r"\bandroid_id_[a-f0-9]{16}\b", # Android ID
]

class PrivacyAuditor:
    """
    Enforces zero-data-leakage guarantee for on-device local execution.
    """
    @classmethod
    def sanitize_context(cls, text: str) -> str:
        sanitized = text
        for pattern in DEVICE_IDENTIFIER_PATTERNS:
            sanitized = re.sub(pattern, "[DEVICE_ID_REDACTED]", sanitized, flags=re.IGNORECASE)
        return sanitized

    @classmethod
    def verify_no_outbound_leakage(cls, network_calls_attempted: int, allow_cloud: bool) -> bool:
        if network_calls_attempted > 0 and not allow_cloud:
            raise PrivacyViolationException("Outbound network activity detected during strict on-device inference")
        return True
