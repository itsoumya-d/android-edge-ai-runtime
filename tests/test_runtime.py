import pytest
from src.edge_ai.models import (
    DeviceStatus,
    InferenceTask,
    QuantizationTier,
    ThermalEmergencyException,
    PrivacyViolationException
)
from src.edge_ai.governor import ThermalGovernor
from src.edge_ai.privacy import PrivacyAuditor
from src.edge_ai.engine import EdgeInferenceEngine

def test_governor_fp16_policy():
    status = DeviceStatus(battery_pct=85.0, temperature_c=31.0, is_charging=False, available_ram_mb=3072)
    tier, speed = ThermalGovernor.evaluate_policy(status)
    assert tier == QuantizationTier.FP16
    assert speed == 1.0

def test_governor_int8_policy():
    status = DeviceStatus(battery_pct=40.0, temperature_c=35.0, is_charging=False, available_ram_mb=3072)
    tier, speed = ThermalGovernor.evaluate_policy(status)
    assert tier == QuantizationTier.INT8
    assert speed == 1.0

def test_governor_int4_thermal_throttling():
    status = DeviceStatus(battery_pct=75.0, temperature_c=42.5, is_charging=False, available_ram_mb=3072)
    tier, speed = ThermalGovernor.evaluate_policy(status)
    assert tier == QuantizationTier.INT4
    assert speed == 0.5

def test_governor_emergency_cutoff():
    status = DeviceStatus(battery_pct=90.0, temperature_c=49.0, is_charging=True, available_ram_mb=4096)
    with pytest.raises(ThermalEmergencyException) as excinfo:
        ThermalGovernor.evaluate_policy(status)
    assert "Thermal limit exceeded" in str(excinfo.value)
    assert excinfo.value.current_temp == 49.0

def test_privacy_identifier_sanitization():
    prompt = "Diagnostic log for imei 860123456789012 and mac a1:b2:c3:d4:e5:f6 user query"
    clean = PrivacyAuditor.sanitize_context(prompt)
    assert "860123456789012" not in clean
    assert "a1:b2:c3:d4:e5:f6" not in clean
    assert "[DEVICE_ID_REDACTED]" in clean

def test_privacy_leakage_assertion():
    assert PrivacyAuditor.verify_no_outbound_leakage(0, allow_cloud=False) is True
    assert PrivacyAuditor.verify_no_outbound_leakage(1, allow_cloud=True) is True
    with pytest.raises(PrivacyViolationException):
        PrivacyAuditor.verify_no_outbound_leakage(2, allow_cloud=False)

def test_edge_inference_engine_execution():
    engine = EdgeInferenceEngine()
    task = InferenceTask(
        task_id="task-fairphone-001",
        model_name="tiny-llama-onnx",
        prompt="Explain on-device sustainable battery life and privacy features.",
        max_tokens=64
    )
    status = DeviceStatus(battery_pct=90.0, temperature_c=32.0, is_charging=True, available_ram_mb=4096)
    res = engine.run_inference(task, status)

    assert res.task_id == "task-fairphone-001"
    assert res.chosen_tier == QuantizationTier.FP16
    assert res.tokens_generated > 0
    assert res.latency_ms > 0
    assert res.energy_used_mwh > 0
    assert res.privacy_verified is True
    assert "[OnDevice:FP16]" in res.output_text
