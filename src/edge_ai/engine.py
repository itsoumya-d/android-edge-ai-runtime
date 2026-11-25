import time
from .models import DeviceStatus, InferenceTask, InferenceResult, QuantizationTier
from .governor import ThermalGovernor
from .privacy import PrivacyAuditor

TIER_PERFORMANCE = {
    QuantizationTier.FP16: {"ms_per_token": 8.0, "mwh_per_token": 0.050, "ram_multiplier": 1.0},
    QuantizationTier.INT8: {"ms_per_token": 4.5, "mwh_per_token": 0.025, "ram_multiplier": 0.5},
    QuantizationTier.INT4: {"ms_per_token": 2.5, "mwh_per_token": 0.012, "ram_multiplier": 0.25},
}

class EdgeInferenceEngine:
    """
    Local on-device inference orchestrator managing thermal budgets and privacy.
    """
    def __init__(self):
        self.privacy_auditor = PrivacyAuditor()

    def run_inference(self, task: InferenceTask, status: DeviceStatus) -> InferenceResult:
        # 1. Thermal & battery evaluation
        tier, speed_factor = ThermalGovernor.evaluate_policy(status)

        # 2. Local privacy scrub
        clean_prompt = self.privacy_auditor.sanitize_context(task.prompt)

        # 3. Privacy leak assertion
        self.privacy_auditor.verify_no_outbound_leakage(network_calls_attempted=0, allow_cloud=task.allow_cloud_fallback)

        # 4. Simulation of on-device token generation
        perf = TIER_PERFORMANCE[tier]
        tokens_to_generate = min(task.max_tokens, max(10, len(clean_prompt) // 3))
        
        # Calculate simulated execution time and energy
        nominal_latency_ms = tokens_to_generate * perf["ms_per_token"]
        adjusted_latency_ms = nominal_latency_ms / speed_factor
        energy_mwh = round(tokens_to_generate * perf["mwh_per_token"], 4)

        output_text = f"[OnDevice:{tier.value}] Processed prompt ({len(clean_prompt)} chars). Generated {tokens_to_generate} tokens."

        return InferenceResult(
            task_id=task.task_id,
            output_text=output_text,
            chosen_tier=tier,
            tokens_generated=tokens_to_generate,
            latency_ms=round(adjusted_latency_ms, 2),
            energy_used_mwh=energy_mwh,
            privacy_verified=True
        )
