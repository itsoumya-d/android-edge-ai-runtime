from .models import DeviceStatus, QuantizationTier, ThermalEmergencyException

class ThermalGovernor:
    """
    Dynamically adjusts model quantization and execution throttling based on
    device thermal headroom and battery reserve.
    """
    CRITICAL_TEMP_C = 48.0
    THROTTLE_TEMP_C = 40.0
    LOW_BATTERY_PCT = 20.0
    MIN_RAM_MB_FP16 = 2048

    @classmethod
    def evaluate_policy(cls, status: DeviceStatus) -> tuple[QuantizationTier, float]:
        """
        Returns (QuantizationTier, speed_factor) where speed_factor regulates token generation rate.
        """
        if status.temperature_c >= cls.CRITICAL_TEMP_C:
            raise ThermalEmergencyException(
                f"Thermal limit exceeded: {status.temperature_c}°C >= {cls.CRITICAL_TEMP_C}°C",
                current_temp=status.temperature_c
            )

        # Severe battery or hot thermal state -> INT4 quantized, half-speed
        if status.temperature_c >= cls.THROTTLE_TEMP_C or status.battery_pct <= cls.LOW_BATTERY_PCT:
            return QuantizationTier.INT4, 0.50

        # Limited RAM or moderate battery -> INT8 quantized, full-speed
        if status.available_ram_mb < cls.MIN_RAM_MB_FP16 or status.battery_pct < 50.0:
            return QuantizationTier.INT8, 1.0

        # Plentiful battery / charging with low temp -> FP16 full precision
        if status.is_charging or status.battery_pct >= 50.0:
            return QuantizationTier.FP16, 1.0

        return QuantizationTier.INT8, 1.0
