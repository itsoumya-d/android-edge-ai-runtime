# Android Edge AI Runtime

On-device mobile AI inference execution manager and privacy governor designed for sustainable, resource-constrained mobile platforms.

## Core Capabilities
- Thermal and battery-aware dynamic quantization selection (FP16, INT8, INT4).
- Offline-first execution ensuring zero telemetry leakage over network sockets.
- Device thermal throttling mitigation with token throughput rate adaptation.
- Memory budget allocation and peak footprint profiling.

## Testing
```bash
PYTHONPATH=. pytest -v
```
