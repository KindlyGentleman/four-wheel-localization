"""
sensors.py

Top-level convenience facade for sensor interfaces, registry, and health supervision.
Supports dynamic enabling/disabling, fault injection, and health state reporting.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure local src/ is accessible
_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from robot_localization.sensors.registry import (
    SensorRegistry,
    SensorConfig,
    SensorObservation,
    SensorType,
    SensorHealthState,
)
from robot_localization.sensors.health import (
    SensorHealthMonitor,
    DiagnosticMetrics,
    LocalizationConfidence,
)
from robot_localization.sensors.encoder import FourWheelEncoders, EncoderChannel, EncoderSample
from robot_localization.sensors.mpu9250 import MPU9250Sensor, MPU9250Sample
from robot_localization.sensors.cmps12 import CMPS12Sensor, CMPS12Sample
from robot_localization.sensors.gnss import GNSSReceiver, GNSSFix
from robot_localization.sensors.lidar import KnownLandmarkLiDAR, LandmarkObservation, LiDARScan
from robot_localization.sensors.ultrasonic import UltrasonicArray, UltrasonicEcho

# Backwards-compatible aliases
SensorHealthMetrics = DiagnosticMetrics
SensorStatus = SensorHealthState
SensorChannelHealth = DiagnosticMetrics
IMUSample = MPU9250Sample
CompassSample = CMPS12Sample
UltrasonicReading = UltrasonicEcho
PlanarLiDAR2D = KnownLandmarkLiDAR


def load_sensor_registry(yaml_path: Optional[str] = None) -> SensorRegistry:
    """
    Load a configured SensorRegistry instance from a YAML file or defaults.

    Args:
        yaml_path: Path to YAML sensor configuration. If None, default sensors are registered.

    Returns:
        SensorRegistry instance
    """
    if yaml_path is not None and Path(yaml_path).exists():
        return SensorRegistry.from_yaml(Path(yaml_path).read_text(encoding="utf-8"))
    return SensorRegistry()


__all__ = [
    "SensorRegistry",
    "SensorConfig",
    "SensorObservation",
    "SensorType",
    "SensorStatus",
    "SensorHealthState",
    "SensorHealthMonitor",
    "SensorHealthMetrics",
    "SensorChannelHealth",
    "LocalizationConfidence",
    "FourWheelEncoders",
    "EncoderChannel",
    "EncoderSample",
    "MPU9250Sensor",
    "IMUSample",
    "CMPS12Sensor",
    "CompassSample",
    "GNSSReceiver",
    "GNSSFix",
    "KnownLandmarkLiDAR",
    "PlanarLiDAR2D",
    "LandmarkObservation",
    "LiDARScan",
    "UltrasonicArray",
    "UltrasonicReading",
    "load_sensor_registry",
]
