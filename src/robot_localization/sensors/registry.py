"""
robot_localization.sensors.registry

Modular, configurable sensor registry supporting dynamic sensor configuration,
lifecycle management (enable, disable, fault injection), and standardized
sensor observation structures.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import copy
import yaml
import numpy as np

from robot_localization.models.robot_4wd import Robot4WDConfig


class SensorType(str, Enum):
    WHEEL_ENCODERS = "wheel_encoders"
    IMU = "imu"
    COMPASS = "compass"
    GNSS = "gnss"
    STEREO = "stereo"
    LIDAR = "lidar"
    ULTRASONIC = "ultrasonic"


class SensorHealthState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULTY = "faulty"
    INACTIVE = "inactive"


@dataclass
class SensorObservation:
    """
    Standardized observation packet emitted by all sensor modalities.
    """
    timestamp: float                    # Sensor measurement timestamp (s)
    sensor_id: str                      # Identifier e.g. 'wheel_encoders', 'gnss_0'
    frame_id: str                       # Coordinate frame e.g. 'base_link', 'gnss_antenna'
    measurement: Any                    # Numerical payload (array, scalar, dict, or images)
    covariance: Optional[np.ndarray] = None # Measurement error covariance
    validity: bool = True               # False if sensor reports internal failure or dropout
    quality_indicators: Dict[str, Any] = field(default_factory=dict) # Metadata e.g. HDOP, inliers
    arrival_timestamp: float = 0.0      # Arrival timestamp at estimator queue (s)

    def __post_init__(self):
        if self.arrival_timestamp <= 0.0:
            self.arrival_timestamp = self.timestamp


@dataclass
class SensorConfig:
    """Configuration parameters for an individual sensor module."""
    sensor_type: SensorType
    enabled: bool = True
    rate_hz: float = 50.0
    frame_id: str = "base_link"
    mounting_offset_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mounting_rpy_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    params: Dict[str, Any] = field(default_factory=dict)
    active_faults: Dict[str, Any] = field(default_factory=dict)


class SensorRegistry:
    """
    Registry and lifecycle coordinator for mobile robot sensors.
    Supports enable/disable, fault injection, configuration validation,
    and history tracking.
    """

    def __init__(self, robot_config: Optional[Robot4WDConfig] = None):
        self.robot_config = robot_config if robot_config is not None else Robot4WDConfig()
        self._configs: Dict[str, SensorConfig] = {}
        self._health_states: Dict[str, SensorHealthState] = {}
        self._observation_history: Dict[str, List[SensorObservation]] = {}
        self._max_history_len: int = 200

        # Register default sensor specifications
        self._register_default_sensors()

    def _register_default_sensors(self):
        """Register the standard suite of ground-rover sensors."""
        defaults = [
            SensorConfig(
                sensor_type=SensorType.WHEEL_ENCODERS,
                enabled=True,
                rate_hz=50.0,
                frame_id="base_link",
                params={"ticks_per_rev": 2048, "quantization_noise": True}
            ),
            SensorConfig(
                sensor_type=SensorType.IMU,
                enabled=True,
                rate_hz=50.0,
                frame_id="imu_link",
                mounting_offset_xyz=(0.0, 0.0, 0.10),
                params={"gyro_bias_rad_s": 0.015, "gyro_walk_std": 0.001, "estimate_bias": True}
            ),
            SensorConfig(
                sensor_type=SensorType.COMPASS,
                enabled=False,
                rate_hz=10.0,
                frame_id="base_link",
                params={"declination_rad": 0.0, "noise_std_deg": 1.5}
            ),
            SensorConfig(
                sensor_type=SensorType.GNSS,
                enabled=True,
                rate_hz=1.0,
                frame_id="gnss_antenna",
                mounting_offset_xyz=(0.05, 0.0, 0.25),
                params={"horizontal_std_m": 0.50, "dropout_prob": 0.0}
            ),
            SensorConfig(
                sensor_type=SensorType.STEREO,
                enabled=False,
                rate_hz=10.0,
                frame_id="camera_link",
                mounting_offset_xyz=(0.20, 0.0, 0.20),
                params={
                    "width": 640,
                    "height": 480,
                    "baseline_m": 0.10,
                    "focal_length_px": 500.0,
                    "disparity_method": "sgbm"
                }
            ),
            SensorConfig(
                sensor_type=SensorType.LIDAR,
                enabled=False,
                rate_hz=5.0,
                frame_id="lidar_link",
                mounting_offset_xyz=(0.15, 0.0, 0.22),
                params={"num_beams": 16, "max_range_m": 12.0, "fov_deg": 360.0}
            ),
            SensorConfig(
                sensor_type=SensorType.ULTRASONIC,
                enabled=False,
                rate_hz=10.0,
                frame_id="base_link",
                params={"num_transducers": 3, "max_range_m": 3.0}
            )
        ]
        for cfg in defaults:
            s_id = cfg.sensor_type.value
            self._configs[s_id] = cfg
            self._health_states[s_id] = SensorHealthState.HEALTHY if cfg.enabled else SensorHealthState.INACTIVE
            self._observation_history[s_id] = []

    def configure_sensor(
        self,
        sensor_id: str,
        enabled: Optional[bool] = None,
        rate_hz: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
        mounting_offset_xyz: Optional[Tuple[float, float, float]] = None
    ) -> None:
        """Update sensor configuration dynamically."""
        if sensor_id not in self._configs:
            raise KeyError(f"Sensor '{sensor_id}' is not registered.")
        cfg = self._configs[sensor_id]
        if enabled is not None:
            cfg.enabled = enabled
            self._health_states[sensor_id] = SensorHealthState.HEALTHY if enabled else SensorHealthState.INACTIVE
        if rate_hz is not None:
            if rate_hz <= 0:
                raise ValueError(f"Rate must be positive, got {rate_hz}")
            cfg.rate_hz = rate_hz
        if params is not None:
            cfg.params.update(params)
        if mounting_offset_xyz is not None:
            cfg.mounting_offset_xyz = mounting_offset_xyz

    def enable_sensor(self, sensor_id: str) -> None:
        self.configure_sensor(sensor_id, enabled=True)

    def disable_sensor(self, sensor_id: str) -> None:
        self.configure_sensor(sensor_id, enabled=False)

    def is_enabled(self, sensor_id: str) -> bool:
        return self._configs.get(sensor_id, SensorConfig(SensorType.WHEEL_ENCODERS, enabled=False)).enabled

    def inject_fault(self, sensor_id: str, fault_type: str, **params) -> None:
        """Inject a fault scenario (e.g. blackout, noise inflation, bias jump)."""
        if sensor_id not in self._configs:
            raise KeyError(f"Sensor '{sensor_id}' is not registered.")
        self._configs[sensor_id].active_faults[fault_type] = params
        self._health_states[sensor_id] = SensorHealthState.FAULTY

    def clear_faults(self, sensor_id: Optional[str] = None) -> None:
        """Clear active faults on a specific sensor or all sensors."""
        targets = [sensor_id] if sensor_id else list(self._configs.keys())
        for sid in targets:
            if sid in self._configs:
                self._configs[sid].active_faults.clear()
                if self._configs[sid].enabled:
                    self._health_states[sid] = SensorHealthState.HEALTHY

    def get_config(self, sensor_id: str) -> SensorConfig:
        if sensor_id not in self._configs:
            raise KeyError(f"Sensor '{sensor_id}' is not registered.")
        return copy.deepcopy(self._configs[sensor_id])

    def get_all_configs(self) -> Dict[str, SensorConfig]:
        return {sid: copy.deepcopy(cfg) for sid, cfg in self._configs.items()}

    def get_enabled_sensors(self) -> List[str]:
        return [sid for sid, cfg in self._configs.items() if cfg.enabled]

    def record_observation(self, obs: SensorObservation) -> None:
        """Record an observation into the rolling history."""
        if obs.sensor_id not in self._observation_history:
            self._observation_history[obs.sensor_id] = []
        hist = self._observation_history[obs.sensor_id]
        hist.append(obs)
        if len(hist) > self._max_history_len:
            hist.pop(0)

    def get_history(self, sensor_id: str) -> List[SensorObservation]:
        return list(self._observation_history.get(sensor_id, []))

    def set_health_state(self, sensor_id: str, state: SensorHealthState) -> None:
        self._health_states[sensor_id] = state

    def get_health_state(self, sensor_id: str) -> SensorHealthState:
        return self._health_states.get(sensor_id, SensorHealthState.INACTIVE)

    def validate_for_estimator(self, estimator_name: str, has_prior_map: bool = False) -> Tuple[bool, str]:
        """
        Validate whether the current sensor configuration can support the requested estimator.
        Rejects incompatible or logically impossible configurations.
        """
        enabled = self.get_enabled_sensors()

        # Encoders are required for all dead-reckoning / skid-steer filters
        # unless visual odometry + IMU is explicitly active
        has_wheels = SensorType.WHEEL_ENCODERS.value in enabled
        has_imu = SensorType.IMU.value in enabled
        has_stereo = SensorType.STEREO.value in enabled
        has_gnss = SensorType.GNSS.value in enabled
        has_lidar = SensorType.LIDAR.value in enabled

        if not has_wheels and not (has_stereo and has_imu):
            return False, f"Estimator '{estimator_name}' requires wheel encoders or stereo+IMU."

        if estimator_name in ("particle_filter", "amcl"):
            if not has_prior_map:
                return False, f"{estimator_name.upper()} requires a known prior occupancy map."
            if not has_lidar and not has_stereo:
                return False, f"{estimator_name.upper()} requires range observations (LiDAR or stereo point cloud)."

        if estimator_name == "bias_aware_ekf" and not has_imu:
            return False, "Bias-aware EKF requires an active gyroscope / IMU to co-estimate bias."

        if estimator_name == "gnss_only" and not has_gnss:
            return False, "GNSS-only mode requires GNSS to be enabled."

        return True, "Configuration valid."

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "SensorRegistry":
        """Construct registry from YAML configuration string."""
        data = yaml.safe_load(yaml_str)
        registry = cls()
        sensors_dict = data.get("sensors", {})
        for s_name, s_cfg in sensors_dict.items():
            if s_name in registry._configs:
                enabled = s_cfg.get("enabled", None)
                rate = s_cfg.get("rate_hz", None)
                offset = s_cfg.get("mounting_offset_xyz", None)
                if offset is not None:
                    offset = tuple(offset)
                # Parameters are other keys
                reserved = {"enabled", "rate_hz", "frame_id", "mounting_offset_xyz", "mounting_rpy_rad"}
                params = {k: v for k, v in s_cfg.items() if k not in reserved}
                registry.configure_sensor(
                    s_name,
                    enabled=enabled,
                    rate_hz=rate,
                    params=params,
                    mounting_offset_xyz=offset
                )
        return registry

    def to_yaml(self) -> str:
        """Export current sensor configuration to YAML format."""
        out = {"robot": {"model": "skid_steer_4wd"}, "sensors": {}}
        for sid, cfg in self._configs.items():
            s_dict = {
                "enabled": cfg.enabled,
                "rate_hz": float(cfg.rate_hz),
                "frame_id": cfg.frame_id,
                "mounting_offset_xyz": list(cfg.mounting_offset_xyz),
            }
            s_dict.update(cfg.params)
            out["sensors"][sid] = s_dict
        return yaml.dump(out, sort_keys=False)
