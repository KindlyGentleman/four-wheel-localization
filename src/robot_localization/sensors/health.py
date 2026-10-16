"""
robot_localization.sensors.health

Real-time sensor-health monitoring and fault diagnosis for ground robot localization.
Evaluates availability, observation latency, innovation statistics, and cross-sensor
consistency using strictly observable runtime data (no simulator ground truth).
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class LocalizationConfidence(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    LOST = "lost"
    RECOVERY_PENDING = "recovery_pending"


@dataclass
class DiagnosticMetrics:
    """Observable runtime health metrics for an individual sensor."""
    sensor_id: str
    is_available: bool = True
    last_timestamp: float = 0.0
    age_seconds: float = 0.0
    empirical_rate_hz: float = 0.0
    valid_fraction: float = 1.0
    mean_nis: float = 1.0
    total_observations: int = 0
    rejected_count: int = 0
    accepted_count: int = 0
    fault_flags: Dict[str, bool] = field(default_factory=dict)


class SensorHealthMonitor:
    """
    Supervises sensor streams and cross-sensor physical plausibility.
    Provides fault alarms, sensor gating recommendations, and overall localization confidence.
    """

    def __init__(
        self,
        slip_omega_thresh_rad_s: float = 0.08,    # ~4.6 deg/s yaw rate divergence
        slip_speed_thresh_m_s: float = 0.06,
        compass_max_nis: float = 6.635,          # chi^2_1 at 99%
        gnss_max_nis: float = 9.210,             # chi^2_2 at 99%
        outage_timeout_s: float = 2.0,
        history_window: int = 50
    ):
        self.slip_omega_thresh = slip_omega_thresh_rad_s
        self.slip_speed_thresh = slip_speed_thresh_m_s
        self.compass_max_nis = compass_max_nis
        self.gnss_max_nis = gnss_max_nis
        self.outage_timeout_s = outage_timeout_s
        self.history_window = history_window

        # Sensor state trackers
        self._metrics: Dict[str, DiagnosticMetrics] = {}
        self._timestamps_hist: Dict[str, List[float]] = {}
        self._nis_hist: Dict[str, List[float]] = {}

        # Cross-sensor buffer
        self._last_wheel_omega = 0.0
        self._last_wheel_v = 0.0
        self._last_gyro_z = 0.0
        self._last_gyro_bg = 0.0
        self._last_wheel_disagreement = 0.0

        # Overall localization status
        self._confidence = LocalizationConfidence.HEALTHY
        self._consecutive_rejected_gnss = 0
        self._recovery_buffer: List[Tuple[float, np.ndarray]] = []

    def register_sensor(self, sensor_id: str, nominal_rate_hz: float = 50.0):
        self._metrics[sensor_id] = DiagnosticMetrics(
            sensor_id=sensor_id,
            empirical_rate_hz=nominal_rate_hz
        )
        self._timestamps_hist[sensor_id] = []
        self._nis_hist[sensor_id] = []

    def update_wheel_odometry(
        self,
        t: float,
        v_forward: float,
        omega_wheel: float,
        wheel_disagreement: float = 0.0
    ):
        self._last_wheel_v = v_forward
        self._last_wheel_omega = omega_wheel
        self._last_wheel_disagreement = wheel_disagreement
        self._record_timestamp("wheel_encoders", t)

    def update_imu(self, t: float, gyro_z: float, estimated_bias: float = 0.0):
        self._last_gyro_z = gyro_z
        self._last_gyro_bg = estimated_bias
        self._record_timestamp("imu", t)

    def record_innovation(self, sensor_id: str, t: float, nis: float, accepted: bool):
        if sensor_id not in self._metrics:
            self.register_sensor(sensor_id)
        m = self._metrics[sensor_id]
        m.total_observations += 1
        if accepted:
            m.accepted_count += 1
        else:
            m.rejected_count += 1

        hist = self._nis_hist.setdefault(sensor_id, [])
        hist.append(nis)
        if len(hist) > self.history_window:
            hist.pop(0)
        m.mean_nis = float(np.mean(hist))

        if sensor_id == "gnss":
            if accepted:
                self._consecutive_rejected_gnss = 0
            else:
                self._consecutive_rejected_gnss += 1

    def _record_timestamp(self, sensor_id: str, t: float):
        if sensor_id not in self._metrics:
            self.register_sensor(sensor_id)
        m = self._metrics[sensor_id]
        m.last_timestamp = t
        thist = self._timestamps_hist.setdefault(sensor_id, [])
        thist.append(t)
        if len(thist) > self.history_window:
            thist.pop(0)
        if len(thist) >= 2:
            dt_avg = (thist[-1] - thist[0]) / (len(thist) - 1)
            m.empirical_rate_hz = 1.0 / dt_avg if dt_avg > 0 else 0.0

    def evaluate_health(self, current_time: float) -> Tuple[LocalizationConfidence, Dict[str, Any]]:
        """
        Evaluate full system health at current time instant using observable signals.
        """
        diagnostics = {}
        flags = {}

        # 1. Physical wheel-slip detection
        unbiased_gyro = self._last_gyro_z - self._last_gyro_bg
        delta_omega = abs(self._last_wheel_omega - unbiased_gyro)
        slip_active = (delta_omega > self.slip_omega_thresh) or (self._last_wheel_disagreement > self.slip_speed_thresh)
        flags["wheel_slip_suspected"] = slip_active
        flags["omega_divergence_rad_s"] = float(delta_omega)

        # 2. GNSS availability & health
        gnss_metrics = self._metrics.get("gnss")
        gnss_active = False
        gnss_outage = False
        if gnss_metrics is not None and gnss_metrics.total_observations > 0:
            gnss_age = current_time - gnss_metrics.last_timestamp
            gnss_metrics.age_seconds = gnss_age
            gnss_outage = gnss_age > self.outage_timeout_s
            gnss_active = not gnss_outage
            flags["gnss_outage"] = gnss_outage
            flags["gnss_consecutive_rejections"] = self._consecutive_rejected_gnss
            flags["gnss_multipath_suspected"] = (self._consecutive_rejected_gnss == 1 and not gnss_outage)

        # 3. Compass disturbance
        cmps_metrics = self._metrics.get("compass")
        if cmps_metrics is not None and cmps_metrics.total_observations > 0:
            compass_disturbed = cmps_metrics.mean_nis > self.compass_max_nis
            flags["compass_disturbance_suspected"] = compass_disturbed

        # 4. Overall confidence arbitration
        if gnss_outage and slip_active:
            confidence = LocalizationConfidence.LOST
        elif gnss_outage or slip_active or (self._consecutive_rejected_gnss > 0):
            if self._consecutive_rejected_gnss >= 2:
                # Potential returning GNSS waiting for consistency confirmation
                confidence = LocalizationConfidence.RECOVERY_PENDING
            else:
                confidence = LocalizationConfidence.DEGRADED
        else:
            confidence = LocalizationConfidence.HEALTHY

        self._confidence = confidence
        diagnostics["confidence"] = confidence.value
        diagnostics["flags"] = flags
        diagnostics["sensors"] = {sid: {
            "rate_hz": m.empirical_rate_hz,
            "age_s": current_time - m.last_timestamp if m.last_timestamp > 0 else -1.0,
            "mean_nis": m.mean_nis,
            "rejected": m.rejected_count,
            "accepted": m.accepted_count
        } for sid, m in self._metrics.items()}

        return confidence, diagnostics
