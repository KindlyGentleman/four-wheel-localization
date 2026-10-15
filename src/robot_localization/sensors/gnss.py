"""
robot_localization.sensors.gnss

Configurable Global Navigation Satellite System (GNSS / GPS) receiver simulation.
Features:
- Local East-North-Up (ENU) coordinates with fixed origin datum
- WGS-84 ellipsoidal coordinate conversion (latitude, longitude, altitude)
- Antenna lever-arm transformation relative to robot center
- Additive Gaussian noise, time-correlated bias drift, and multipath disturbances
- Stochastic packet dropout and contiguous blackout intervals
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from robot_localization.models.kinematics import rotation_matrix_2d


@dataclass
class GNSSFix:
    """A discrete GNSS fix packet."""
    timestamp: float
    x_enu_m: float
    y_enu_m: float
    lat_deg: float
    lon_deg: float
    alt_m: float
    cov_enu: np.ndarray       # 2x2 horizontal covariance matrix
    is_valid: bool            # False during dropout or outage
    fix_quality: int          # 0 = invalid, 1 = autonomous GNSS, 2 = DGPS / RTK float
    multipath_active: bool    # Diagnostic flag


class GNSSReceiver:
    """
    Simulated GNSS receiver with realistic error dynamics and datum transformations.
    """

    def __init__(
        self,
        sample_rate_hz: float = 1.0,
        horizontal_std_m: float = 0.50,
        bias_drift_std_m_s: float = 0.005,
        dropout_prob: float = 0.0,
        lever_arm_m: Tuple[float, float] = (0.05, 0.0),
        outage_intervals: Optional[List[Tuple[float, float]]] = None,
        datum_lat_deg: float = -6.200000,
        datum_lon_deg: float = 106.816666,
        datum_alt_m: float = 25.0,
        rng: Optional[np.random.RandomState] = None
    ):
        self.sample_rate_hz = sample_rate_hz
        self.dt = 1.0 / sample_rate_hz
        self.horizontal_std_m = horizontal_std_m
        self.bias_drift_std_m_s = bias_drift_std_m_s
        self.dropout_prob = dropout_prob
        self.lever_arm_m = np.array(lever_arm_m, dtype=float)
        self.outage_intervals = outage_intervals or []

        # Reference datum coordinates
        self.datum_lat = datum_lat_deg
        self.datum_lon = datum_lon_deg
        self.datum_alt = datum_alt_m

        # WGS-84 ellipsoid constants
        self.WGS84_A = 6378137.0         # semi-major axis (meters)
        self.WGS84_E2 = 0.00669437999014 # first eccentricity squared

        self.rng = rng if rng is not None else np.random.RandomState()

        # Slowly varying bias state (Gauss-Markov process)
        self.bias_enu = np.zeros(2)
        # Temporary multipath bias state
        self.multipath_bias = np.zeros(2)
        self.multipath_active = False

        self.last_sample_time = -1.0

    def enu_to_geodetic(self, x_east: float, y_north: float, z_up: float = 0.0) -> Tuple[float, float, float]:
        """
        Convert local ENU coordinates (meters) to WGS-84 (lat deg, lon deg, alt m).
        Uses local planar approximation around reference datum.
        """
        phi = math.radians(self.datum_lat)
        # Radii of curvature
        N = self.WGS84_A / math.sqrt(1.0 - self.WGS84_E2 * (math.sin(phi) ** 2))
        M = self.WGS84_A * (1.0 - self.WGS84_E2) / ((1.0 - self.WGS84_E2 * (math.sin(phi) ** 2)) ** 1.5)

        delta_lat = y_north / (M + self.datum_alt)
        delta_lon = x_east / ((N + self.datum_alt) * math.cos(phi))

        lat = self.datum_lat + math.degrees(delta_lat)
        lon = self.datum_lon + math.degrees(delta_lon)
        alt = self.datum_alt + z_up
        return lat, lon, alt

    def trigger_multipath_event(self, bias_east: float = 3.0, bias_north: float = 3.0, active: bool = True):
        """Inject or clear a multipath bias disturbance."""
        self.multipath_active = active
        self.multipath_bias = np.array([bias_east, bias_north]) if active else np.zeros(2)

    def is_in_outage(self, t: float) -> bool:
        """Check if current timestamp falls within a programmed outage interval."""
        for t_start, t_end in self.outage_intervals:
            if t_start <= t <= t_end:
                return True
        return False

    def update(
        self,
        timestamp: float,
        robot_pose: np.ndarray,
        dt_since_last: Optional[float] = None
    ) -> GNSSFix:
        """
        Generate GNSS fix given true robot pose [x, y, theta].
        """
        dt = dt_since_last if dt_since_last is not None else self.dt
        self.last_sample_time = timestamp

        # True antenna location in world frame
        theta = float(robot_pose[2])
        R = rotation_matrix_2d(theta)
        p_chassis = np.array([robot_pose[0], robot_pose[1]])
        p_ant_true = p_chassis + R @ self.lever_arm_m

        # Check contiguous outages and stochastic dropout
        in_outage = self.is_in_outage(timestamp)
        is_dropped = (self.rng.rand() < self.dropout_prob)
        is_valid = not (in_outage or is_dropped)

        # Update slowly varying bias (random walk)
        if dt > 0 and self.bias_drift_std_m_s > 0:
            drift_step = self.rng.normal(0.0, self.bias_drift_std_m_s * math.sqrt(dt), size=2)
            self.bias_enu += drift_step

        # Measurement noise
        meas_noise = self.rng.normal(0.0, self.horizontal_std_m, size=2)

        # Combine
        total_error = meas_noise + self.bias_enu + (self.multipath_bias if self.multipath_active else 0.0)
        p_meas = p_ant_true + total_error

        # Covariance matrix reported by receiver
        cov = np.diag([self.horizontal_std_m ** 2, self.horizontal_std_m ** 2])

        # Convert to lat / lon
        lat, lon, alt = self.enu_to_geodetic(p_meas[0], p_meas[1], 0.0)

        quality = 1 if is_valid else 0

        return GNSSFix(
            timestamp=timestamp,
            x_enu_m=p_meas[0],
            y_enu_m=p_meas[1],
            lat_deg=lat,
            lon_deg=lon,
            alt_m=alt,
            cov_enu=cov,
            is_valid=is_valid,
            fix_quality=quality,
            multipath_active=self.multipath_active
        )
