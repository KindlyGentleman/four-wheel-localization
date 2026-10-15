"""
robot_localization.models.robot_4wd

Platform geometry, wheel configurations, mounting parameters, and encoder specifications.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import yaml


@dataclass
class WheelMountConfig:
    """Configuration for an individual wheel mount and dimensions."""
    name: str
    radius_m: float = 0.075
    x_offset_m: float = 0.175  # Longitudinal offset from body center (+ forward)
    y_offset_m: float = 0.180  # Lateral offset from body center (+ left)
    ticks_per_rev: int = 2048
    gear_ratio: float = 1.0


@dataclass
class Robot4WDConfig:
    """
    Configuration of the four-wheel skid-steer mobile robot platform.
    Allows decoupling between true physical geometry and estimator-assumed geometry.
    """
    drive_type: str = "skid_steer_4wd"
    body_length_m: float = 0.55
    body_width_m: float = 0.40
    wheel_radius_m: float = 0.075
    wheel_base_m: float = 0.35        # Front-rear axle distance
    track_width_m: float = 0.36       # Geometric lateral wheel center distance
    effective_track_width_m: float = 0.42 # Calibrated effective track width (b_eff > b)
    mass_kg: float = 18.5
    ticks_per_rev: int = 2048

    # Individual wheel configurations
    wheels: Dict[str, WheelMountConfig] = field(default_factory=dict)

    def __post_init__(self):
        if not self.wheels:
            # Default 4-wheel geometry: FL, FR, RL, RR
            half_base = self.wheel_base_m / 2.0
            half_track = self.track_width_m / 2.0
            self.wheels = {
                "FL": WheelMountConfig("FL", self.wheel_radius_m, half_base, half_track, self.ticks_per_rev),
                "FR": WheelMountConfig("FR", self.wheel_radius_m, half_base, -half_track, self.ticks_per_rev),
                "RL": WheelMountConfig("RL", self.wheel_radius_m, -half_base, half_track, self.ticks_per_rev),
                "RR": WheelMountConfig("RR", self.wheel_radius_m, -half_base, -half_track, self.ticks_per_rev),
            }

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Robot4WDConfig":
        """Load robot configuration from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        r_data = data.get("robot", {})
        enc_data = data.get("encoders", {})
        ticks = enc_data.get("ticks_per_wheel_revolution", 2048)

        config = cls(
            drive_type=r_data.get("drive_type", "skid_steer_4wd"),
            body_length_m=r_data.get("body_length_m", 0.55),
            body_width_m=r_data.get("body_width_m", 0.40),
            wheel_radius_m=r_data.get("wheel_radius_m", 0.075),
            wheel_base_m=r_data.get("wheel_base_m", 0.35),
            track_width_m=r_data.get("track_width_m", 0.36),
            effective_track_width_m=r_data.get("effective_track_width_m", 0.42),
            mass_kg=r_data.get("mass_kg", 18.5),
            ticks_per_rev=ticks,
        )

        # Override individual wheels if specified in yaml
        ind_data = data.get("individual_wheels", {})
        if ind_data:
            config.wheels = {}
            for name, w_info in ind_data.items():
                config.wheels[name] = WheelMountConfig(
                    name=name,
                    radius_m=w_info.get("radius_m", config.wheel_radius_m),
                    x_offset_m=w_info.get("x_offset_m", 0.0),
                    y_offset_m=w_info.get("y_offset_m", 0.0),
                    ticks_per_rev=ticks,
                    gear_ratio=enc_data.get("gear_ratio", 1.0),
                )
        return config
