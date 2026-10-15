"""
robot_localization.sensors.encoder

Four independent incremental optical/magnetic wheel encoder channels.
Simulates integer quantization, missed pulses, count jitter, signed cumulative ticks,
and physically consistent derivation of wheel rotation and linear travel.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from robot_localization.models.robot_4wd import Robot4WDConfig, WheelMountConfig


@dataclass
class EncoderSample:
    """A single discrete encoder measurement packet."""
    timestamp: float
    wheel_id: str                      # 'FL', 'FR', 'RL', 'RR'
    cumulative_ticks: int              # Signed integer tick accumulator
    delta_ticks: int                   # Incremental ticks since last sample
    dt: float                          # Time interval since last sample
    delta_angle_rad: float             # Angular displacement in radians
    angular_speed_rad_s: float         # Angular velocity in rad/s
    linear_travel_m: float             # Assumed travel on ground: 2*pi*r*ticks/N
    linear_speed_m_s: float            # Assumed wheel linear speed v_i = r_i * omega_i
    is_valid: bool                     # True if measurement is valid
    variance_speed: float              # Theoretical quantization & jitter variance


class EncoderChannel:
    """
    Simulation of an individual wheel optical encoder channel.
    Tracks exact continuous wheel angle, quantizes into discrete integer counts,
    and applies realistic corruption (quantization, jitter, missed counts, dropouts).
    """

    def __init__(
        self,
        wheel_id: str,
        wheel_config: WheelMountConfig,
        ticks_per_rev: int = 2048,
        quantization_enabled: bool = True,
        jitter_std_ticks: float = 0.5,
        missed_count_prob: float = 0.0005,
        rng: Optional[np.random.RandomState] = None
    ):
        self.wheel_id = wheel_id
        self.config = wheel_config
        self.ticks_per_rev = ticks_per_rev
        self.quantization_enabled = quantization_enabled
        self.jitter_std_ticks = jitter_std_ticks
        self.missed_count_prob = missed_count_prob
        self.rng = rng if rng is not None else np.random.RandomState()

        # Continuous state
        self.continuous_angle_rad = 0.0
        self.last_quantized_ticks = 0
        self.cumulative_ticks = 0
        self.last_timestamp: Optional[float] = None

    def update(
        self,
        timestamp: float,
        rotational_speed_rad_s: float,
        dt: float,
        force_dropout: bool = False
    ) -> EncoderSample:
        """
        Integrate continuous angle and produce discrete encoder sample.
        """
        # Integrate true continuous angle
        self.continuous_angle_rad += rotational_speed_rad_s * dt

        # Convert continuous angle to ideal tick count: ticks = angle * N / (2*pi)
        ideal_ticks_float = self.continuous_angle_rad * self.ticks_per_rev / (2.0 * math.pi)

        # Apply count jitter if enabled
        if self.jitter_std_ticks > 0.0:
            jitter = self.rng.normal(0.0, self.jitter_std_ticks)
            measured_ticks_float = ideal_ticks_float + jitter
        else:
            measured_ticks_float = ideal_ticks_float

        # Quantize to integer
        if self.quantization_enabled:
            current_quantized = int(round(measured_ticks_float))
        else:
            current_quantized = int(measured_ticks_float)

        # Compute delta
        delta_ticks = current_quantized - self.last_quantized_ticks

        # Apply missed counts (stochastic pulse loss under high dynamics/noise)
        if self.missed_count_prob > 0.0 and delta_ticks != 0:
            if self.rng.rand() < self.missed_count_prob:
                # Miss 1 tick
                sign = 1 if delta_ticks > 0 else -1
                delta_ticks -= sign
                current_quantized = self.last_quantized_ticks + delta_ticks

        self.last_quantized_ticks = current_quantized
        self.cumulative_ticks = current_quantized
        self.last_timestamp = timestamp

        is_valid = not force_dropout
        if force_dropout:
            delta_ticks = 0

        # Physical conversions:
        # delta_theta = 2*pi * delta_ticks / N
        delta_angle = 2.0 * math.pi * delta_ticks / self.ticks_per_rev
        angular_speed = delta_angle / dt if dt > 0 else 0.0

        # Linear conversions with wheel radius:
        # delta_s = 2*pi*r * delta_ticks / N
        linear_travel = 2.0 * math.pi * self.config.radius_m * delta_ticks / self.ticks_per_rev
        linear_speed = linear_travel / dt if dt > 0 else 0.0

        # Theoretical speed variance due to uniform quantization over [-0.5, 0.5] ticks:
        # Var(ticks) = 1/12 + sigma_jitter^2
        # Var(v) = (2*pi*r / (N*dt))^2 * Var(ticks)
        tick_var = (1.0 / 12.0) + (self.jitter_std_ticks ** 2)
        scale_factor = (2.0 * math.pi * self.config.radius_m) / (self.ticks_per_rev * dt)
        speed_var = (scale_factor ** 2) * tick_var

        return EncoderSample(
            timestamp=timestamp,
            wheel_id=self.wheel_id,
            cumulative_ticks=self.cumulative_ticks,
            delta_ticks=delta_ticks,
            dt=dt,
            delta_angle_rad=delta_angle,
            angular_speed_rad_s=angular_speed,
            linear_travel_m=linear_travel,
            linear_speed_m_s=linear_speed,
            is_valid=is_valid,
            variance_speed=speed_var
        )


class FourWheelEncoders:
    """
    Manages four distinct encoder channels: FL, FR, RL, RR.
    """

    def __init__(
        self,
        robot_config: Robot4WDConfig,
        quantization_enabled: bool = True,
        jitter_std_ticks: float = 0.5,
        missed_count_prob: float = 0.0005,
        rng: Optional[np.random.RandomState] = None
    ):
        self.robot_config = robot_config
        self.rng = rng if rng is not None else np.random.RandomState()
        self.channels: Dict[str, EncoderChannel] = {
            wheel_id: EncoderChannel(
                wheel_id=wheel_id,
                wheel_config=robot_config.wheels[wheel_id],
                ticks_per_rev=robot_config.ticks_per_rev,
                quantization_enabled=quantization_enabled,
                jitter_std_ticks=jitter_std_ticks,
                missed_count_prob=missed_count_prob,
                rng=self.rng
            )
            for wheel_id in ["FL", "FR", "RL", "RR"]
        }

    def update(
        self,
        timestamp: float,
        wheel_rot_speeds: Dict[str, float],
        dt: float,
        dropouts: Optional[Dict[str, bool]] = None
    ) -> Dict[str, EncoderSample]:
        """
        Step all four encoders forward by dt and return individual samples.
        """
        dropouts = dropouts or {}
        samples = {}
        for wheel_id, channel in self.channels.items():
            speed = wheel_rot_speeds.get(wheel_id, 0.0)
            is_dropped = dropouts.get(wheel_id, False)
            samples[wheel_id] = channel.update(timestamp, speed, dt, force_dropout=is_dropped)
        return samples
