"""
robot_localization.models.skid_steer

Kinematics of a 4-wheel skid-steered mobile robot, including wheel speed aggregation,
chassis motion derivation, individual wheel disagreement indicators, and longitudinal slip.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

from robot_localization.models.robot_4wd import Robot4WDConfig


@dataclass
class WheelSpeeds:
    """Individual angular speeds of the four wheels in rad/s."""
    fl: float
    fr: float
    rl: float
    rr: float

    def to_array(self) -> np.ndarray:
        return np.array([self.fl, self.fr, self.rl, self.rr])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "WheelSpeeds":
        return cls(fl=float(arr[0]), fr=float(arr[1]), rl=float(arr[2]), rr=float(arr[3]))


@dataclass
class ChassisVelocity:
    """Forward linear speed v (m/s) and yaw rate omega (rad/s)."""
    v: float
    omega: float

    def to_array(self) -> np.ndarray:
        return np.array([self.v, self.omega])


class SkidSteerKinematics:
    """
    Kinematic model for a planar four-wheel skid-steer robot.

    Equations:
        v_FL = r_FL * omega_FL,   v_FR = r_FR * omega_FR
        v_RL = r_RL * omega_RL,   v_RR = r_RR * omega_RR

        v_left  = (v_FL + v_RL) / 2
        v_right = (v_FR + v_RR) / 2

        v_forward = (v_left + v_right) / 2
        omega     = (v_right - v_left) / b_effective

    Under wheel slip ratio s_i in (-inf, 1]:
        v_ground_i = (1 - s_i) * r_i * omega_wheel_i
    """

    def __init__(self, config: Robot4WDConfig):
        self.config = config
        self.r_fl = config.wheels["FL"].radius_m
        self.r_fr = config.wheels["FR"].radius_m
        self.r_rl = config.wheels["RL"].radius_m
        self.r_rr = config.wheels["RR"].radius_m
        self.b_eff = config.effective_track_width_m

    def wheel_speeds_to_chassis(self, speeds: WheelSpeeds) -> ChassisVelocity:
        """
        Compute forward velocity v and angular velocity omega from wheel rotational speeds.
        Assumes no slip (pure kinematic mapping).
        """
        v_fl = speeds.fl * self.r_fl
        v_fr = speeds.fr * self.r_fr
        v_rl = speeds.rl * self.r_rl
        v_rr = speeds.rr * self.r_rr

        v_left = 0.5 * (v_fl + v_rl)
        v_right = 0.5 * (v_fr + v_rr)

        v = 0.5 * (v_left + v_right)
        omega = (v_right - v_left) / self.b_eff
        return ChassisVelocity(v=v, omega=omega)

    def chassis_to_wheel_speeds(self, cmd: ChassisVelocity) -> WheelSpeeds:
        """
        Inverse kinematics: Given commanded chassis forward speed and yaw rate,
        calculate required ideal wheel rotational speeds in rad/s.
        """
        v_left = cmd.v - 0.5 * self.b_eff * cmd.omega
        v_right = cmd.v + 0.5 * self.b_eff * cmd.omega

        omega_fl = v_left / self.r_fl
        omega_rl = v_left / self.r_rl
        omega_fr = v_right / self.r_fr
        omega_rr = v_right / self.r_rr

        return WheelSpeeds(fl=omega_fl, fr=omega_fr, rl=omega_rl, rr=omega_rr)

    def compute_ground_chassis_velocity(
        self,
        wheel_rot_speeds: WheelSpeeds,
        slip_ratios: Dict[str, float]
    ) -> ChassisVelocity:
        """
        Compute true chassis ground velocity under longitudinal wheel slip.

        v_ground_i = (1 - s_i) * r_i * omega_i
        """
        s_fl = slip_ratios.get("FL", 0.0)
        s_fr = slip_ratios.get("FR", 0.0)
        s_rl = slip_ratios.get("RL", 0.0)
        s_rr = slip_ratios.get("RR", 0.0)

        vg_fl = (1.0 - s_fl) * self.r_fl * wheel_rot_speeds.fl
        vg_fr = (1.0 - s_fr) * self.r_fr * wheel_rot_speeds.fr
        vg_rl = (1.0 - s_rl) * self.r_rl * wheel_rot_speeds.rl
        vg_rr = (1.0 - s_rr) * self.r_rr * wheel_rot_speeds.rr

        vg_left = 0.5 * (vg_fl + vg_rl)
        vg_right = 0.5 * (vg_fr + vg_rr)

        v_ground = 0.5 * (vg_left + vg_right)
        omega_ground = (vg_right - vg_left) / self.b_eff

        return ChassisVelocity(v=v_ground, omega=omega_ground)

    def compute_wheel_disagreement(self, linear_wheel_speeds: Dict[str, float]) -> Dict[str, float]:
        """
        Compute internal disagreement between front and rear wheels on each side,
        plus overall 4-wheel variance.

        Note on Limitation:
        Simultaneous slip on both wheels of one side produces d_left = 0,
        remaining invisible to a front/rear disagreement detector.
        """
        v_fl = linear_wheel_speeds["FL"]
        v_fr = linear_wheel_speeds["FR"]
        v_rl = linear_wheel_speeds["RL"]
        v_rr = linear_wheel_speeds["RR"]

        d_left = abs(v_fl - v_rl)
        d_right = abs(v_fr - v_rr)
        speed_var = float(np.var([v_fl, v_fr, v_rl, v_rr]))

        return {
            "d_left": d_left,
            "d_right": d_right,
            "speed_variance": speed_var,
            "v_left_avg": 0.5 * (v_fl + v_rl),
            "v_right_avg": 0.5 * (v_fr + v_rr),
        }

    def get_kinematic_jacobian(self) -> np.ndarray:
        """
        Jacobian mapping linear wheel speeds [v_FL, v_FR, v_RL, v_RR]^T to chassis [v, omega]^T:
            [v]     = [ 1/4   1/4   1/4   1/4 ] [v_FL]
            [omega]   [-1/(2b) 1/(2b) -1/(2b) 1/(2b)] [v_FR]
                                                      [v_RL]
                                                      [v_RR]
        """
        b = self.b_eff
        J = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [-1.0 / (2.0 * b), 1.0 / (2.0 * b), -1.0 / (2.0 * b), 1.0 / (2.0 * b)]
        ])
        return J
