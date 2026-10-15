# Novelty Claims and Physical Limitations Ledger

## 1. Scope of Contribution and Non-Claims

### 1.1 What This Study DOES NOT Claim
1. **No Algorithmic Novelty in 2D EKF:** Extended Kalman Filter sensor fusion for mobile robots is a mature, well-established discipline pioneered by Smith, Cheeseman (1986), Thrun et al. (2005), and Moore & Stouch (ROS `robot_localization`, 2014). This study does not claim novelty in the basic EKF formulation.
2. **No Physical Hardware Validation:** All evaluations are conducted on a validated CPU-based numerical simulator. No claims of field deployment readiness or physical vehicle durability are made.
3. **No Sensor Manufacturer Superiority:** The CMPS12 and MPU-9250 profiles reflect representative sensor characteristics and documented datasheets, but do not imply empirical benchmarking of physical semiconductor chips.

### 1.2 Legitimate Reproducibility & Engineering Contributions
1. **Decoupled Ground-Truth Kinematics:** Resolves the common simulation flaw of using encoder-integrated odometry as ground truth; chassis motion is integrated strictly from ground-wheel contact velocities under physical slip.
2. **Four Independent Quantized Wheel Encoder Streams:** Implements 4 distinct incremental channels with integer pulse quantization, jitter, and missed counts, with Jacobian-based covariance propagation.
3. **Audit and Correction of Upstream Reference:** Formally identifies and corrects the covariance-versus-standard-deviation noise scaling bug and state redundancy in the popular PythonRobotics reference code.
4. **Transparent Evaluation of Sensor Failure Interactions:** Demonstrates empirically that adding sensors does not monotonically improve localization: uncalibrated compasses degrade localization by $14\times$ during magnetic disturbance, and gyro bias corrupts EKF course prediction unless explicitly estimated.

---

## 2. Explicit Physical & Modeling Limitations

1. **Kinematic Slip vs. Dynamic Terramechanics:**
   The simulator employs an algebraic longitudinal slip ratio $v_{\text{ground}} = (1-s) r \omega$. It does not model nonlinear Pacejka magic formula tire dynamics, terramechanics shear displacement, soil sinkage, or suspension roll/pitch dynamics.
2. **Planar 2D Motion Assumption:**
   The state vector $x = [p_x, p_y, \theta]^T$ assumes motion constrained to a horizontal plane. Undulating terrain, curbs, and 3D slope dynamics are neglected.
3. **Front/Rear Slip Disagreement Blind Spot:**
   The front/rear wheel speed difference metric $d_{\text{left}} = |v_{\text{FL}} - v_{\text{RL}}|$ is strictly blind to simultaneous, uniform slip on both wheels of a side. Detection in such cases depends entirely on gyro angular rate discrepancy $\Delta\omega = |\omega_{\text{wheel}} - \omega_{\text{gyro}}|$.
4. **Known Landmark Association Assumption:**
   The LiDAR range-bearing update assumes known data association (landmarks possess explicit identifiers). Real-world perceptual ambiguity and false correspondences in unstructured environments are outside this model scope.
