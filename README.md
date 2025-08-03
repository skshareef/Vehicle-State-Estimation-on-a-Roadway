# Vehicle State Estimation on a Roadway

This project implements an Error-State **Extended Kalman Filter (ES-EKF)** to localize a vehicle using data from the [CARLA](https://carla.org/) simulator.  
We fuse high-rate IMU data with GNSS and LIDAR position measurements to estimate position, velocity, and orientation in real-time.

<p align="center">
  <img src="images/diagram.png" width="500"/>
</p>

---

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. ES-EKF Algorithm Overview](#2-es-ekf-algorithm-overview)
- [3. Implementation Details](#3-implementation-details)
  - [3.1. State Initialization](#31-state-initialization)
  - [3.2. Prediction Step](#32-prediction-step)
  - [3.3. Correction Step](#33-correction-step)
- [4. Results](#4-results)
  - [4.1. Trajectory Comparison](#41-trajectory-comparison)
  - [4.2. Estimation Error](#42-estimation-error)
- [5. How to Run](#5-how-to-run)
- [6. References](#6-references)

---

## 1. Introduction

This project is the final programming assignment for the [State Estimation and Localization for Self-Driving Cars](https://www.coursera.org/learn/state-estimation-localization-self-driving-cars?) course from [Coursera](https://www.coursera.org/), using starter code from the University of Toronto.

The **Kalman Filter** algorithm works in two stages:
- **Prediction:** Using the vehicle's motion model and IMU data.
- **Correction:** Using GNSS and LIDAR position measurements.

---

## 2. ES-EKF Algorithm Overview

### State Vector

| Variable       | Description                          | Dimensions         |
| -------------- | ------------------------------------ | ------------------ |
| **$\mathbf{p}$** | Position (E, N, U)                  | 3                  |
| **$\mathbf{v}$** | Velocity (E, N, U)                  | 3                  |
| **$\mathbf{q}$** | Orientation (quaternion)            | 4                  |

**IMU provides:**

| Variable            | Description                           | Dimensions      |
| ------------------- | ------------------------------------- | --------------- |
| **$\mathbf{f}$**    | Specific force (acceleration, body frame) | 3            |
| **$\mathbf{\omega}$** | Angular rate (body frame)               | 3            |

### ES-EKF Pipeline

| Step        | Action                       | Source         | Frequency       |
| ----------- | ---------------------------- | -------------- | --------------- |
| Prediction  | Propagate state & covariance | IMU            | High-rate (Hz)  |
| Correction  | Update state & covariance    | GNSS/LIDAR     | Low-rate (async)|

---

## 3. Implementation Details

### 3.1. State Initialization

```python
p_est = np.zeros([imu_f.data.shape[0], 3])    # Position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])    # Velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])    # Orientation (quaternion)
p_cov = np.zeros([imu_f.data.shape[0], 9, 9]) # Covariance matrices

# Set initial values from ground truth
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
p_cov[0] = np.zeros(9)
gnss_i  = 0
lidar_i = 0
gnss_t = list(gnss.t)
lidar_t = list(lidar.t)
