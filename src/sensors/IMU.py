# -*- coding: utf-8 -*-
# Filename: IMU.py

"""
A python script containing various inertial measurement unit (IMU) models
Created on 27 DEC 2025
Author: Alex Springer
"""

import numpy as np

class Accelerometer:
    def __init__(
        self,
        r_sensor,
        R_sb=np.eye(3),
        bias=np.zeros(3),
        bias_rw_std=0.0,
        noise_std=0.0,
        scale_factors=np.ones(3),
        seed=None
    ):
        """
        r_sensor        : 3-vector, sensor position in body frame
        R_sb            : 3x3 DCM, sensor frame relative to body frame
        bias            : initial accelerometer bias
        bias_rw_std     : bias random walk standard deviation (per sqrt(sec))
        noise_std       : white noise standard deviation
        scale_factors   : per-axis scale factors
        """

        self.r_sensor = np.asarray(r_sensor, dtype=float)
        self.R_sb = np.asarray(R_sb, dtype=float)

        self.bias = np.asarray(bias, dtype=float)
        self.bias_rw_std = bias_rw_std
        self.noise_std = noise_std
        self.scale_factors = np.asarray(scale_factors, dtype=float)

        self.rng = np.random.default_rng(seed)

    def step(
        self,
        a_cg,
        omega,
        alpha,
        R_bn,
        gravity_n,
        dt
    ):
        """
        a_cg        : translational acceleration of CG in nav frame
        omega       : angular rate in body frame
        alpha       : angular acceleration in body frame
        R_bn        : DCM body-to-nav
        gravity_n   : gravity vector in nav frame
        dt          : time step
        """

        # Rotate translational acceleration into body frame
        a_cg_b = R_bn.T @ a_cg

        # Lever-arm acceleration terms (body frame)
        a_euler = np.cross(alpha, self.r_sensor)
        a_centripetal = np.cross(omega, np.cross(omega, self.r_sensor))

        # Gravity in body frame
        g_b = R_bn.T @ gravity_n

        # True specific force at sensor (body frame)
        f_true_b = a_cg_b + a_euler + a_centripetal - g_b

        # Apply scale factors
        f_scaled = self.scale_factors * f_true_b

        # Bias random walk update
        if self.bias_rw_std > 0.0:
            self.bias += self.bias_rw_std * np.sqrt(dt) * self.rng.standard_normal(3)

        # White noise
        noise = np.zeros(3)
        if self.noise_std > 0.0:
            noise = self.noise_std * self.rng.standard_normal(3)

        # Sensor frame output
        f_meas_s = self.R_sb @ (f_scaled + self.bias + noise)

        return f_meas_s


class Gyroscope:
    def __init__(
        self,
        R_sb=np.eye(3),
        bias=np.zeros(3),
        bias_rw_std=0.0,
        noise_std=0.0,
        scale_factors=np.ones(3),
        seed=None,
    ):
        """
        R_sb            : 3x3 DCM, sensor frame relative to body frame
        bias            : initial gyro bias (rad/s)
        bias_rw_std     : bias random walk standard deviation (rad/sqrt(s))
        noise_std       : white noise standard deviation (rad/s)
        scale_factors   : per-axis scale factors
        """

        self.R_sb = np.asarray(R_sb, dtype=float)

        self.bias = np.asarray(bias, dtype=float)
        self.bias_rw_std = bias_rw_std
        self.noise_std = noise_std
        self.scale_factors = np.asarray(scale_factors, dtype=float)

        self.rng = np.random.default_rng(seed)

    def step(
        self,
        omega_body,
        dt,
    ):
        """
        omega_body : true angular rate in body frame (rad/s)
        dt         : time step (s)
        """

        omega_body = np.asarray(omega_body, dtype=float)

        # Bias random walk update
        if self.bias_rw_std > 0.0:
            self.bias += (
                self.bias_rw_std
                * np.sqrt(dt)
                * self.rng.standard_normal(3)
            )

        # White noise
        noise = np.zeros(3)
        if self.noise_std > 0.0:
            noise = (
                self.noise_std
                * self.rng.standard_normal(3)
            )

        # Apply scale factors
        omega_scaled = self.scale_factors * omega_body

        # Sensor frame output
        omega_meas = (
            self.R_sb
            @ (omega_scaled + self.bias + noise)
        )

        return omega_meas
