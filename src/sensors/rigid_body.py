# -*- coding: utf-8 -*-
# Filename: rigid_body.py

"""
A python script defining rigid body motion for a fixed measurement point.
Created on 27 DEC 2025
Author: Alex Springer
"""

import numpy as np


def skew_symmetric(vector):
    """Return skew-symmetric matrix for cross product."""
    x, y, z = vector
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


class RigidBodyMotion:
    def __init__(
        self,
        pivot_position=np.zeros(3),
        pivot_velocity=np.zeros(3),
        pivot_acceleration=np.zeros(3),
        orientation=np.eye(3),
        angular_rate=np.zeros(3),
        angular_acceleration=np.zeros(3),
    ):
        """
        pivot_position         : pivot position in navigation frame
        pivot_velocity         : pivot velocity in navigation frame
        pivot_acceleration     : pivot acceleration in navigation frame
        orientation            : DCM body-to-nav
        angular_rate            : angular rate in body frame
        angular_acceleration    : angular acceleration in body frame
        """

        self.pivot_position = np.asarray(pivot_position, dtype=float)
        self.pivot_velocity = np.asarray(pivot_velocity, dtype=float)
        self.pivot_acceleration = np.asarray(
            pivot_acceleration,
            dtype=float,
        )

        self.orientation = np.asarray(orientation, dtype=float)

        self.angular_rate = np.asarray(angular_rate, dtype=float)
        self.angular_acceleration = np.asarray(
            angular_acceleration,
            dtype=float,
        )

    def step(
        self,
        dt,
        new_pivot_acceleration=None,
        new_angular_rate=None,
        new_angular_acceleration=None,
        new_pivot_position=None,
    ):
        """
        Advance rigid body state by one time step.

        dt                       : time step (s)
        new_pivot_acceleration   : optional pivot acceleration update
        new_angular_rate         : optional angular rate override
        new_angular_acceleration : optional angular acceleration update
        new_pivot_position       : optional pivot position override
        """

        # Update translational quantities
        if new_pivot_acceleration is not None:
            self.pivot_acceleration = np.asarray(
                new_pivot_acceleration,
                dtype=float,
            )

        self.pivot_velocity += self.pivot_acceleration * dt
        self.pivot_position += self.pivot_velocity * dt

        if new_pivot_position is not None:
            self.pivot_position = np.asarray(
                new_pivot_position,
                dtype=float,
            )

        # Update angular quantities
        if new_angular_acceleration is not None:
            self.angular_acceleration = np.asarray(
                new_angular_acceleration,
                dtype=float,
            )

        self.angular_rate += self.angular_acceleration * dt

        if new_angular_rate is not None:
            self.angular_rate = np.asarray(
                new_angular_rate,
                dtype=float,
            )

        # Propagate orientation (first-order integration)
        omega_skew = skew_symmetric(self.angular_rate)
        self.orientation += self.orientation @ omega_skew * dt

        # Re-orthonormalize DCM
        u, _, v = np.linalg.svd(self.orientation)
        self.orientation = u @ v

    def get_state(self):
        """Return current rigid body state."""
        return {
            "pivot_position": self.pivot_position.copy(),
            "pivot_velocity": self.pivot_velocity.copy(),
            "pivot_acceleration": self.pivot_acceleration.copy(),
            "orientation": self.orientation.copy(),
            "angular_rate": self.angular_rate.copy(),
            "angular_acceleration": self.angular_acceleration.copy(),
        }
