# -*- coding: utf-8 -*-
# Filename: environment.py

"""
A python script defining an environment object for use in a ballistics simulation environment.
Created on 10 JAN 2025
Author: Alex Springer
"""

import numpy as np

class Environment:
    """
    Aerodynamic environment model based on altitude and temperature condition.

    Coordinate-agnostic; provides scalar atmospheric properties for use
    in external force models (drag, Magnus, etc.).

    Assumptions:
    - ISA troposphere
    - Dry air
    - Ideal gas
    """

    # Physical constants
    g = 9.80665            # m/s^2
    R = 287.058            # J/(kg·K) specific gas constant for dry air
    gamma = 1.4            # ratio of specific heats
    P0 = 101325.0          # Pa
    T0 = 288.15            # K (15°C)
    rho0 = 1.225           # kg/m^3
    L = -0.0065            # K/m temperature lapse rate (troposphere)

    def __init__(self, altitude_m: float = 0.0, ground: float=0.0, temperature: str = "standard", wind_m_s: np.ndarray | None = None):
        """
        Parameters
        ----------
        altitude_m : float
            Geometric altitude above sea level [meters]
        ground : float
            Geometric altitude of ground above sea level [meters]
        temperature : str
            One of ['cold', 'standard', 'hot']
        wind_m_s : array-like, optional
            Static wind vector in global frame [wx, wy, wz] (m/s)
        """

        self.altitude_m = altitude_m
        self.temperature_condition = temperature.lower()
        self.ground = ground

        self.wind_m_s = np.zeros(3) if wind_m_s is None else np.asarray(wind_m_s, dtype=float)

        self._validate_inputs()

        # Compute atmospheric state
        self.temperature_K = self._compute_temperature()
        self.pressure_Pa = self._compute_pressure()
        self.density_kg_m3 = self._compute_density()
        self.speed_of_sound_m_s = self._compute_speed_of_sound()

        # misc
        self.g_m_s2 = self.g  # m/s^2

    def _validate_inputs(self):
        if self.temperature_condition not in ["cold", "standard", "hot"]:
            raise ValueError("temperature must be 'cold', 'standard', or 'hot'")

        if self.altitude_m < -500:
            raise ValueError("Altitude below -500 m not supported")
        
        if self.wind_m_s.shape != (3,):
            raise ValueError("wind_m_s must be a 3-element vector [wx, wy, wz]")

    # -- Atmospherics --

    def _compute_temperature(self) -> float:
        """
        Temperature with ISA lapse rate and offset.
        """
        T_isa = self.T0 + self.L * self.altitude_m

        # Temperature offsets (engineering typical)
        if self.temperature_condition == "cold":
            return T_isa - 10.0
        elif self.temperature_condition == "hot":
            return T_isa + 15.0
        else:
            return T_isa

    def _compute_pressure(self) -> float:
        """
        Pressure from hydrostatic equation (ISA troposphere).
        """
        T0 = self.T0
        L = self.L
        h = self.altitude_m

        return self.P0 * (1 + L * h / T0) ** (-self.g / (self.R * L))

    def _compute_density(self) -> float:
        """
        Density from ideal gas law.
        """
        return self.pressure_Pa / (self.R * self.temperature_K)
    
    def _compute_speed_of_sound(self) -> float:
        """
        Speed of sound of ideal gas (air)
        """
        return np.sqrt(self.gamma * self.R * self.temperature_K)

    # -- Wind utilities --

    def relative_air_velocity(self, body_velocity_m_s: np.ndarray) -> np.ndarray:
        """
        Computes air-relative velocity vector.

        v_air = v_body - v_wind
        """
        body_velocity_m_s = np.asarray(body_velocity_m_s, dtype=float)

        if body_velocity_m_s.shape != (3,):
            raise ValueError("body_velocity_m_s must be a 3-element vector")

        return body_velocity_m_s - self.wind_m_s

    def mach_number(self, body_velocity_m_s: np.ndarray) -> float:
        """
        Mach number based on air-relative velocity magnitude.
        """
        v_rel = self.relative_air_velocity(body_velocity_m_s)
        return np.linalg.norm(v_rel) / self.speed_of_sound_m_s

    def summary(self) -> dict:
        """
        Returns a dictionary of environment properties.
        """
        return {
            "altitude_m": self.altitude_m,
            "temperature_condition": self.temperature_condition,
            "temperature_K": self.temperature_K,
            "pressure_Pa": self.pressure_Pa,
            "density_kg_m3": self.density_kg_m3,
            "speed_of_sound_m_s": self.speed_of_sound_m_s,
            "wind_m_s": self.wind_m_s.copy()
        }

class Target:
    def __init__(self, x=1000.0, y=0, z=0, radius=0.5):
        """
        Docstring for Target
        
        :param self: self
        :param x: Target location in X direction (m)
        :param y: Target location in Y direction (m)
        :param z: Target location in Z direction (m)
        :param radius: Target radius (m)
        """
        self.position = np.array((x, y, z))
        self.radius = radius

    def check_interaction(self, bullet):
        """
        Checks if the bullet line segment intersects the target plane.
        Returns:
            status: 0 (Not reached), 1 (Hit), -1 (Miss/Passed)
            dy: Y displacement from center at crossing
            dz: Z displacement from center at crossing
        """
        target_x = self.position[0]
        curr_x = bullet.state[0]
        prev_x = bullet.history[-1][0]
        status = 0  # Bullet has not yet reached target
        dy = 0
        dz = 0

        # Check if bullet passed target plane
        if prev_x < target_x and curr_x >= target_x:
            
            # Linear Interpolation to find XYZ of intercept
            if (curr_x - prev_x) == 0:
                fraction = 0
            else:
                fraction = (target_x - prev_x) / (curr_x - prev_x)
            
            impact_y = bullet.history[-1][1] + fraction * (bullet.state[1] - bullet.history[-1][1])
            impact_z = bullet.history[-1][2] + fraction * (bullet.state[2] - bullet.history[-1][2])

            dy = impact_y - self.position[1]
            dz = impact_z - self.position[2]

            miss_dist = np.sqrt(dy**2 + dz**2)

            if miss_dist <= self.radius:
                status = 1  # HIT
            else:
                status = -1  # MISS

        return status, dy, dz
