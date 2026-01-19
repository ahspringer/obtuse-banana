# -*- coding: utf-8 -*-
# Filename: bullets.py

"""
A python script defining bullet objects for use in a ballistics simulation.
Created on 10 JAN 2025
Author: Alex Springer
"""

import numpy as np
from environment import Environment

def twist_rate_convert(twist_rate_per_inch):
    # 1:9" is one twist every 9 inches
    twist_rate_m_t = twist_rate_per_inch * 0.0254
    print(f'A 1:{twist_rate_per_inch}\" twist rate converts to {twist_rate_m_t} meters per turn')
    return twist_rate_m_t  # meters per turn

def euler(roll, pitch, yaw, vel=0):
    """
    Convert roll, pitch, yaw (in radians) and speed to global velocity vector.
    
    :param roll:  rotation around X axis (rad)
    :param pitch: rotation around Y axis (rad)
    :param yaw:   rotation around Z axis (rad)
    :param vel:   scalar exit/muzzle velocity
    
    Returns:
        (vx, vy, vz): global velocity components
    """
    # Rotation matrices for yaw, pitch, roll
    R_yaw = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0 ],
        [ np.sin(yaw),  np.cos(yaw), 0 ],
        [          0,           0,  1 ]
    ])

    R_pitch = np.array([
        [  np.cos(pitch), 0, np.sin(pitch) ],
        [              0, 1,             0 ],
        [ -np.sin(pitch), 0, np.cos(pitch) ]
    ])

    R_roll = np.array([
        [1,           0,            0],
        [0,  np.cos(roll), -np.sin(roll)],
        [0,  np.sin(roll),  np.cos(roll)]
    ])

    # Combine rotations: yaw → pitch → roll
    R = R_yaw @ R_pitch @ R_roll

    # Forward direction in body frame (X pointed forward)
    forward_body = np.array([1.0, 0.0, 0.0])

    # Rotate into global (world) frame
    forward_global = R @ forward_body

    # Scale by speed to get velocity
    vx, vy, vz = vel * forward_global

    return vx, vy, vz

class Bullet:
    def __init__(self, mass, diameter, bc_g7, muzzle_vel, twist_rate, environment, initial_pos=None, initial_orientation=None):
        """
        Docstring for Bullet

        GLOBAL COORDINATE SYSTEM:
            +X = Downrange
            +Y = Right
            +Z = Down
        
        GLOBAL ORIGIN:
            Set to location and orientation of exit of gun's muzzle at simulation start
        
        :param self: self
        :param mass: Mass (g)
        :param diameter: Caliber (mm)
        :param bc_g7: G7 Ballistic Coefficient
        :param muzzle_vel: Muzzle velocity (m/s)
        :param twist_rate: Twist rate in meters per turn (e.g. 1:10" = 0.254m/turn)
        :param environment: Object of <class environment.Environment> defining aerodynamic model
        :param initial_pos: (optional) Bullet originating position in [x, y, z] array (m) 
        :param initial_orientation: (optional) Bullet originating position in [roll, pitch, yaw] relative to GLOBAL ORIGIN (deg) 
        """
        self.m = mass * 1000  # kg
        self.d = diameter  * 1000  # m
        self.area = np.pi * (diameter / 2)**2  # m^2
        self.bc_g7 = bc_g7
        self.env = environment
        self.ground_alt = 0  # Ground clamped to 0 in z-axis

        # Stability / Aerodynamic constants (Estimated for standard spitzer projectiles)
        self.Cx = 0.5 * self.env.density_kg_m3 * self.area # Drag reference area factor
        self.Ix = 0.5 * mass * (diameter/2)**2     # Approx axial moment of inertia
        self.Cl_alpha = 2.0  # Lift slope derivative
        self.Cm_alpha = 4.5  # Overturning moment derivative
        self.C_mag = 0.03    # Magnus coefficient (approximate)
        self.k_spin = 0.00005  # Approximated damping factor
        self.form_factor = (self.m / (self.d**2)) / (self.bc_g7 * 703.07)

        # Initial State Setup
        if initial_pos is None:
            initial_pos = [0, 0, 0]
        if initial_orientation is None:
            initial_orientation = [0, 0, 0]

        # Initial Velocity
        vx, vy, vz = euler(np.deg2rad(initial_orientation[0]), np.deg2rad(initial_orientation[1]), np.deg2rad(initial_orientation[2]), muzzle_vel)
        
        # Initial Spin (rad/s) = Velocity / Twist * 2pi
        spin = (muzzle_vel / twist_rate) * 2 * np.pi
        
        # State Vector: [x, y, z, vx, vy, vz, spin]
        self.state = np.array([initial_pos[0], initial_pos[1], initial_pos[2],
                               vx, vy, vz, spin], dtype=float)

        # Print initial state 
        self.print_state()

        self.history = [self.state.copy()]

        self.status = 0  # 0 = In-flight; 1 = hit target; 2 = hit ground
    
    def get_drag_coefficient(self, mach):
        """
        An improved analytical approximation of the G7 drag curve.
        This model accounts for the sharp rise at Mach 0.8 and the 
        peak slightly past Mach 1.
        """
        # 1. Base G7 Standard Projectile Drag Curve (Standard C_g7)
        if mach <= 0.80:
            # Subsonic: Fairly flat drag
            c_g7 = 0.22
        elif mach <= 1.05:
            # Transonic rise: Cubic-style ramp up
            # At 0.8 -> 0.22 | At 1.05 -> ~0.41
            c_g7 = 0.22 + 0.75 * (mach - 0.8)**2
        elif mach <= 1.5:
            # Transonic peak and early supersonic decay
            # Peaks near Mach 1.1 (~0.42) then starts dropping
            c_g7 = 0.42 - 0.28 * (mach - 1.05)**0.5
        elif mach <= 5.0:
            # Supersonic decay: Classic power law decay
            # Most bullets follow a mach^-0.4 to mach^-0.6 decay here
            c_g7 = 0.35 / (mach**0.55)
        else:
            # Hypersonic plateau
            c_g7 = 0.15

        # 2. Scale by Form Factor (i)
        # The G7 BC is defined as Sectional_Density / Form_Factor
        # Form Factor i = SD / BC
        # For a .338 300gr with BC 0.40, i is roughly 0.95-1.05
        
        # Calculate Sectional Density (lb/in^2 converted to kg/m^2 logic)
        # Or more simply, if the user provided BC_G7:
        # i = (mass_kg / (diameter_m^2)) / (BC_G7_lb_in2 * 703.07)
        
        # If you haven't calculated 'self.form_factor' in __init__, 
        # you can do it once there:
        # self.form_factor = (self.m / (self.d**2)) / (self.bc_g7 * 703.07)
        
        return c_g7 * self.form_factor
    
    def _calculate_forces(self, state=None):
        """
        Docstring for _calculate_forces
        
        :param state: (optional) a state to calculate from for RK4 integration
        """
        # Unpack state
        work_state = state if state is not None else self.state
        pos = work_state[0:3]
        u = work_state[3:6]     # ground velocity (m/s)
        p = work_state[6]       # spin rate (rad/s)

        u_mag = np.linalg.norm(u)
        if u_mag < 1e-5: return np.zeros(7)  # Bullet isn't moving
        u_hat = u / u_mag

        # 1. Air Velocity
        v_air = self.env.relative_air_velocity(u)
        v_air_mag = np.linalg.norm(v_air)
        v_air_hat = v_air / (v_air_mag + 1e-9)

        # 2. Mach Number
        mach = self.env.mach_number(u)

        # 3. Dynamic Pressure
        q = 0.5 * self.env.density_kg_m3 * self.area * v_air_mag**2

        # -- DRAG FORCE --
        Cd = self.get_drag_coefficient(mach)
        F_drag = -0.5 * self.env.density_kg_m3 * self.area * Cd * v_air_mag * v_air

        # -- YAW OF REPOSE --
        g_cross_u = np.cross([0, 0, self.env.g_m_s2], u)
        denom_repose = (self.env.density_kg_m3 * self.area * self.d * (u_mag**4) * self.Cm_alpha) + 1e-9
        alpha_repose_vec = (2 * self.Ix * p * g_cross_u) / denom_repose

        # Lift due to repose
        F_lift_repose = q * self.Cl_alpha * alpha_repose_vec

        # -- MAGNUS EFFECT --
        # Assumption: spin axis aligned with ground-velocity (no wobble or yaw)
        magnus_dir = np.cross(u_hat, v_air_hat)
        spin_param = (p * self.d) / (v_air_mag + 1e-9)  # non-dimensional spin parameter
        F_magnus = q * self.area * self.C_mag * spin_param * magnus_dir

        # -- TOTAL FORCES --
        F_total = F_drag + F_lift_repose + F_magnus + np.dot(self.m, [0, 0, self.env.g_m_s2])

        # -- SPIN DECAY --
        # Simple exponential decay due to skin friction
        dp_dt = -self.k_spin * p * v_air_mag

        # Return derivatives [vx, vy, vz, ax, ay, az, dp_dt]
        accel = F_total / self.m
        return np.concatenate((u, accel, [dp_dt]))
    
    def step (self, dt=0.01, env=None, hit=False):
        """
        Docstring for step

        Performs one RK4 integration step for the bullet object.
        
        :param dt: time step length (default 0.01 seconds)
        :param env: (optional) An updated environment definition
        """
        # If the bullet hit the target, we're done
        if hit:
            self.status = 1
            return

        # Don't waste resources if the bullet is not flying
        if self.status != 0:
            return
        
        if env is not None:
            # Update environment if required
            self.env = env
        
        # Stop calculations if bullet hits ground (z < ground)
        if self.state[2] > env.ground:
            self.state = [self.state[0], self.state[1], self.state[2], 0, 0, 0, 0]
            self.history.append(self.state.copy())
            self.status = 2  # Hit ground
            return

        state = self.state
        
        k1 = self._calculate_forces(state)
        k2 = self._calculate_forces(state + k1 * 0.5 * dt)
        k3 = self._calculate_forces(state + k2 * 0.5 * dt)
        k4 = self._calculate_forces(state + k3 * dt)

        # Update object state
        self.state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Update history
        self.history.append(self.state.copy())

    def print_state(self):
        print(f'BULLET STATE:\n\tx={self.state[0]}\n\ty={self.state[1]}\n\tz={self.state[2]}\n\tvx={self.state[3]}\n\tvy={self.state[4]}\n\tvz={self.state[5]}\n\tspin={self.state[6]}')
       