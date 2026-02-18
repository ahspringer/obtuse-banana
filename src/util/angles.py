import numpy as np
from typing import Tuple

"""HELPER FUNCTIONS FOR DCM GENERATION"""

def dcm_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Generate DCM from Euler angles (rad).
    Rotation order: yaw (Z) -> pitch (Y) -> roll (X)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    dcm = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])
    return dcm

def euler_from_dcm(dcm: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract Euler angles (roll, pitch, yaw) from DCM.
    """
    pitch = np.arcsin(-dcm[2, 0])
    roll = np.arctan2(dcm[2, 1], dcm[2, 2])
    yaw = np.arctan2(dcm[1, 0], dcm[0, 0])
    return roll, pitch, yaw

def quat_from_dcm(dcm: np.ndarray) -> np.ndarray:
    """
    Extract quaternion (w, x, y, z) from DCM using Shepperd's method.
    """
    trace = dcm[0, 0] + dcm[1, 1] + dcm[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (dcm[2, 1] - dcm[1, 2]) * s
        qy = (dcm[0, 2] - dcm[2, 0]) * s
        qz = (dcm[1, 0] - dcm[0, 1]) * s
    elif dcm[0, 0] > dcm[1, 1] and dcm[0, 0] > dcm[2, 2]:
        s = 2.0 * np.sqrt(1.0 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2])
        qw = (dcm[2, 1] - dcm[1, 2]) / s
        qx = 0.25 * s
        qy = (dcm[0, 1] + dcm[1, 0]) / s
        qz = (dcm[0, 2] + dcm[2, 0]) / s
    elif dcm[1, 1] > dcm[2, 2]:
        s = 2.0 * np.sqrt(1.0 + dcm[1, 1] - dcm[0, 0] - dcm[2, 2])
        qw = (dcm[0, 2] - dcm[2, 0]) / s
        qx = (dcm[0, 1] + dcm[1, 0]) / s
        qy = 0.25 * s
        qz = (dcm[1, 2] + dcm[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + dcm[2, 2] - dcm[0, 0] - dcm[1, 1])
        qw = (dcm[1, 0] - dcm[0, 1]) / s
        qx = (dcm[0, 2] + dcm[2, 0]) / s
        qy = (dcm[1, 2] + dcm[2, 1]) / s
        qz = 0.25 * s
    
    return np.array([qw, qx, qy, qz])

""" QUATERNION UTILITIES """

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication q1 * q2 (q = [q0, q1, q2, q3] scalar-first)."""
    q0_1, q1_1, q2_1, q3_1 = q1[0], q1[1], q1[2], q1[3]
    q0_2, q1_2, q2_2, q3_2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        q0_1*q0_2 - q1_1*q1_2 - q2_1*q2_2 - q3_1*q3_2,
        q0_1*q1_2 + q1_1*q0_2 + q2_1*q3_2 - q3_1*q2_2,
        q0_1*q2_2 - q1_1*q3_2 + q2_1*q0_2 + q3_1*q1_2,
        q0_1*q3_2 + q1_1*q2_2 - q2_1*q1_2 + q3_1*q0_2
    ])

def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate: [q0, -q1, -q2, -q3]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_norm(q: np.ndarray) -> float:
    """Euclidean norm of quaternion."""
    return np.linalg.norm(q)

def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit norm."""
    return q / quaternion_norm(q)

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (q0, q1, q2, q3) to 3x3 rotation matrix (body->NED)."""
    q = quaternion_normalize(q)
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [    2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2),     2*(q2*q3 - q0*q1)],
        [    2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])