"""Utility helpers package.

Most client code should import conveniently from here rather than diving into
individual submodules.  The quaternion/euler functions live in
``angles.py`` and CSV utilities are in ``csv_convert.py``.

Example::

    from src.util import dcm_from_euler, quaternion_multiply, convert_to_csv

"""

# re-export commonly used symbols from submodules
from .angles import (
    dcm_from_euler,
    euler_from_dcm,
    quat_from_dcm,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_norm,
    quaternion_normalize,
    quaternion_to_rotation_matrix,
)
from .csv_convert import convert_to_csv

__all__ = [
    # angles
    "dcm_from_euler",
    "euler_from_dcm",
    "quat_from_dcm",
    "quaternion_multiply",
    "quaternion_conjugate",
    "quaternion_norm",
    "quaternion_normalize",
    "quaternion_to_rotation_matrix",
    # csv
    "convert_to_csv",
]