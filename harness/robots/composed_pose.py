"""Shared candidate arrival limits; independent of sensing and motion control.

Navigation, refinement, dynamic goals and sim-attachment admission share these
inclusive limits. A successful, stationary arrival inside them needs no further
refinement. Navigation checks yaw only when the requested goal requires it;
attachment always checks the pickup heading.
"""
POSITION_TOLERANCE_M = 0.15
YAW_TOLERANCE_RAD = 0.15
