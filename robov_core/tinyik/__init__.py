"""A simple and naive inverse kinematics solver (pure numpy).

Stripped-down fork of tinyik: no autograd, no scipy, no visualizer.
"""

from .core import Actuator
from .component import Link, Joint
from .solver import FKSolver, IKSolver


__all__ = (
    'Actuator',
    'Link', 'Joint',
    'FKSolver', 'IKSolver',
)