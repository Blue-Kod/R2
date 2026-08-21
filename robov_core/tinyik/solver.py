"""Solvers (pure numpy)."""

from functools import reduce

import numpy as np

from .component import Joint


class FKSolver(object):
    """A forward kinematics solver."""

    def __init__(self, components):
        """Generate a FK solver from link and joint instances."""
        joint_indexes = [
            i for i, c in enumerate(components) if isinstance(c, Joint)
        ]

        def matrices(angles):
            joints = dict(zip(joint_indexes, angles))
            a = [joints.get(i, None) for i in range(len(components))]
            return [c.matrix(a[i]) for i, c in enumerate(components)]

        self._matrices = matrices
        self.joint_indexes = joint_indexes
        self.components = components

    def solve(self, angles, p=None, index=None):
        """Calculate a position of the end-effector and return it."""
        if p is None:
            p = [0., 0., 0., 1.]
        if index is None:
            index = len(self.components) - 1
        return reduce(
            lambda a, m: np.dot(m, a),
            reversed(self._matrices(angles)[:index + 1]),
            np.array(p)
        )[:3]


class IKSolver(object):
    """An inverse kinematics solver (Levenberg-Marquardt, pure numpy).

    Uses a numerical Jacobian of the end-effector position and damped
    least-squares steps, so no autograd/scipy dependency is needed.
    """

    def __init__(self, fk_solver, optimizer=None):
        """Generate an IK solver from a FK solver instance."""
        self._fk_solver = fk_solver

    def solve(self, angles0, target, tol=1e-10, maxiter=200, lamb=1e-3):
        """Calculate joint angles that reach the target."""
        angles = np.array(angles0, dtype=float)
        target = np.asarray(target, dtype=float)
        prev_err = None
        for _ in range(maxiter):
            ee = self._fk_solver.solve(angles)
            error = ee - target
            err = float(np.linalg.norm(error))
            if prev_err is not None and abs(prev_err - err) < tol:
                break
            prev_err = err
            jac = self._jacobian(angles)
            step = np.linalg.lstsq(
                jac.T @ jac + lamb * np.eye(jac.shape[1]),
                jac.T @ (-error),
                rcond=None)[0]
            angles = angles + step
        return angles

    def _jacobian(self, angles, eps=1e-7):
        fk = self._fk_solver
        n = len(angles)
        ee0 = fk.solve(angles)
        jac = np.empty((3, n))
        for i in range(n):
            delta = angles.copy()
            delta[i] += eps
            jac[:, i] = (fk.solve(delta) - ee0) / eps
        return jac