import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydrake.systems.framework import LeafSystem, InputPort, BasicVector, Context


class AnalyticalJacobianTransposeGain(LeafSystem):
    def __init__(
        self,
        link_lens: tuple[float, ...],
        calc_analytical_jacobian: Callable[
            [float, float, float, float], NDArray[np.double]
        ],
    ):
        super().__init__()

        self.link_lens = link_lens
        self.num_dofs = len(link_lens)
        assert self.num_dofs == 2

        self.calc_analytical_jacobian = calc_analytical_jacobian

        # `q` (joint positions) is input port 0
        self.DeclareVectorInputPort(name="q", size=self.num_dofs)

        # `f_c` (operational space control action) is input port 1
        self.DeclareVectorInputPort(name="f_c", size=self.num_dofs)

        # `tau` (control torques) is the only output port
        self.DeclareVectorOutputPort(
            name="tau",
            size=self.num_dofs,
            calc=self.apply_analytical_jacobian_transpose,
        )

    def get_q_input_port(self) -> InputPort:
        return self.get_input_port(0)

    def get_f_c_input_port(self) -> InputPort:
        return self.get_input_port(1)

    def apply_analytical_jacobian_transpose(
        self, context: Context, output: BasicVector
    ):
        q: Any = self.get_q_input_port().Eval(context)
        operational_space_control_action = self.get_f_c_input_port().Eval(context)
        J_A = self.calc_analytical_jacobian(*q, *self.link_lens)

        output.set_value(J_A.T @ operational_space_control_action)


class AnalyticalJacobianGain(LeafSystem):
    def __init__(
        self,
        link_lens: tuple[float, ...],
        calc_analytical_jacobian: Callable[
            [float, float, float, float], NDArray[np.double]
        ],
    ):
        super().__init__()

        self.link_lens = link_lens
        self.num_dofs = len(link_lens)
        assert self.num_dofs == 2

        self.calc_analytical_jacobian = calc_analytical_jacobian

        # `q` (joint positions) is input port 0
        self.DeclareVectorInputPort(name="q", size=self.num_dofs)

        # `qdot` (joint velocities) is input port 1
        self.DeclareVectorInputPort(name="qdot", size=self.num_dofs)

        # `xdot` (operational space velocities) is the only output port
        self.DeclareVectorOutputPort(
            name="xdot",
            size=self.num_dofs,
            calc=self.apply_analytical_jacobian,
        )

    def get_q_input_port(self) -> InputPort:
        return self.get_input_port(0)

    def get_qdot_input_port(self) -> InputPort:
        return self.get_input_port(1)

    def apply_analytical_jacobian(self, context: Context, output: BasicVector):
        q: Any = self.get_q_input_port().Eval(context)
        qdot = self.get_qdot_input_port().Eval(context)
        J_A = self.calc_analytical_jacobian(*q, *self.link_lens)

        output.set_value(J_A @ qdot)


class GeometricJacobianTransposeGain(LeafSystem):
    def __init__(
        self,
        link_lens: tuple[float, ...],
    ):
        super().__init__()

        self.link_lens = link_lens
        self.num_dofs = len(link_lens)
        assert self.num_dofs == 2

        # `q` (joint positions) is input port 0
        self.DeclareVectorInputPort(name="q", size=self.num_dofs)

        # `f_e` (end-effector force) is input port 1
        self.DeclareVectorInputPort(name="f_e", size=self.num_dofs)

        # `tau_e` (joint torques due to end-effector force) is the only output port
        self.DeclareVectorOutputPort(
            name="tau_e",
            size=self.num_dofs,
            calc=self.apply_geometric_transpose_jacobian,
        )

    def get_q_input_port(self) -> InputPort:
        return self.get_input_port(0)

    def get_f_e_input_port(self) -> InputPort:
        return self.get_input_port(1)

    def apply_geometric_transpose_jacobian(self, context: Context, output: BasicVector):
        q: Any = self.get_q_input_port().Eval(context)
        f_e = self.get_f_e_input_port().Eval(context)
        J = self.calc_geometric_jacobian(*q)

        output.set_value(J.T @ f_e)

    def calc_geometric_jacobian(self, q1: float, q2: float) -> NDArray[np.double]:
        a1, a2 = self.link_lens
        J = np.empty((2, 2), dtype=np.double)

        q_12 = q1 + q2
        J[0, 1] = -a2 * math.sin(q_12)
        J[0, 0] = -a1 * math.sin(q1) + J[0, 1]
        J[1, 1] = a2 * math.cos(q_12)
        J[1, 0] = a1 * math.cos(q1) + J[1, 1]

        return J
