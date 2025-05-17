import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydrake.common import RandomDistribution
from pydrake.systems.framework import (
    LeafSystem,
    BasicVector,
    Context,
    InputPort,
    Diagram,
    DiagramBuilder,
    OutputPort,
)
from pydrake.systems.primitives import MatrixGain, PassThrough, RandomSource

from mechae263C_helpers.drake import LinearCombination
from mechae263C_helpers.hw5.arm import Arm, CoriolisCentripetal, Gravity
from mechae263C_helpers.hw5.jacobian_gains import (
    GeometricJacobianTransposeGain,
    AnalyticalJacobianGain,
)


class OuterLoopController(LeafSystem):
    def __init__(
        self,
        link_lens: tuple[float, float],
        calc_analytical_jacobian: Callable[
            [float, float, float, float], NDArray[np.double]
        ],
    ):
        super().__init__()
        self.link_lens = link_lens
        self.calc_analytical_jacobian = calc_analytical_jacobian
        self.num_dofs = len(self.link_lens)

        self.DeclareVectorInputPort("q", size=self.num_dofs)
        self.DeclareVectorInputPort("qdot", size=self.num_dofs)
        self.DeclareVectorInputPort("u", size=self.num_dofs)
        self.DeclareVectorOutputPort(
            "y", size=self.num_dofs, calc=self.apply_analytical_jacobian_inv
        )

    def apply_analytical_jacobian_inv(self, context: Context, output: BasicVector):
        q: Any = self.get_q_input_port().Eval(context)
        qdot: Any = self.get_qdot_input_port().Eval(context)
        u: Any = self.get_u_input_port().Eval(context)

        J_Ainv = np.linalg.inv(self.calc_analytical_jacobian(*q, *self.link_lens))
        J_Adot = self.calc_J_Adot(q, qdot)

        outer_loop_control_action = J_Ainv @ (u - J_Adot @ qdot)  # = y
        output.set_value(outer_loop_control_action)

    def get_q_input_port(self) -> InputPort:
        return self.get_input_port(0)

    def get_qdot_input_port(self) -> InputPort:
        return self.get_input_port(1)

    def get_u_input_port(self) -> InputPort:
        return self.get_input_port(2)

    def calc_J_Adot(
        self, q: NDArray[np.double], qdot: NDArray[np.double]
    ) -> NDArray[np.double]:
        s1, c1 = math.sin(q[0]), math.cos(q[0])
        s12, c12 = math.sin(q[0] + q[1]), math.cos(q[0] + q[1])
        a1, a2 = self.link_lens

        J_Adot = np.empty((2, 2), dtype=np.double)
        J_Adot[:, 1] = (qdot[0] + qdot[1]) * a2 * -np.array([c12, s12])
        J_Adot[:, 0] = qdot[0] * a1 * -np.array([c1, s1]) + J_Adot[:, 1]

        return J_Adot


class InertiaMatrix(LeafSystem):
    def __init__(
        self,
        link_lens: tuple[float, float],
        dyn_params: NDArray[np.double],
        transmission_ratio2: float,
    ):
        super().__init__()
        self.link_lens = link_lens
        self.num_dofs = len(self.link_lens)
        self.dyn_params = dyn_params
        self.transmission_ratio2 = transmission_ratio2

        self.DeclareVectorInputPort("q", size=self.num_dofs)
        self.DeclareVectorInputPort("y", size=self.num_dofs)
        self.DeclareVectorOutputPort(
            "B(q) @ y", size=self.num_dofs, calc=self.apply_inertia_matrix
        )

    def apply_inertia_matrix(self, context: Context, output: BasicVector):
        q: Any = self.get_q_input_port().Eval(context)
        y: Any = self.get_y_input_port().Eval(context)

        B = self.calc_inertia_mat(math.cos(q[1]))

        output.set_value(B @ y)

    def get_q_input_port(self) -> InputPort:
        return self.get_input_port(0)

    def get_y_input_port(self) -> InputPort:
        return self.get_input_port(1)

    def calc_inertia_mat(self, cos_q2: float):
        inertia_mat = np.zeros((2, 2), dtype=np.double)

        inertia_mat[0, 0] = (
            self.link_lens[0] * self.dyn_params[0]
            + self.dyn_params[1]
            + (self.link_lens[1] + 2 * self.link_lens[0] * cos_q2) * self.dyn_params[2]
            + self.dyn_params[3]
        )

        inertia_mat[0, 1] = inertia_mat[1, 0] = (
            (self.link_lens[1] + self.link_lens[0] * cos_q2) * self.dyn_params[2]
            + self.dyn_params[3]
            + self.transmission_ratio2 * self.dyn_params[4]
        )

        inertia_mat[1, 1] = (
            self.link_lens[1] * self.dyn_params[2]
            + self.dyn_params[3]
            + self.transmission_ratio2 * self.transmission_ratio2 * self.dyn_params[4]
        )

        return inertia_mat


class DirectKinematics(LeafSystem):
    def __init__(
        self,
        link_lens: tuple[float, ...],
        calc_direct_kinematics: Callable[
            [float, float, float, float], NDArray[np.double]
        ],
    ):
        super().__init__()

        self.link_lens = link_lens
        self.num_dofs = len(link_lens)
        assert self.num_dofs == 2
        self._calc_direct_kinematics = calc_direct_kinematics

        self.DeclareVectorInputPort(name="q", size=self.num_dofs)
        self.DeclareVectorOutputPort(
            name="o_e", size=self.num_dofs, calc=self.calc_direct_kinematics
        )

    def get_q_input_port(self) -> InputPort:
        return self.get_input_port()

    def calc_direct_kinematics(self, context: Context, output: BasicVector):
        q = self.get_input_port().Eval(context)

        x_e: Any = self._calc_direct_kinematics(*q, *self.link_lens)

        output.set_value(x_e)


class NonlinearCompensation(Diagram):
    def __init__(
        self,
        link_lens: tuple[float, ...],
        dyn_params: NDArray[np.double],
        F_v: NDArray[np.double],
    ):
        super().__init__()
        self.link_lens = link_lens
        self.num_dofs = len(link_lens)
        self.dyn_params = dyn_params
        self.F_v = F_v

        builder = DiagramBuilder()
        self.coriolis_centripetal: CoriolisCentripetal = builder.AddNamedSystem(
            "C(q, qdot) @ qdot",
            CoriolisCentripetal(k3=float(-self.link_lens[0] * self.dyn_params[2])),
        )
        self.gravity: Gravity = builder.AddNamedSystem("g(q)", Gravity(self.dyn_params))
        self.viscous_friction: MatrixGain = builder.AddNamedSystem(
            "F_v @ qdot", MatrixGain(self.F_v)
        )
        self.total_nonlinear_torque = builder.AddNamedSystem(
            "n(q, qdot)", LinearCombination(input_coeffs=(1, 1, 1), input_shapes=(2,))
        )
        self.q: PassThrough = builder.AddNamedSystem("q", PassThrough(vector_size=2))
        self.qdot: PassThrough = builder.AddNamedSystem(
            "qdot", PassThrough(vector_size=2)
        )

        # Export Inputs
        builder.ExportInput(name="q", input=self.q.get_input_port())
        builder.ExportInput(name="qdot", input=self.qdot.get_input_port())

        # Export Output
        builder.ExportOutput(
            name="n(q, qdot)", output=self.total_nonlinear_torque.get_output_port()
        )

        # Connect systems
        # Coriolis and Centripetal
        builder.Connect(
            self.q.get_output_port(), self.coriolis_centripetal.get_q_input_port()
        )
        builder.Connect(
            self.qdot.get_output_port(), self.coriolis_centripetal.get_qdot_input_port()
        )
        builder.Connect(
            self.coriolis_centripetal.get_output_port(),
            self.total_nonlinear_torque.get_input_port(0),
        )

        # Gravity
        builder.Connect(self.q.get_output_port(), self.gravity.get_q_input_port())
        builder.Connect(
            self.gravity.get_output_port(),
            self.total_nonlinear_torque.get_input_port(1),
        )

        # Viscous Friction
        builder.Connect(
            self.qdot.get_output_port(), self.viscous_friction.get_input_port()
        )
        builder.Connect(
            self.viscous_friction.get_output_port(),
            self.total_nonlinear_torque.get_input_port(2),
        )

        builder.BuildInto(self)

    def get_q_input_port(self) -> InputPort:
        return self.get_input_port(0)

    def get_qdot_input_port(self) -> InputPort:
        return self.get_input_port(1)


class OperationalSpaceDecoupledArm(Diagram):
    def __init__(
        self,
        link_lens: tuple[float, float],
        calc_direct_kinematics: Callable[
            [float, float, float, float], NDArray[np.double]
        ],
        calc_analytical_jacobian: Callable[
            [float, float, float, float], NDArray[np.double]
        ],
    ):
        super().__init__()
        self.link_lens = link_lens
        self.num_dofs = len(self.link_lens)
        assert self.num_dofs == 2

        self.calc_direct_kinematics = calc_direct_kinematics
        self.calc_analytical_jacobian = calc_analytical_jacobian

        builder = DiagramBuilder()
        self.outer_loop_controller: OuterLoopController = builder.AddNamedSystem(
            "y", OuterLoopController(self.link_lens, self.calc_analytical_jacobian)
        )
        self.arm: Arm = builder.AddNamedSystem("arm", Arm())
        self.inertia_matrix: InertiaMatrix = builder.AddNamedSystem(
            "B(q)",
            InertiaMatrix(
                self.link_lens,
                self.arm.dyn_params,
                self.arm.motors[1].transmission_ratio,
            ),
        )
        self.direct_kinematics: DirectKinematics = builder.AddNamedSystem(
            "k(q)", DirectKinematics(self.link_lens, self.calc_direct_kinematics)
        )
        self.geometric_jacobian_transpose_gain: GeometricJacobianTransposeGain = (
            builder.AddNamedSystem(
                "J.T",
                GeometricJacobianTransposeGain(link_lens),
            )
        )
        self.analytical_jacobian_gain: AnalyticalJacobianGain = builder.AddNamedSystem(
            "J_A", AnalyticalJacobianGain(link_lens, self.calc_analytical_jacobian)
        )
        self.net_control_torque: LinearCombination = builder.AddNamedSystem(
            "control_torques",
            LinearCombination(input_coeffs=(1, 1, 1), input_shapes=(2,)),
        )
        self.nonlinear_compensation: NonlinearCompensation = builder.AddNamedSystem(
            "n(q, qdot)",
            NonlinearCompensation(link_lens, self.arm.dyn_params, self.arm.F_v),
        )

        q_feedback_noise = builder.AddNamedSystem(
            "q_feedback_noise",
            RandomSource(
                RandomDistribution.kGaussian, num_outputs=2, sampling_interval_sec=1e-3
            ),
        )
        qdot_feedback_noise = builder.AddNamedSystem(
            "qdot_feedback_noise",
            RandomSource(
                RandomDistribution.kGaussian, num_outputs=2, sampling_interval_sec=1e-3
            ),
        )
        feedback_noise_stddev = 0.1
        q_noisy: LinearCombination = builder.AddNamedSystem(
            "q_noisy",
            LinearCombination(
                input_coeffs=(feedback_noise_stddev, 1), input_shapes=(2,)
            ),
        )
        qdot_noisy: LinearCombination = builder.AddNamedSystem(
            "qdot_noisy",
            LinearCombination(
                input_coeffs=(feedback_noise_stddev, 1), input_shapes=(2,)
            ),
        )

        # Export Inputs
        builder.ExportInput(
            name="u", input=self.outer_loop_controller.get_u_input_port()
        )
        builder.ExportInput(
            name="f_e",
            input=self.geometric_jacobian_transpose_gain.get_f_e_input_port(),
        )

        # Export Outputs
        builder.ExportOutput(
            name="o_e", output=self.direct_kinematics.get_output_port()
        )
        builder.ExportOutput(
            name="odot_e", output=self.analytical_jacobian_gain.get_output_port()
        )

        # Connect systems
        builder.Connect(
            self.inertia_matrix.get_output_port(),
            self.net_control_torque.get_input_port(0),
        )
        builder.Connect(
            self.nonlinear_compensation.get_output_port(),
            self.net_control_torque.get_input_port(1),
        )
        builder.Connect(
            self.geometric_jacobian_transpose_gain.get_output_port(),
            self.net_control_torque.get_input_port(2),
        )
        builder.Connect(
            self.net_control_torque.get_output_port(), self.arm.get_input_port()
        )

        q, qdot = self.arm.get_q_output_port(), self.arm.get_qdot_output_port()
        builder.Connect(q_feedback_noise.get_output_port(), q_noisy.get_input_port(0))
        builder.Connect(q, q_noisy.get_input_port(1))
        # builder.Connect(
        #     qdot_feedback_noise.get_output_port(), qdot_noisy.get_input_port(0)
        # )
        # builder.Connect(qdot, qdot_noisy.get_input_port(1))
        q_measured = q
        qdot_measured = qdot
        builder.Connect(q_measured, self.outer_loop_controller.get_q_input_port())
        builder.Connect(q_measured, self.nonlinear_compensation.get_q_input_port())
        builder.Connect(
            q_measured, self.geometric_jacobian_transpose_gain.get_q_input_port()
        )
        builder.Connect(q_measured, self.direct_kinematics.get_q_input_port())
        builder.Connect(q_measured, self.inertia_matrix.get_q_input_port())
        builder.Connect(q_measured, self.analytical_jacobian_gain.get_q_input_port())
        builder.Connect(qdot_measured, self.outer_loop_controller.get_qdot_input_port())
        builder.Connect(qdot_measured, self.nonlinear_compensation.get_qdot_input_port())
        builder.Connect(
            qdot_measured, self.analytical_jacobian_gain.get_qdot_input_port()
        )

        builder.Connect(
            self.outer_loop_controller.get_output_port(),
            self.inertia_matrix.get_y_input_port(),
        )

        builder.BuildInto(self)

    def get_u_input_port(self) -> InputPort:
        """Returns the input port for the control command"""
        return self.get_input_port(0)

    def get_f_e_input_port(self) -> InputPort:
        """Returns the input port for operational space end-effector force"""
        return self.get_input_port(1)

    def get_o_e_output_port(self) -> OutputPort:
        """Returns the output port for operational space end-effector position"""
        return self.get_output_port(0)

    def get_odot_e_output_port(self) -> OutputPort:
        """Returns the output port for operational space end-effector velocity"""
        return self.get_output_port(1)


class Environment(LeafSystem):
    def __init__(
        self,
        equilibrium_position: NDArray[np.double],
        stiffness: NDArray[np.double],
    ):
        super().__init__()
        self.equilibrium_position = equilibrium_position
        self.stiffness = stiffness

        self.DeclareVectorInputPort(name="o_e", size=2)
        self.DeclareVectorOutputPort(
            name="f_e", size=2, calc=self.calc_end_effector_forces
        )

    def calc_end_effector_forces(self, context: Context, output: BasicVector):
        o_e: Any = self.get_o_e_input_port().Eval(context)
        o_r = self.equilibrium_position
        K = self.stiffness

        f_e = K @ (o_e - o_r)
        output.set_value(f_e)

    def get_o_e_input_port(self) -> InputPort:
        """Returns the input port for operational space end-effector position"""

        return self.get_input_port()

    def get_f_e_output_port(self) -> OutputPort:
        """Returns the output port for operational space end-effector force"""

        return self.get_output_port()
