import math

import numpy as np
from numpy.typing import NDArray


class NotInWorkSpaceError(Exception):
    __module__ = "builtins"

    def __init__(
        self, link_lens: tuple[float, ...], end_effector_position: NDArray[np.double]
    ):
        super().__init__(
            f"The end effector position {end_effector_position} is not in the "
            f"workspace or the 2R planar arm with link lengths a1={link_lens[0]} and "
            f"a2={link_lens[1]}."
        )


def calc_2R_planar_inverse_kinematics(
    link_lens: tuple[float, ...],
    end_effector_position: NDArray[np.double],
    use_elbow_up_soln: bool = True,
) -> NDArray[np.double]:
    a1, a2 = link_lens
    x, y = end_effector_position

    try:
        q2 = math.acos((x * x + y * y - a1 * a1 - a2 * a2) / 2 / a1 / a2)
        if use_elbow_up_soln:
            q2 = -q2
    except ValueError as err:
        raise NotInWorkSpaceError(link_lens, end_effector_position) from err

    if math.isclose(q2, math.pi, abs_tol=1e-6):
        raise RuntimeError("Links are overlapping! (q2 = math.pi)")

    q1 = math.atan2(y, x) - math.atan2(a2 * math.sin(q2), a1 + a2 * math.cos(q2))

    return np.array([q1, q2], dtype=np.double)
