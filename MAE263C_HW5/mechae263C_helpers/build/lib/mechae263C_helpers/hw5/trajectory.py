import numpy as np
from numpy.typing import NDArray
from pydrake.systems.framework import LeafSystem, BasicVector, Context
from scipy.interpolate import CubicSpline


class PrePlannedTrajectorySource(LeafSystem):
    def __init__(
        self,
        name: str,
        num_joints: int,
        times: NDArray[np.double],
        values: NDArray[np.double],
    ):
        super().__init__()

        self.num_joints = max(round(abs(num_joints)), 1)
        self.times = np.asarray(times, dtype=np.double)
        self.values = np.asarray(values, dtype=np.double)

        self.ix: int = 0
        self.DeclareVectorOutputPort(
            name=name, size=self.num_joints, calc=self.calc_output
        )

    def calc_output(self, context: Context, output: BasicVector):
        t = context.get_time()

        if t >= self.times[self.ix]:
            self.ix += 1

        self.ix = min(self.ix, len(self.times) - 1)

        output.set_value(self.values[:, self.ix])  # noqaq


def eval_cubic_spline_traj(
    times: NDArray[np.double],
    waypoint_times: NDArray[np.double],
    waypoints: NDArray[np.double],
) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double]]:
    times = np.asarray(times, dtype=np.double)
    waypoint_times = np.asarray(waypoint_times, dtype=np.double)
    waypoints = np.asarray(waypoints, dtype=np.double)

    spl = CubicSpline(x=waypoint_times, y=waypoints, axis=1, bc_type="clamped")

    return spl(times), spl(times, 1), spl(times, 2)


def eval_trapz_traj(
    times: NDArray[np.double],
    max_velocity: float,
    final_time: float,
    initial_position: NDArray[np.double],
    final_position: NDArray[np.double],
) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double]]:
    times = np.asarray(times, dtype=np.double)

    total_traj_dist = np.linalg.norm(final_position - initial_position)
    normalized_max_velocity = max_velocity / total_traj_dist

    ix = np.argmin(np.abs(final_time - times))
    normalized_times = np.asarray(times[:ix], dtype=np.double)
    normalized_times /= normalized_times[-1]

    sdot_max = abs(normalized_max_velocity)
    t_final = 1
    t_ramp = t_final - 1 / sdot_max
    sddot_max = sdot_max / t_ramp
    start_segment_mask = normalized_times <= t_ramp
    middle_segment_mask = (t_ramp < normalized_times) & (
        normalized_times <= t_final - t_ramp
    )
    end_segment_mask = (t_final - t_ramp < normalized_times) & (
        normalized_times <= t_final
    )

    s = np.zeros_like(normalized_times)
    sdot = np.zeros_like(normalized_times)
    sddot = np.zeros_like(normalized_times)

    # Start Segment
    t = normalized_times[start_segment_mask]
    s[start_segment_mask] = sddot_max * t * t / 2
    sdot[start_segment_mask] = sddot_max * t
    sddot[start_segment_mask] = sddot_max

    # Middle Segment
    t = normalized_times[middle_segment_mask]
    s[middle_segment_mask] = sdot_max * (t - t_ramp / 2)
    sdot[middle_segment_mask] = sdot_max

    # End Segment
    t = normalized_times[end_segment_mask]
    s[end_segment_mask] = 1 - sddot_max * np.square(t - t_final) / 2
    sdot[end_segment_mask] = sddot_max * (t_final - t)
    sddot[end_segment_mask] = -sddot_max

    position_traj = np.concatenate(
        (
            initial_position[:, None]
            + s[None, :] * (final_position[:, None] - initial_position[:, None]),
            np.tile(final_position[:, None], (1, times[ix:].shape[0])),
        ),
        axis=1,
    )
    velocity_traj = np.concatenate(
        (
            sdot[None, :] * (final_position[:, None] - initial_position[:, None]),
            np.zeros((final_position.shape[0], times[ix:].shape[0])),
        ),
        axis=1,
    )
    acceleration_traj = np.concatenate(
        (
            sddot[None, :] * (final_position[:, None] - initial_position[:, None]),
            np.zeros((final_position.shape[0], times[ix:].shape[0])),
        ),
        axis=1,
    )

    return position_traj, velocity_traj, acceleration_traj
