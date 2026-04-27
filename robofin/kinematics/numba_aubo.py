"""
Numba-based forward kinematics for the Aubo i3H robot.

Joint origin matrices are derived from aubo_i3H.urdf:

  shoulder_joint : xyz=(0, 0, 0.157),    rpy=(0,      0,    pi)
  upperArm_joint : xyz=(0, 0.119, 0),    rpy=(-pi/2, -pi/2, 0)
  foreArm_joint  : xyz=(0.266, 0, 0),    rpy=(-pi,    0,    0)
  wrist1_joint   : xyz=(0.2565, 0, 0),   rpy=(pi,     0,    pi/2)
  wrist2_joint   : xyz=(0, 0.1025, 0),   rpy=(-pi/2,  0,    0)
  wrist3_joint   : xyz=(0, -0.094, 0),   rpy=(pi/2,   0,    0)

All revolute joints rotate about their local z-axis.
"""

import numba
import numpy as np
from geometrout.maths import transform_in_place


@numba.jit(nopython=True, cache=True)
def axis_angle(axis, angle):
    sina = np.sin(angle)
    cosa = np.cos(angle)
    axis = axis / np.linalg.norm(axis)

    M = np.diag(np.array([cosa, cosa, cosa, 1.0]))
    M[:3, :3] += np.outer(axis, axis) * (1.0 - cosa)

    axis = axis * sina
    M[:3, :3] += np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return M


@numba.jit(nopython=True, cache=True)
def aubo_eef_link_fk(prismatic_joint: float, base_pose: np.ndarray) -> np.ndarray:
    """
    Fast Numba FK for the Aubo i3H end-effector links.

    Unlike the Franka gripper, the stock Aubo i3H URDF in this repository
    does not add any fixed hand/finger/tool links after ``wrist3_Link``.
    Therefore the end-effector chain contains only one pose:

        [
            "wrist3_Link",
        ]

    :param prismatic_joint: Unused. Kept only for API compatibility with the
        Franka helper functions.
    :param base_pose: 4x4 pose of ``wrist3_Link`` in the world frame.
    :return: poses shape (1, 4, 4)
    """
    poses = np.zeros((1, 4, 4))
    poses[0, :, :] = base_pose
    return poses


@numba.jit(nopython=True, cache=True)
def aubo_eef_visual_fk(prismatic_joint: float, base_pose: np.ndarray) -> np.ndarray:
    """
    Visual FK for the Aubo i3H end-effector chain.

    In the current Aubo i3H URDF, the only end-effector visual link is
    ``wrist3_Link`` itself, and its visual geometry is defined directly in the
    link frame. So the visual FK is identical to the link FK.

    :param prismatic_joint: Unused. Kept only for API compatibility with the
        Franka helper functions.
    :param base_pose: 4x4 pose of ``wrist3_Link`` in the world frame.
    :return: poses shape (1, 4, 4), in the following order:
        [
            "wrist3_Link",
        ]
    """
    return aubo_eef_link_fk(prismatic_joint, base_pose)


@numba.jit(nopython=True, cache=True)
def aubo_arm_link_fk(cfg: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    """
    Fast Numba FK for the Aubo i3H (6-DOF, no gripper).

    :param cfg: joint angles, shape (6,)
    :param base_pose: 4x4 base transform
    :return: poses shape (7, 4, 4) in the following order:
        [
            "base_link",       # 0
            "shoulder_Link",   # 1
            "upperArm_Link",   # 2
            "foreArm_Link",    # 3
            "wrist1_Link",     # 4
            "wrist2_Link",     # 5
            "wrist3_Link",     # 6
        ]
    """
    joint_axes = np.array(
        [
            [0.0, 0.0, 1.0],  # shoulder_joint
            [0.0, 0.0, 1.0],  # upperArm_joint
            [0.0, 0.0, 1.0],  # foreArm_joint
            [0.0, 0.0, 1.0],  # wrist1_joint
            [0.0, 0.0, 1.0],  # wrist2_joint
            [0.0, 0.0, 1.0],  # wrist3_joint
        ]
    )

    # T = Trans(xyz) * Rot_RPY(roll, pitch, yaw),  RPY = Rz(yaw)*Ry(pitch)*Rx(roll)
    joint_origins = np.array(
        [
            [  # shoulder_joint: xyz=(0,0,0.157), rpy=(0,0,pi)  -> Rz(pi)
                [-1.0,  0.0,  0.0,  0.0],
                [ 0.0, -1.0,  0.0,  0.0],
                [ 0.0,  0.0,  1.0,  0.157],
                [ 0.0,  0.0,  0.0,  1.0],
            ],
            [  # upperArm_joint: xyz=(0,0.119,0), rpy=(-pi/2,-pi/2,0) -> [[0,1,0],[0,0,1],[1,0,0]]
                [ 0.0,  1.0,  0.0,  0.0],
                [ 0.0,  0.0,  1.0,  0.119],
                [ 1.0,  0.0,  0.0,  0.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ],
            [  # foreArm_joint: xyz=(0.266,0,0), rpy=(-pi,0,0) -> Rx(-pi)
                [ 1.0,  0.0,  0.0,  0.266],
                [ 0.0, -1.0,  0.0,  0.0],
                [ 0.0,  0.0, -1.0,  0.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ],
            [  # wrist1_joint: xyz=(0.2565,0,0), rpy=(pi,0,pi/2) -> Rz(pi/2)*Rx(pi)
                [ 0.0,  1.0,  0.0,  0.2565],
                [ 1.0,  0.0,  0.0,  0.0],
                [ 0.0,  0.0, -1.0,  0.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ],
            [  # wrist2_joint: xyz=(0,0.1025,0), rpy=(-pi/2,0,0) -> Rx(-pi/2)
                [ 1.0,  0.0,  0.0,  0.0],
                [ 0.0,  0.0,  1.0,  0.1025],
                [ 0.0, -1.0,  0.0,  0.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ],
            [  # wrist3_joint: xyz=(0,-0.094,0), rpy=(pi/2,0,0) -> Rx(pi/2)
                [ 1.0,  0.0,  0.0,  0.0],
                [ 0.0,  0.0, -1.0, -0.094],
                [ 0.0,  1.0,  0.0,  0.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ],
        ]
    )

    poses = np.zeros((7, 4, 4))
    poses[0, :, :] = base_pose
    for i in range(6):
        poses[i + 1, :, :] = np.dot(
            poses[i], np.dot(joint_origins[i], axis_angle(joint_axes[i], cfg[i]))
        )
    return poses


@numba.jit(nopython=True, cache=True)
def aubo_arm_visual_fk(cfg: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    """
    Visual FK for the Aubo i3H.  Same as link FK since each visual mesh
    is defined in the link's own frame without an extra transform.

    Returns poses in the following order:
        base_link, shoulder_Link, upperArm_Link, foreArm_Link,
        wrist1_Link, wrist2_Link, wrist3_Link
    """
    return aubo_arm_link_fk(cfg, base_pose)


@numba.jit(nopython=True, cache=True)
def label(array, lbl):
    return np.concatenate((array, lbl * np.ones((array.shape[0], 1))), axis=1)


@numba.jit(nopython=True, cache=True)
def get_points_on_aubo_arm_from_poses(
    poses,
    sample,
    base_link_points,
    shoulder_Link_points,
    upperArm_Link_points,
    foreArm_Link_points,
    wrist1_Link_points,
    wrist2_Link_points,
    wrist3_Link_points,
):
    """
    Transform pre-sampled link points into world frame and concatenate them.

    :param poses: output of aubo_arm_visual_fk, shape (7, 4, 4)
    :param sample: number of points to randomly sub-sample (0 = return all)
    :return: (N, 4) array of [x, y, z, link_idx]
    """
    assert len(poses) == 7
    all_points = np.concatenate(
        (
            label(transform_in_place(np.copy(base_link_points),    poses[0]), 0.0),
            label(transform_in_place(np.copy(shoulder_Link_points), poses[1]), 1.0),
            label(transform_in_place(np.copy(upperArm_Link_points), poses[2]), 2.0),
            label(transform_in_place(np.copy(foreArm_Link_points),  poses[3]), 3.0),
            label(transform_in_place(np.copy(wrist1_Link_points),   poses[4]), 4.0),
            label(transform_in_place(np.copy(wrist2_Link_points),   poses[5]), 5.0),
            label(transform_in_place(np.copy(wrist3_Link_points),   poses[6]), 6.0),
        ),
        axis=0,
    )
    if sample > 0:
        return all_points[
            np.random.choice(all_points.shape[0], sample, replace=False), :
        ]
    return all_points


@numba.jit(nopython=True, cache=True)
def get_points_on_aubo_arm(
    cfg,
    sample,
    base_link_points,
    shoulder_Link_points,
    upperArm_Link_points,
    foreArm_Link_points,
    wrist1_Link_points,
    wrist2_Link_points,
    wrist3_Link_points,
):
    """
    Compute FK then return labeled surface point cloud for the Aubo i3H arm.

    :param cfg: joint angles, shape (6,)
    :param sample: number of points to sub-sample (0 = return all)
    :return: (N, 4) array of [x, y, z, link_idx]
    """
    fk = aubo_arm_visual_fk(cfg, base_pose=np.eye(4))
    return get_points_on_aubo_arm_from_poses(
        fk,
        sample,
        base_link_points,
        shoulder_Link_points,
        upperArm_Link_points,
        foreArm_Link_points,
        wrist1_Link_points,
        wrist2_Link_points,
        wrist3_Link_points,
    )
