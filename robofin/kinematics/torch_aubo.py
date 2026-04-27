# ---------------------------------------------------------------------------
# Aubo i3H forward kinematics
# Joint origin matrices derived from aubo_i3H.urdf:
#   shoulder_joint : xyz=(0,0,0.157),   rpy=(0,     0,   pi)
#   upperArm_joint : xyz=(0,0.119,0),   rpy=(-pi/2,-pi/2,0)
#   foreArm_joint  : xyz=(0.266,0,0),   rpy=(-pi,   0,   0)
#   wrist1_joint   : xyz=(0.2565,0,0),  rpy=(pi,    0,   pi/2)
#   wrist2_joint   : xyz=(0,0.1025,0),  rpy=(-pi/2, 0,   0)
#   wrist3_joint   : xyz=(0,-0.094,0),  rpy=(pi/2,  0,   0)
# All revolute joints rotate about their local z-axis.
# ---------------------------------------------------------------------------

import torch

# torch.compile was added in PyTorch 2.0; fall back to identity on 1.x
if not hasattr(torch, "compile"):
    torch.compile = lambda f: f  # type: ignore[attr-defined]


def axis_angle(axis, angle):
    sina = torch.sin(angle)
    cosa = torch.cos(angle)
    axis = axis / torch.linalg.norm(axis)

    # rotation matrix around unit vector
    B = angle.size(0)
    M = torch.eye(4).unsqueeze(0).repeat((B, 1, 1)).type_as(angle)
    M[:, 0, 0] *= cosa
    M[:, 1, 1] *= cosa
    M[:, 2, 2] *= cosa
    M[:, :3, :3] += (
        torch.outer(axis, axis)[None, :, :].repeat((B, 1, 1))
        * (1.0 - cosa)[:, None, None]
    )

    M[:, 0, 1] += -axis[2] * sina
    M[:, 0, 2] += axis[1] * sina
    M[:, 1, 0] += axis[2] * sina
    M[:, 1, 2] += -axis[0] * sina
    M[:, 2, 0] += -axis[1] * sina
    M[:, 2, 1] += axis[0] * sina

    return M


@torch.compile
def aubo_eef_link_fk(prismatic_joint: float, base_pose: torch.Tensor) -> torch.Tensor:
    """
    Fast Torch-based FK for the Aubo i3H end-effector links.

    Unlike the Franka gripper, the stock Aubo i3H URDF in this repository
    does not add any fixed hand/finger/tool links after ``wrist3_Link``.
    Therefore the end-effector chain contains only one pose:

        [
            "wrist3_Link",
        ]

    :param prismatic_joint: Unused. Kept only for API compatibility with the
        Franka helper functions.
    :param base_pose: pose of ``wrist3_Link``, shape (4, 4) or (B, 4, 4)
    :return: poses shape (1, 4, 4) or (B, 1, 4, 4)
    """
    squeeze = False
    if base_pose.ndim == 2:
        base_pose = base_pose.unsqueeze(0)
        squeeze = True

    B = base_pose.size(0)
    poses = torch.zeros((B, 1, 4, 4)).type_as(base_pose)
    poses[:, 0, :, :] = base_pose

    if squeeze:
        return poses.squeeze(0)
    return poses


@torch.compile
def aubo_eef_visual_fk(prismatic_joint: float, base_pose: torch.Tensor) -> torch.Tensor:
    """
    Visual FK for the Aubo i3H end-effector chain.

    In the current Aubo i3H URDF, the only end-effector visual link is
    ``wrist3_Link`` itself, and its visual geometry is defined directly in the
    link frame. So the visual FK is identical to the link FK.

    :param prismatic_joint: Unused. Kept only for API compatibility with the
        Franka helper functions.
    :param base_pose: pose of ``wrist3_Link``, shape (4, 4) or (B, 4, 4)
    :return: poses shape (1, 4, 4) or (B, 1, 4, 4)
    """
    return aubo_eef_link_fk(prismatic_joint, base_pose)


@torch.compile
def aubo_arm_link_fk(
    cfg: torch.Tensor, base_pose: torch.Tensor
) -> torch.Tensor:
    """
    Fast PyTorch FK for the Aubo i3H (6-DOF, no gripper).

    :param cfg: joint angles, shape (6,) or (B, 6)
    :param base_pose: base transform, shape (4, 4) or (B, 4, 4)
    :return: poses shape (B, 7, 4, 4) or (7, 4, 4):
        [base_link, shoulder_Link, upperArm_Link, foreArm_Link,
         wrist1_Link, wrist2_Link, wrist3_Link]
    """
    joint_axes = torch.Tensor(
        [
            [0.0, 0.0, 1.0],  # shoulder_joint
            [0.0, 0.0, 1.0],  # upperArm_joint
            [0.0, 0.0, 1.0],  # foreArm_joint
            [0.0, 0.0, 1.0],  # wrist1_joint
            [0.0, 0.0, 1.0],  # wrist2_joint
            [0.0, 0.0, 1.0],  # wrist3_joint
        ]
    ).type_as(cfg)

    joint_origins = torch.Tensor(
        [
            [  # shoulder_joint: xyz=(0,0,0.157), rpy=(0,0,pi)
                [-1.0,  0.0,  0.0,  0.0],
                [ 0.0, -1.0,  0.0,  0.0],
                [ 0.0,  0.0,  1.0,  0.157],
                [ 0.0,  0.0,  0.0,  1.0],
            ],
            [  # upperArm_joint: xyz=(0,0.119,0), rpy=(-pi/2,-pi/2,0)
                [ 0.0,  1.0,  0.0,  0.0],
                [ 0.0,  0.0,  1.0,  0.119],
                [ 1.0,  0.0,  0.0,  0.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ],
            [  # foreArm_joint: xyz=(0.266,0,0), rpy=(-pi,0,0)
                [ 1.0,  0.0,  0.0,  0.266],
                [ 0.0, -1.0,  0.0,  0.0],
                [ 0.0,  0.0, -1.0,  0.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ],
            [  # wrist1_joint: xyz=(0.2565,0,0), rpy=(pi,0,pi/2)
                [ 0.0,  1.0,  0.0,  0.2565],
                [ 1.0,  0.0,  0.0,  0.0],
                [ 0.0,  0.0, -1.0,  0.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ],
            [  # wrist2_joint: xyz=(0,0.1025,0), rpy=(-pi/2,0,0)
                [ 1.0,  0.0,  0.0,  0.0],
                [ 0.0,  0.0,  1.0,  0.1025],
                [ 0.0, -1.0,  0.0,  0.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ],
            [  # wrist3_joint: xyz=(0,-0.094,0), rpy=(pi/2,0,0)
                [ 1.0,  0.0,  0.0,  0.0],
                [ 0.0,  0.0, -1.0, -0.094],
                [ 0.0,  1.0,  0.0,  0.0],
                [ 0.0,  0.0,  0.0,  1.0],
            ],
        ]
    ).type_as(cfg)

    squeeze = False
    if cfg.ndim == 1:
        cfg = cfg.unsqueeze(0)
        squeeze = True
    if base_pose.ndim == 2:
        base_pose = base_pose.unsqueeze(0)

    B = cfg.size(0)
    poses = [base_pose.expand(B, -1, -1)]
    for i in range(6):
        pose = torch.matmul(
            poses[-1],
            torch.matmul(joint_origins[i], axis_angle(joint_axes[i], cfg[:, i])),
        )
        poses.append(pose)
    poses = torch.stack(poses, dim=1)

    if squeeze:
        return poses.squeeze(0)
    return poses


@torch.compile
def aubo_arm_visual_fk(
    cfg: torch.Tensor, base_pose: torch.Tensor
) -> torch.Tensor:
    """
    Visual FK for the Aubo i3H.  Identical to aubo_arm_link_fk since each
    visual mesh is defined in the link's own frame.

    Returns poses in the order:
        base_link, shoulder_Link, upperArm_Link, foreArm_Link,
        wrist1_Link, wrist2_Link, wrist3_Link
    """
    return aubo_arm_link_fk(cfg, base_pose)
