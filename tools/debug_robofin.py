#!/usr/bin/env python3
"""Small debug playground for learning Robofin from the debugger."""

import argparse
from pprint import pprint


def _section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def _show_pose(name, pose):
    print(f"{name}.xyz  = {pose.xyz}")
    print(f"{name}.quat = {pose.so3.q}")
    print(f"{name}.matrix =")
    print(pose.matrix)


def debug_aubo_fk(device):
    import numpy as np
    import torch

    from robofin.robot_constants_aubo import AuboConstants
    from robofin.robots_aubo import AuboRobot
    from robofin.torch_urdf import TorchURDF

    _section("Aubo FK")
    cfg = AuboConstants.NEUTRAL.astype(np.float32)
    pose = AuboRobot.fk(cfg)
    _show_pose("AuboRobot.fk(neutral)", pose)

    robot = TorchURDF.load(AuboConstants.urdf, lazy_load_meshes=True, device=device)
    cfg_t = torch.as_tensor(cfg, dtype=torch.float32, device=device).unsqueeze(0)
    link_fk = robot.link_fk_batch(cfg_t, use_names=True)

    print("TorchURDF link names:")
    pprint(list(link_fk.keys()))
    print("wrist3_Link pose from TorchURDF:")
    print(link_fk["wrist3_Link"].squeeze(0).cpu().numpy())
    print("position delta vs AuboRobot.fk:")
    print(link_fk["wrist3_Link"].squeeze(0).cpu().numpy()[:3, 3] - np.asarray(pose.xyz))


def debug_aubo_ik():
    import numpy as np

    from robofin.robot_constants_aubo import AuboConstants
    from robofin.robots_aubo import AuboRobot

    _section("Aubo IK")
    cfg = AuboConstants.NEUTRAL.astype(np.float64)
    target_pose = AuboRobot.fk(cfg)
    solutions = AuboRobot.ik(target_pose)

    print(f"number of IK solutions = {len(solutions)}")
    if solutions:
        first = np.asarray(solutions[0])
        print("first IK solution =")
        print(first)
        print("delta to neutral =")
        print(first - cfg)


def debug_aubo_sampler(device, num_robot_points, num_eef_points):
    import torch

    from robofin.robot_constants_aubo import AuboConstants
    from robofin.samplers_aubo import TorchAuboSampler

    _section("Aubo Sampler")
    sampler = TorchAuboSampler(
        num_robot_points=num_robot_points,
        num_eef_points=num_eef_points,
        use_cache=True,
        device=device,
    )
    cfg = torch.as_tensor(AuboConstants.NEUTRAL, dtype=torch.float32, device=device)
    arm_pc = sampler.sample(cfg)
    eef_pose = sampler.end_effector_pose(cfg)
    eef_pc = sampler.sample_end_effector(eef_pose)

    print(f"sampler.device = {sampler.device}")
    print(f"available cached keys = {sorted(sampler.points.keys())}")
    print(f"arm point cloud shape = {tuple(arm_pc.shape)}")
    print(f"eef pose shape        = {tuple(eef_pose.shape)}")
    print(f"eef point cloud shape = {tuple(eef_pc.shape)}")
    print("first 5 arm points [x, y, z, link_idx] =")
    print(arm_pc[0, :5].detach().cpu().numpy())


def debug_aubo_urdf(device):
    from robofin.robot_constants_aubo import AuboConstants
    from robofin.torch_urdf import TorchURDF

    _section("Aubo URDF")
    robot = TorchURDF.load(AuboConstants.urdf, lazy_load_meshes=True, device=device)
    print(f"urdf path = {AuboConstants.urdf}")
    print(f"number of links  = {len(robot.links)}")
    print(f"number of joints = {len(robot.joints)}")
    print("first 10 links:")
    pprint([link.name for link in robot.links[:10]])
    print("first 10 joints:")
    pprint([joint.name for joint in robot.joints[:10]])


def debug_franka_fk(device):
    import numpy as np
    import torch

    from robofin.robot_constants import FrankaConstants
    from robofin.robots import FrankaRobot
    from robofin.torch_urdf import TorchURDF

    _section("Franka FK")
    cfg = FrankaConstants.NEUTRAL.astype(np.float32)
    gripper = 0.04
    pose = FrankaRobot.fk(cfg)
    _show_pose("FrankaRobot.fk(neutral)", pose)

    robot = TorchURDF.load(FrankaConstants.urdf, lazy_load_meshes=True, device=device)
    cfg_t = torch.as_tensor([*cfg, gripper], dtype=torch.float32, device=device).unsqueeze(0)
    link_fk = robot.link_fk_batch(cfg_t, use_names=True)

    print("TorchURDF link names:")
    pprint(list(link_fk.keys()))
    print("panda_link8 pose from TorchURDF:")
    print(link_fk["panda_link8"].squeeze(0).cpu().numpy())
    print("right_gripper translation from FrankaRobot.fk:")
    print(np.asarray(pose.xyz))


def debug_franka_ik():
    import numpy as np

    from robofin.robot_constants import FrankaConstants
    from robofin.robots import FrankaRobot

    _section("Franka IK")
    cfg = FrankaConstants.NEUTRAL.astype(np.float64)
    target_pose = FrankaRobot.fk(cfg)
    solutions = FrankaRobot.ik(target_pose, panda_link7=cfg[-1])

    print(f"number of IK solutions = {len(solutions)}")
    if solutions:
        first = np.asarray(solutions[0])
        print("first IK solution =")
        print(first)
        print("delta to neutral =")
        print(first - cfg)


def debug_franka_sampler(device, num_robot_points, num_eef_points):
    import torch

    from robofin.robot_constants import FrankaConstants
    from robofin.robots import FrankaRobot
    from robofin.samplers import TorchFrankaSampler

    _section("Franka Sampler")
    sampler = TorchFrankaSampler(
        num_robot_points=num_robot_points,
        num_eef_points=num_eef_points,
        use_cache=True,
        with_base_link=True,
        device=device,
    )
    cfg = torch.as_tensor(FrankaConstants.NEUTRAL, dtype=torch.float32, device=device)
    gripper = 0.04
    arm_pc = sampler.sample(cfg, gripper)
    eef_pose = torch.as_tensor(
        FrankaRobot.fk(FrankaConstants.NEUTRAL).matrix,
        dtype=torch.float32,
        device=device,
    )
    eef_pc = sampler.sample_end_effector(eef_pose, gripper)

    print(f"sampler.device = {sampler.device}")
    print(f"available cached keys = {sorted(sampler.points.keys())}")
    print(f"arm point cloud shape = {tuple(arm_pc.shape)}")
    print(f"eef point cloud shape = {tuple(eef_pc.shape)}")
    print("first 5 arm points [x, y, z, link_idx] =")
    print(arm_pc[0, :5].detach().cpu().numpy())


def debug_franka_urdf(device):
    from robofin.robot_constants import FrankaConstants
    from robofin.torch_urdf import TorchURDF

    _section("Franka URDF")
    robot = TorchURDF.load(FrankaConstants.urdf, lazy_load_meshes=True, device=device)
    print(f"urdf path = {FrankaConstants.urdf}")
    print(f"number of links  = {len(robot.links)}")
    print(f"number of joints = {len(robot.joints)}")
    print("first 10 links:")
    pprint([link.name for link in robot.links[:10]])
    print("first 10 joints:")
    pprint([joint.name for joint in robot.joints[:10]])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Debugger-friendly playground for learning the Robofin codebase."
    )
    parser.add_argument("--robot", choices=["aubo", "franka"], default="aubo")
    parser.add_argument(
        "--mode",
        choices=["all", "fk", "ik", "sampler", "urdf"],
        default="all",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-robot-points", type=int, default=512)
    parser.add_argument("--num-eef-points", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        if args.robot == "aubo":
            if args.mode in ("all", "fk"):
                debug_aubo_fk(args.device)
            if args.mode in ("all", "ik"):
                debug_aubo_ik()
            if args.mode in ("all", "sampler"):
                debug_aubo_sampler(
                    args.device,
                    args.num_robot_points,
                    args.num_eef_points,
                )
            if args.mode in ("all", "urdf"):
                debug_aubo_urdf(args.device)
            return

        if args.mode in ("all", "fk"):
            debug_franka_fk(args.device)
        if args.mode in ("all", "ik"):
            debug_franka_ik()
        if args.mode in ("all", "sampler"):
            debug_franka_sampler(
                args.device,
                args.num_robot_points,
                args.num_eef_points,
            )
        if args.mode in ("all", "urdf"):
            debug_franka_urdf(args.device)
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            "Missing Python dependency while starting the debug playground: "
            f"{missing}\n"
            "Select the project's Python/conda interpreter in VS Code first, "
            "then install the dependencies declared in setup.cfg."
        ) from exc


if __name__ == "__main__":
    main()
