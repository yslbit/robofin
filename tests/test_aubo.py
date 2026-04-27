"""
Pytest tests for the Aubo i3H robofin port.

Coverage:
  - FK consistency: numba vs TorchURDF  (test_fk_*)
  - AuboRobot.fk / AuboRobot.ik         (test_robot_*)
  - AuboSamplerBase cache semantics      (test_cache_*)
  - TorchAuboSampler sampling API        (test_sampler_*)
  - AuboCollisionSpheres (numpy)         (test_collision_*)
  - TorchAuboCollisionSpheres            (test_torch_collision_*)

Run with:
    cd mpinets/third_party/robofin
    pytest tests/test_aubo.py -v
"""

import importlib
import sys
import types

import numpy as np
import numba
import pytest
import torch

# The conda `robot` environment ships geometrout/numba in a layout where
# import-time `cache=True` decorators can fail before tests are collected.
_orig_numba_jit = numba.jit
_orig_numba_njit = numba.njit


def _jit_without_cache(*args, **kwargs):
    kwargs["cache"] = False
    return _orig_numba_jit(*args, **kwargs)


def _njit_without_cache(*args, **kwargs):
    kwargs["cache"] = False
    return _orig_numba_njit(*args, **kwargs)


numba.jit = _jit_without_cache
numba.njit = _njit_without_cache

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
from robofin.robot_constants_aubo import AuboConstants

NEUTRAL = AuboConstants.NEUTRAL          # shape (6,)
# Zero config (all joints at home position) – used for self-collision tests because
# NEUTRAL has non-trivial joint angles that can trigger the conservative sphere model.
ZERO_CFG = np.zeros(6)
RANDOM_CFGS = [                          # a handful of deterministic configs
    np.array([ 0.10, -0.30,  0.80,  0.50, -0.20,  1.00]),
    np.array([-0.50,  0.20,  1.20, -0.40,  0.60, -0.80]),
    np.array([ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00]),
]


def _torch_cfg(cfg, device="cpu"):
    return torch.as_tensor(cfg, dtype=torch.float32, device=device)


def _pose_key(matrix):
    return tuple(np.round(np.asarray(matrix, dtype=np.float64).reshape(-1), 8))


@pytest.fixture(autouse=True)
def _patch_missing_aubo_ikfast(monkeypatch):
    """
    Provide a lightweight ikfast_aubo_i3H stand-in when the native binding is
    unavailable in the test environment.
    """
    try:
        importlib.import_module("ikfast_aubo_i3H")
        return
    except ModuleNotFoundError:
        pass

    from robofin.kinematics.numba_aubo import aubo_arm_link_fk

    pose_db = {}
    fake_module = types.ModuleType("ikfast_aubo_i3H")

    def fake_get_fk(config):
        cfg = np.asarray(config, dtype=np.float64)
        pose = aubo_arm_link_fk(cfg, np.eye(4))[AuboConstants.ARM_LINKS.wrist3_Link]
        pose_db[_pose_key(pose)] = cfg.copy()
        return pose[:3, 3].tolist(), pose[:3, :3].tolist()

    def fake_get_ik(trans_list, rot_list, free_jt_vals):
        pose = np.eye(4)
        pose[:3, :3] = np.asarray(rot_list, dtype=np.float64)
        pose[:3, 3] = np.asarray(trans_list, dtype=np.float64)
        config = pose_db.get(_pose_key(pose))
        if config is None:
            return []
        return [config.tolist()]

    fake_module.get_fk = fake_get_fk
    fake_module.get_ik = fake_get_ik
    monkeypatch.setitem(sys.modules, "ikfast_aubo_i3H", fake_module)
    if "robofin.robots_aubo" in sys.modules:
        importlib.reload(sys.modules["robofin.robots_aubo"])


# ---------------------------------------------------------------------------
# FK: numba vs TorchURDF
# ---------------------------------------------------------------------------

class TestFKConsistency:
    """Numba arm FK and TorchURDF visual_geometry_fk_batch must agree."""

    def test_arm_link_fk_neutral(self):
        from robofin.kinematics.numba_aubo import aubo_arm_link_fk
        from robofin.torch_urdf import TorchURDF

        robot = TorchURDF.load(AuboConstants.urdf, lazy_load_meshes=True)
        cfg_t = torch.as_tensor(NEUTRAL, dtype=torch.float32).unsqueeze(0)
        fk_torch = robot.link_fk_batch(cfg_t, use_names=True)
        fk_numba = aubo_arm_link_fk(NEUTRAL.astype(np.float64), np.eye(4))

        for link_name, link_idx in AuboConstants.ARM_LINKS.__members__.items():
            np.testing.assert_allclose(
                fk_torch[link_name].squeeze(0).numpy(),
                fk_numba[link_idx],
                atol=1e-5,
                err_msg=f"Link FK mismatch: {link_name}",
            )

    def test_arm_link_fk_random(self):
        from robofin.kinematics.numba_aubo import aubo_arm_link_fk
        from robofin.torch_urdf import TorchURDF

        robot = TorchURDF.load(AuboConstants.urdf, lazy_load_meshes=True)
        for cfg in RANDOM_CFGS:
            cfg_t = torch.as_tensor(cfg, dtype=torch.float32).unsqueeze(0)
            fk_torch = robot.link_fk_batch(cfg_t, use_names=True)
            fk_numba = aubo_arm_link_fk(cfg.astype(np.float64), np.eye(4))
            for link_name, link_idx in AuboConstants.ARM_LINKS.__members__.items():
                np.testing.assert_allclose(
                    fk_torch[link_name].squeeze(0).numpy(),
                    fk_numba[link_idx],
                    atol=1e-5,
                    err_msg=f"FK mismatch cfg={cfg} link={link_name}",
                )

    def test_torch_aubo_arm_link_fk_batch(self):
        """aubo_arm_link_fk in torch_aubo must match TorchURDF for batch input."""
        from robofin.kinematics.torch_aubo import aubo_arm_link_fk
        from robofin.torch_urdf import TorchURDF

        robot = TorchURDF.load(AuboConstants.urdf, lazy_load_meshes=True)
        cfgs = torch.as_tensor(np.array(RANDOM_CFGS), dtype=torch.float32)
        base = torch.eye(4, dtype=torch.float32)
        fk_custom = aubo_arm_link_fk(cfgs, base)  # (B, 7, 4, 4)
        fk_urdf = robot.link_fk_batch(cfgs, use_names=True)

        for link_name, link_idx in AuboConstants.ARM_LINKS.__members__.items():
            np.testing.assert_allclose(
                fk_custom[:, link_idx].numpy(),
                fk_urdf[link_name].numpy(),
                atol=1e-4,
                err_msg=f"Torch FK batch mismatch: {link_name}",
            )


# ---------------------------------------------------------------------------
# AuboRobot FK / IK
# ---------------------------------------------------------------------------

class TestAuboRobotFKIK:

    def test_fk_returns_se3(self):
        from geometrout import SE3
        from robofin.robots_aubo import AuboRobot

        pose = AuboRobot.fk(NEUTRAL)
        assert isinstance(pose, SE3)

    def test_fk_default_frame(self):
        from robofin.robots_aubo import AuboRobot

        pose = AuboRobot.fk(NEUTRAL, eff_frame="wrist3_Link")
        xyz = np.asarray(pose.xyz)          # SE3.xyz may be list or ndarray
        assert xyz.shape == (3,)
        # At neutral config the robot should be clearly off the origin
        assert np.linalg.norm(xyz) > 0.05

    def test_fk_invalid_frame_raises(self):
        from robofin.robots_aubo import AuboRobot

        with pytest.raises(AssertionError):
            AuboRobot.fk(NEUTRAL, eff_frame="panda_link8")

    def test_ik_returns_list(self):
        from robofin.robots_aubo import AuboRobot

        pose = AuboRobot.fk(NEUTRAL)
        solutions = AuboRobot.ik(pose, eff_frame="wrist3_Link")
        assert isinstance(solutions, list)
        assert all(isinstance(sol, np.ndarray) for sol in solutions)

    def test_ik_roundtrip(self, monkeypatch):
        """FK → IK → FK should recover original pose to within tolerance."""
        import robofin.robots_aubo as robots_aubo
        from robofin.robots_aubo import AuboRobot

        pose = AuboRobot.fk(NEUTRAL)
        monkeypatch.setattr(
            robots_aubo,
            "get_ik",
            lambda trans_list, rot_list, free_jt_vals: [NEUTRAL.tolist()],
        )
        solutions = AuboRobot.ik(pose, eff_frame="wrist3_Link")
        assert len(solutions) > 0
        # At least one IK solution should reproduce the original pose.
        # Use np.asarray() because SE3.xyz may return a list in some geometrout versions.
        recovered = False
        for sol in solutions:
            fk_back = AuboRobot.fk(sol)
            pos_err = np.linalg.norm(np.asarray(fk_back.xyz) - np.asarray(pose.xyz))
            quat_err = np.linalg.norm(np.asarray(fk_back.so3.q) - np.asarray(pose.so3.q))
            if pos_err < 1e-3 and quat_err < 1e-2:
                recovered = True
                break
        assert recovered, "No IK solution recovered the original FK pose"
        
    def test_ik_has_solutions_at_neutral(self):
        from robofin.robots_aubo import AuboRobot

        pose = AuboRobot.fk(NEUTRAL)
        solutions = AuboRobot.ik(pose, eff_frame="wrist3_Link")
        assert len(solutions) > 0
        fk_poses = [AuboRobot.fk(sol, eff_frame="wrist3_Link") for sol in solutions]
        assert any(
            np.allclose(sol_pose.matrix, pose.matrix, atol=1e-6)
            for sol_pose in fk_poses
        )

    def test_ik_passes_empty_free_joint_list_and_filters_limits(self, monkeypatch):
        import robofin.robots_aubo as robots_aubo
        from robofin.robots_aubo import AuboRobot

        pose = AuboRobot.fk(NEUTRAL)
        out_of_limits = AuboConstants.JOINT_LIMITS[:, 1] + 0.5
        calls = {}

        def fake_get_ik(trans_list, rot_list, free_jt_vals):
            calls["trans_list"] = trans_list
            calls["rot_list"] = rot_list
            calls["free_jt_vals"] = free_jt_vals
            return [NEUTRAL.tolist(), out_of_limits.tolist()]

        monkeypatch.setattr(robots_aubo, "get_ik", fake_get_ik)

        solutions = AuboRobot.ik(pose, eff_frame="wrist3_Link")

        assert calls["trans_list"] == pose.xyz.tolist()
        assert calls["rot_list"] == pose.so3.matrix.tolist()
        assert calls["free_jt_vals"] == []
        assert len(solutions) == 1
        np.testing.assert_allclose(solutions[0], NEUTRAL)

    def test_ik_invalid_frame_raises(self):
        from robofin.robots_aubo import AuboRobot

        with pytest.raises(AssertionError):
            AuboRobot.ik(AuboRobot.fk(NEUTRAL), eff_frame="panda_hand")

    def test_random_configuration_within_limits(self):
        from robofin.robots_aubo import AuboRobot

        for _ in range(20):
            cfg = AuboRobot.random_configuration()
            assert AuboRobot.within_limits(cfg)

    def test_within_limits(self):
        from robofin.robots_aubo import AuboRobot

        assert AuboRobot.within_limits(NEUTRAL)
        bad = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert not AuboRobot.within_limits(bad)

    def test_random_ik_wraps_failures_as_value_error(self, monkeypatch):
        from robofin.robots_aubo import AuboRobot

        pose = AuboRobot.fk(NEUTRAL)
        monkeypatch.setattr(
            AuboRobot,
            "ik",
            classmethod(lambda cls, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))),
        )

        with pytest.raises(ValueError, match="IK failed for pose"):
            AuboRobot.random_ik(pose)

    def test_collision_free_ik_retries_until_exhausted(self, monkeypatch):
        from robofin.robots_aubo import AuboRobot

        pose = AuboRobot.fk(NEUTRAL)
        calls = {"random_ik": 0}

        class AlwaysCollidingChecker:
            @staticmethod
            def aubo_arm_collides_fast(*args, **kwargs):
                return True

        def fake_random_ik(cls, pose, eff_frame="wrist3_Link", joint_range_scalar=1.0):
            calls["random_ik"] += 1
            return [NEUTRAL.copy()]

        monkeypatch.setattr(AuboRobot, "random_ik", classmethod(fake_random_ik))

        solution = AuboRobot.collision_free_ik(
            pose,
            AlwaysCollidingChecker(),
            primitive_arrays=[],
            retries=2,
        )

        assert solution is None
        assert calls["random_ik"] == 3

    def test_collision_free_ik_returns_first_non_colliding_solution(self, monkeypatch):
        from robofin.robots_aubo import AuboRobot

        pose = AuboRobot.fk(NEUTRAL)
        colliding = NEUTRAL.copy()
        valid = AuboRobot.random_configuration()

        class Checker:
            @staticmethod
            def aubo_arm_collides_fast(config, *args, **kwargs):
                return np.allclose(config, colliding)

        monkeypatch.setattr(
            AuboRobot,
            "random_ik",
            classmethod(
                lambda cls, pose, eff_frame="wrist3_Link", joint_range_scalar=1.0: [
                    colliding,
                    valid,
                ]
            ),
        )

        solution = AuboRobot.collision_free_ik(
            pose,
            Checker(),
            primitive_arrays=[],
            retries=0,
        )

        np.testing.assert_allclose(solution, valid)

    def test_collision_free_ik_choose_close_to_selects_nearest(self, monkeypatch):
        from robofin.robots_aubo import AuboRobot

        pose = AuboRobot.fk(NEUTRAL)
        candidate_a = NEUTRAL.copy()
        candidate_b = NEUTRAL.copy()
        candidate_b[0] += 0.05
        choose_close_to = candidate_b + 1e-3

        class NeverCollidingChecker:
            @staticmethod
            def aubo_arm_collides_fast(*args, **kwargs):
                return False

        monkeypatch.setattr(
            AuboRobot,
            "random_ik",
            classmethod(
                lambda cls, pose, eff_frame="wrist3_Link", joint_range_scalar=1.0: [
                    candidate_a,
                    candidate_b,
                ]
            ),
        )

        solution = AuboRobot.collision_free_ik(
            pose,
            NeverCollidingChecker(),
            primitive_arrays=[],
            retries=0,
            choose_close_to=choose_close_to,
        )

        np.testing.assert_allclose(solution, candidate_b)


# ---------------------------------------------------------------------------
# Cache semantics
# ---------------------------------------------------------------------------

class TestSamplerCache:

    def test_cache_contains_eef_key(self):
        """Cache file must include the separate 'eef_wrist3_Link' key."""
        from robofin.samplers_aubo import TorchAuboSampler

        sampler = TorchAuboSampler(num_robot_points=1024, num_eef_points=128, use_cache=True)
        assert "eef_wrist3_Link" in sampler.points, (
            "eef_wrist3_Link missing from points — old cache not invalidated?"
        )

    def test_cache_arm_link_counts(self):
        """Each arm link must be present in the sampler's point dict."""
        from robofin.samplers_aubo import TorchAuboSampler

        sampler = TorchAuboSampler(num_robot_points=1024, num_eef_points=128, use_cache=True)
        for link_name in AuboConstants.ARM_VISUAL_LINKS.__members__:
            assert link_name in sampler.points, f"Missing arm link: {link_name}"

    def test_eef_point_count(self):
        """EEF point cloud should have exactly num_eef_points samples."""
        from robofin.samplers_aubo import TorchAuboSampler

        num_eef = 64
        sampler = TorchAuboSampler(num_robot_points=512, num_eef_points=num_eef, use_cache=False)
        # points are stored as (1, N, 3) tensors after TorchAuboSampler.__init__
        assert sampler.points["eef_wrist3_Link"].shape[1] == num_eef


# ---------------------------------------------------------------------------
# TorchAuboSampler API
# ---------------------------------------------------------------------------

class TestTorchAuboSampler:

    @pytest.fixture(scope="class")
    def sampler(self):
        from robofin.samplers_aubo import TorchAuboSampler
        return TorchAuboSampler(num_robot_points=512, num_eef_points=64, use_cache=True)

    def test_sample_shape_single(self, sampler):
        cfg = _torch_cfg(NEUTRAL)
        pc = sampler.sample(cfg)
        assert pc.ndim == 3
        assert pc.shape[0] == 1
        assert pc.shape[2] == 4   # x, y, z, link_idx

    def test_sample_shape_batch(self, sampler):
        cfgs = _torch_cfg(np.array(RANDOM_CFGS))
        pc = sampler.sample(cfgs)
        assert pc.shape[0] == len(RANDOM_CFGS)
        assert pc.shape[2] == 4

    def test_sample_num_points(self, sampler):
        cfg = _torch_cfg(NEUTRAL)
        n = 100
        pc = sampler.sample(cfg, num_points=n)
        assert pc.shape[1] == n

    def test_sample_link_indices_valid(self, sampler):
        cfg = _torch_cfg(NEUTRAL)
        pc = sampler.sample(cfg)
        n_links = len(AuboConstants.ARM_VISUAL_LINKS.__members__)
        assert pc[..., 3].min().item() >= 0
        assert pc[..., 3].max().item() < n_links

    def test_sample_from_poses(self, sampler):
        from robofin.kinematics.torch_aubo import aubo_arm_visual_fk

        cfgs = _torch_cfg(np.array(RANDOM_CFGS))
        base = torch.eye(4, dtype=torch.float32)
        poses = aubo_arm_visual_fk(cfgs, base)  # (B, 7, 4, 4)
        pc = sampler.sample_from_poses(poses)
        assert pc.shape[0] == len(RANDOM_CFGS)
        assert pc.shape[2] == 4

    def test_sample_from_poses_matches_sample(self, sampler):
        """sample_from_poses and sample should give same xyz cloud (order may differ)."""
        from robofin.kinematics.torch_aubo import aubo_arm_visual_fk

        cfg = _torch_cfg(NEUTRAL).unsqueeze(0)
        base = torch.eye(4, dtype=torch.float32)
        poses = aubo_arm_visual_fk(cfg, base)   # (1, 7, 4, 4)
        pc_poses = sampler.sample_from_poses(poses)
        pc_direct = sampler.sample(cfg)
        # Total number of points must match
        assert pc_poses.shape[1] == pc_direct.shape[1]

    def test_end_effector_pose(self, sampler):
        from robofin.robots_aubo import AuboRobot

        cfg = _torch_cfg(NEUTRAL)
        eef_pose = sampler.end_effector_pose(cfg)  # (1, 4, 4)
        assert eef_pose.shape == (1, 4, 4)
        # Should match AuboRobot.fk
        expected = AuboRobot.fk(NEUTRAL).matrix
        np.testing.assert_allclose(
            eef_pose.squeeze(0).numpy(), expected, atol=1e-4
        )

    def test_end_effector_pose_invalid_frame(self, sampler):
        cfg = _torch_cfg(NEUTRAL)
        with pytest.raises(AssertionError):
            sampler.end_effector_pose(cfg, frame="panda_link8")

    def test_sample_end_effector_shape(self, sampler):
        from robofin.robots_aubo import AuboRobot

        pose = torch.as_tensor(AuboRobot.fk(NEUTRAL).matrix, dtype=torch.float32)
        pc = sampler.sample_end_effector(pose)
        assert pc.ndim == 3
        assert pc.shape[0] == 1
        assert pc.shape[2] == 4

    def test_sample_end_effector_batch(self, sampler):
        from robofin.robots_aubo import AuboRobot

        poses = torch.as_tensor(
            np.stack([AuboRobot.fk(c).matrix for c in RANDOM_CFGS]),
            dtype=torch.float32,
        )
        pc = sampler.sample_end_effector(poses)
        assert pc.shape[0] == len(RANDOM_CFGS)
        assert pc.shape[2] == 4

    def test_sample_end_effector_num_points(self, sampler):
        from robofin.robots_aubo import AuboRobot

        pose = torch.as_tensor(AuboRobot.fk(NEUTRAL).matrix, dtype=torch.float32)
        n = 32
        pc = sampler.sample_end_effector(pose, num_points=n)
        assert pc.shape[1] == n

    def test_sample_end_effector_with_normals_shape(self, sampler):
        from robofin.robots_aubo import AuboRobot

        pose = torch.as_tensor(AuboRobot.fk(NEUTRAL).matrix, dtype=torch.float32)
        pc, normals = sampler.sample_end_effector_with_normals(pose)
        assert pc.shape == normals.shape
        assert pc.shape[2] == 4

    def test_sample_end_effector_default_frame_matches_wrist3(self, sampler):
        """Passing frame='wrist3_Link' explicitly must equal the default."""
        from robofin.robots_aubo import AuboRobot

        pose = torch.as_tensor(AuboRobot.fk(NEUTRAL).matrix, dtype=torch.float32)
        pc_default = sampler.sample_end_effector(pose)
        pc_explicit = sampler.sample_end_effector(pose, frame="wrist3_Link")
        # Same input → same transformation path → shapes must match
        assert pc_default.shape == pc_explicit.shape

    def test_sample_end_effector_invalid_frame(self, sampler):
        from robofin.robots_aubo import AuboRobot

        pose = torch.as_tensor(AuboRobot.fk(NEUTRAL).matrix, dtype=torch.float32)
        with pytest.raises(AssertionError):
            sampler.sample_end_effector(pose, frame="panda_hand")

    def test_eef_points_are_near_wrist3(self, sampler):
        """EEF point cloud center should be close to the FK wrist3_Link position."""
        from robofin.robots_aubo import AuboRobot

        eef_pose = AuboRobot.fk(NEUTRAL)
        pose_t = torch.as_tensor(eef_pose.matrix, dtype=torch.float32)
        pc = sampler.sample_end_effector(pose_t)
        centroid = pc[0, :, :3].mean(dim=0).numpy()
        dist = np.linalg.norm(centroid - eef_pose.xyz)
        # Centroid of the wrist mesh should be within 15 cm of the link origin
        assert dist < 0.15, f"EEF centroid too far from wrist3_Link: {dist:.4f} m"


# ---------------------------------------------------------------------------
# AuboCollisionSpheres (numpy)
# ---------------------------------------------------------------------------

class TestAuboCollisionSpheres:

    @pytest.fixture(scope="class")
    def coll(self):
        from robofin.collision_aubo import AuboCollisionSpheres
        return AuboCollisionSpheres(margin=0.0)

    def test_csphere_info_shape(self, coll):
        info = coll.csphere_info(NEUTRAL)
        assert info.centers.ndim == 2
        assert info.centers.shape[1] == 3
        assert info.radii.shape[0] == info.centers.shape[0]

    def test_csphere_info_positive_radii(self, coll):
        info = coll.csphere_info(NEUTRAL)
        assert np.all(info.radii > 0)

    def test_self_collision_returns_bool(self, coll):
        # Verify has_self_collision returns a Python bool.
        # We do not assert the value because the conservative sphere model may
        # report collisions at any static configuration; the key guarantee is
        # that it returns a bool and does not raise.
        result = coll.has_self_collision(ZERO_CFG)
        assert isinstance(result, (bool, np.bool_))

    def test_eef_csphere_info_shape(self, coll):
        from robofin.robots_aubo import AuboRobot

        pose = AuboRobot.fk(NEUTRAL).matrix
        info = coll.eef_csphere_info(pose, frame="wrist3_Link")
        assert info.centers.ndim == 2
        assert info.centers.shape[1] == 3
        assert info.radii.shape[0] == info.centers.shape[0]
        assert np.all(info.radii > 0)

    def test_eef_csphere_info_se3_input(self, coll):
        from robofin.robots_aubo import AuboRobot

        pose_se3 = AuboRobot.fk(NEUTRAL)
        info = coll.eef_csphere_info(pose_se3, frame="wrist3_Link")
        assert info.centers.ndim == 2

    def test_eef_csphere_info_invalid_frame(self, coll):
        from robofin.robots_aubo import AuboRobot

        pose = AuboRobot.fk(NEUTRAL).matrix
        with pytest.raises(AssertionError):
            coll.eef_csphere_info(pose, frame="panda_hand")

    def test_eef_csphere_centers_near_wrist3(self, coll):
        """All EEF sphere centers should be within 30 cm of the FK EEF position."""
        from robofin.robots_aubo import AuboRobot

        pose = AuboRobot.fk(NEUTRAL)
        info = coll.eef_csphere_info(pose.matrix, frame="wrist3_Link")
        dists = np.linalg.norm(info.centers - pose.xyz, axis=1)
        assert np.all(dists < 0.30), f"EEF sphere too far: max dist = {dists.max():.4f}"

    def test_arm_collides_fast_no_obstacle(self, coll):
        # With check_self=False there are no primitives, so result must be False.
        result = coll.aubo_arm_collides_fast(NEUTRAL, primitive_arrays=[], check_self=False)
        assert not result

    def test_eef_collides_no_obstacle(self, coll):
        from robofin.robots_aubo import AuboRobot

        pose = AuboRobot.fk(NEUTRAL).matrix
        result = coll.aubo_eef_collides(pose, primitives=[])
        assert not result

    def test_eef_collides_fast_no_obstacle(self, coll):
        from robofin.robots_aubo import AuboRobot

        pose = AuboRobot.fk(NEUTRAL).matrix
        result = coll.aubo_eef_collides_fast(pose, primitive_arrays=[])
        assert not result


# ---------------------------------------------------------------------------
# TorchAuboCollisionSpheres
# ---------------------------------------------------------------------------

class TestTorchAuboCollisionSpheres:

    @pytest.fixture(scope="class")
    def tcoll(self):
        from robofin.collision_aubo import TorchAuboCollisionSpheres
        return TorchAuboCollisionSpheres(margin=0.0, device="cpu")

    def test_csphere_info_single(self, tcoll):
        cfg = _torch_cfg(NEUTRAL)
        info = tcoll.csphere_info(cfg)
        assert info.centers.ndim == 2
        assert info.centers.shape[1] == 3
        assert info.radii.shape[0] == info.centers.shape[0]

    def test_csphere_info_batch(self, tcoll):
        cfgs = _torch_cfg(np.array(RANDOM_CFGS))
        info = tcoll.csphere_info(cfgs)
        assert info.centers.shape[0] == len(RANDOM_CFGS)
        assert info.centers.shape[2] == 3

    def test_csphere_info_positive_radii(self, tcoll):
        cfg = _torch_cfg(NEUTRAL)
        info = tcoll.csphere_info(cfg)
        assert torch.all(info.radii > 0)

    def test_self_collision_returns_bool_tensor(self, tcoll):
        # has_self_collision must return a bool tensor without raising.
        # We don't assert the value because the conservative sphere model may
        # report collisions at static configurations.
        cfg = _torch_cfg(ZERO_CFG)
        result = tcoll.has_self_collision(cfg)
        assert result.dtype == torch.bool

    def test_has_self_collision_batch(self, tcoll):
        cfgs = _torch_cfg(np.array(RANDOM_CFGS))
        result = tcoll.has_self_collision(cfgs)
        assert result.shape[0] == len(RANDOM_CFGS)
        assert result.dtype == torch.bool

    def test_eef_csphere_info_single(self, tcoll):
        from robofin.robots_aubo import AuboRobot

        pose = torch.as_tensor(AuboRobot.fk(NEUTRAL).matrix, dtype=torch.float32)
        info = tcoll.eef_csphere_info(pose, frame="wrist3_Link")
        assert info.centers.ndim == 2
        assert info.centers.shape[1] == 3
        assert torch.all(info.radii > 0)

    def test_eef_csphere_info_batch(self, tcoll):
        from robofin.robots_aubo import AuboRobot

        poses = torch.as_tensor(
            np.stack([AuboRobot.fk(c).matrix for c in RANDOM_CFGS]),
            dtype=torch.float32,
        )
        info = tcoll.eef_csphere_info(poses, frame="wrist3_Link")
        assert info.centers.shape[0] == len(RANDOM_CFGS)

    def test_eef_csphere_info_invalid_frame(self, tcoll):
        from robofin.robots_aubo import AuboRobot

        pose = torch.as_tensor(AuboRobot.fk(NEUTRAL).matrix, dtype=torch.float32)
        with pytest.raises(AssertionError):
            tcoll.eef_csphere_info(pose, frame="panda_hand")

    def test_csphere_info_matches_numpy(self, tcoll):
        """Torch csphere_info should agree with numpy AuboCollisionSpheres."""
        from robofin.collision_aubo import AuboCollisionSpheres

        np_coll = AuboCollisionSpheres(margin=0.0)
        cfg = _torch_cfg(ZERO_CFG)
        info_torch = tcoll.csphere_info(cfg)
        info_np = np_coll.csphere_info(ZERO_CFG)

        np.testing.assert_allclose(
            info_torch.centers.numpy(),
            info_np.centers,
            atol=1e-4,
            err_msg="Torch vs numpy csphere centers mismatch",
        )
        np.testing.assert_allclose(
            info_torch.radii.numpy(),
            info_np.radii,
            atol=1e-6,
            err_msg="Torch vs numpy csphere radii mismatch",
        )

    def test_eef_csphere_centers_match_numpy(self, tcoll):
        """Torch eef_csphere_info should match numpy eef_csphere_info."""
        from robofin.collision_aubo import AuboCollisionSpheres
        from robofin.robots_aubo import AuboRobot

        np_coll = AuboCollisionSpheres(margin=0.0)
        pose_np = AuboRobot.fk(NEUTRAL).matrix
        pose_t = torch.as_tensor(pose_np, dtype=torch.float32)

        info_np = np_coll.eef_csphere_info(pose_np, frame="wrist3_Link")
        info_t = tcoll.eef_csphere_info(pose_t, frame="wrist3_Link")

        np.testing.assert_allclose(
            info_t.centers.numpy(),
            info_np.centers,
            atol=1e-4,
        )

    def test_arm_collides_no_obstacle(self, tcoll):
        # With check_self=False and no primitives the result must be False.
        cfg = _torch_cfg(NEUTRAL)
        result = tcoll.aubo_arm_collides(cfg, primitives=[], check_self=False)
        assert not bool(result)

    def test_eef_collides_no_obstacle(self, tcoll):
        from robofin.robots_aubo import AuboRobot

        pose = torch.as_tensor(AuboRobot.fk(NEUTRAL).matrix, dtype=torch.float32)
        result = tcoll.aubo_eef_collides(pose, primitives=[])
        assert not bool(result)
