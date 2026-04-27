import logging
from pathlib import Path

import numpy as np
import torch
import trimesh
import urchin

from robofin.point_cloud_tools import transform_point_cloud
from robofin.robot_constants_aubo import AuboConstants
from robofin.torch_urdf import TorchURDF


class AuboSamplerBase:
    def __init__(
        self,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        with_base_link=True,
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        self.woth_base_link = with_base_link
        self.num_robot_points = num_robot_points
        self.num_eef_points = num_eef_points

        if use_cache and self._init_from_cache_():
            return

        robot = urchin.URDF.load(AuboConstants.urdf, lazy_load_meshes=True)
        link_points, link_normals = self._initialize_robot_points(robot, num_robot_points)
        eef_points, eef_normals = self._initialize_eef_points_and_normals(
            robot, num_eef_points
        )
        self.points = {**link_points, **eef_points}
        self.normals = {**link_normals, **eef_normals}

        if use_cache:
            points_to_save = {}
            for key, pc in self.points.items():
                points_to_save[key] = {"pc": pc, "normals": self.normals[key]}
            file_name = self._get_cache_file_name_()
            file_name.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving new file to cache: {file_name}")
            np.save(file_name, points_to_save)

    def _load_link_mesh(self, link):
        """Load a link mesh, preferring visuals and falling back to collisions."""
        mesh_specs = []
        if getattr(link, "visuals", None):
            visual = link.visuals[0].geometry
            if getattr(visual, "mesh", None) is not None:
                mesh_specs.append(visual.mesh.filename)
        for mesh_name in mesh_specs:
            mesh_path = Path(AuboConstants.urdf).parent / mesh_name
            try:
                return trimesh.load(mesh_path, force="mesh")
            except Exception:
                continue
        return None

    def _initialize_eef_points_and_normals(self, robot, N):
        """
        Sample N points from the wrist3_Link mesh, stored under the
        ``eef_wrist3_Link`` key.  These are used exclusively by
        ``sample_end_effector`` so that the EEF point count is exactly N,
        independent of the area-proportional allocation used for full-arm
        sampling.
        """
        eef_link_names = ["wrist3_Link"]
        links = [
            l
            for l in robot.links
            if l.name in set(eef_link_names) and l.visuals
        ]
        points = {}
        normals = {}
        for link in links:
            mesh = self._load_link_mesh(link)
            if mesh is None:
                continue
            link_pc, face_indices = trimesh.sample.sample_surface(mesh, N)
            key = f"eef_{link.name}"
            points[key] = link_pc
            normals[key] = self._init_normals(mesh, link_pc, face_indices)
        return points, normals

    def _initialize_robot_points(self, robot, N):
        arm_link_names = [
            "base_link",
            "shoulder_Link",
            "upperArm_Link",
            "foreArm_Link",
            "wrist1_Link",
            "wrist2_Link",
            "wrist3_Link",
        ]
        links = [l for l in robot.links if l.name in set(arm_link_names) and l.visuals]
        meshes = []
        valid_links = []
        for link in links:
            mesh = self._load_link_mesh(link)
            if mesh is None:
                continue
            meshes.append(mesh)
            valid_links.append(link)

        if not meshes:
            raise RuntimeError(
                "Failed to load any Aubo link mesh for point sampling. "
                "Check URDF mesh paths or install mesh-loading dependencies."
            )

        areas = [m.bounding_box_oriented.area for m in meshes]
        total_area = sum(areas)
        if total_area <= 0:
            raise RuntimeError("Aubo link meshes have zero total area; cannot sample points.")
        num_points = np.round(N * np.array(areas) / total_area).astype(int)
        rounding_error = N - np.sum(num_points)
        while rounding_error > 0:
            num_points[np.random.choice(len(num_points))] += 1
            rounding_error -= 1
        while rounding_error < 0:
            num_points[np.random.choice(len(num_points))] -= 1
            rounding_error += 1

        points = {}
        normals = {}
        for ii, mesh in enumerate(meshes):
            link_pc, face_indices = trimesh.sample.sample_surface(mesh, num_points[ii])
            name = valid_links[ii].name
            points[name] = link_pc
            normals[name] = self._init_normals(mesh, link_pc, face_indices)
        return points, normals

    def _init_normals(self, mesh, pc, face_indices):
        bary = trimesh.triangles.points_to_barycentric(
            triangles=mesh.triangles[face_indices], points=pc
        )
        return trimesh.unitize(
            (
                mesh.vertex_normals[mesh.faces[face_indices]]
                * trimesh.unitize(bary).reshape((-1, 3, 1))
            ).sum(axis=1)
        )

    def _get_cache_file_name_(self):
        return (
            AuboConstants.point_cloud_cache
            / f"aubo_point_cloud_{self.num_robot_points}_{self.num_eef_points}.npy"
        )

    def _init_from_cache_(self):
        file_name = self._get_cache_file_name_()
        if not file_name.is_file():
            return False
        data = np.load(file_name, allow_pickle=True).item()
        # Invalidate old caches that pre-date the separate EEF point cloud.
        if "eef_wrist3_Link" not in data:
            return False
        self.points = {key: v["pc"] for key, v in data.items()}
        self.normals = {key: v["normals"] for key, v in data.items()}
        return True


class TorchAuboSampler(AuboSamplerBase):
    def __init__(
        self,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        with_base_link=True,
        device="cpu",
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        super().__init__(num_robot_points, num_eef_points, use_cache, with_base_link)
        self.robot = TorchURDF.load(
            AuboConstants.urdf, lazy_load_meshes=True, device=device
        )
        self.device = device
        self.points = {
            key: torch.as_tensor(val).unsqueeze(0).to(device)
            for key, val in self.points.items()
        }
        self.normals = {
            key: torch.as_tensor(val).unsqueeze(0).to(device)
            for key, val in self.normals.items()
        }

    # ------------------------------------------------------------------
    # FK helpers
    # ------------------------------------------------------------------

    def end_effector_pose(self, config, frame="wrist3_Link"):
        """
        Return the FK pose of the EEF for a given joint configuration.

        :param config: (6,) or (B, 6) joint angles
        :param frame: EEF frame name; must be in AuboConstants.EEF_LINKS
        :return: (B, 4, 4) pose tensor
        """
        assert frame in AuboConstants.EEF_LINKS.__members__, (
            f"frame must be in {list(AuboConstants.EEF_LINKS.__members__.keys())}"
        )
        if config.ndim == 1:
            config = config.unsqueeze(0)
        fk = self.robot.link_fk_batch(config, use_names=True)
        eef_pose = fk["wrist3_Link"]
        if frame != "wrist3_Link":
            t = torch.as_tensor(
                AuboConstants.EEF_T_LIST[("wrist3_Link", frame)].matrix,
                dtype=eef_pose.dtype,
                device=eef_pose.device,
            )
            eef_pose = eef_pose @ t.unsqueeze(0)
        return eef_pose

    # ------------------------------------------------------------------
    # Full-arm sampling
    # ------------------------------------------------------------------

    def sample(self, config, num_points=None, in_place=True):
        """
        Sample points from the robot surface given a 6-D config.

        :param config: (6,) or (B, 6) joint angles
        :param num_points: number of points to return; None returns all
        :return: (B, N, 4) tensor  [x, y, z, link_idx]
        """
        if config.ndim == 1:
            config = config.unsqueeze(0)
        fk = self.robot.visual_geometry_fk_batch(config, use_names=True)
        fk_points = []
        for link_name, link_idx in AuboConstants.ARM_VISUAL_LINKS.__members__.items():
            if link_name not in self.points:
                continue
            pc = transform_point_cloud(
                self.points[link_name].float().repeat((fk[link_name].shape[0], 1, 1)),
                fk[link_name],
                in_place=in_place,
            )
            fk_points.append(
                torch.cat(
                    (
                        pc,
                        link_idx * torch.ones((pc.size(0), pc.size(1), 1)).type_as(pc),
                    ),
                    dim=-1,
                )
            )
        pc = torch.cat(fk_points, dim=1)
        if num_points is None:
            return pc
        random_idxs = np.random.choice(pc.shape[1], num_points, replace=False)
        return pc[:, random_idxs, :]

    def sample_from_poses(self, poses, num_points=None, in_place=True):
        """
        Build a point cloud from pre-computed FK pose matrices.

        :param poses: (B, N_links, 4, 4) — output of aubo_arm_visual_fk or
                      TorchURDF.visual_geometry_fk_batch stacked along dim 1
        :param num_points: number of points to sub-sample; None returns all
        :return: (B, N, 4) tensor  [x, y, z, link_idx]
        """
        fk_points = []
        for link_name, link_idx in AuboConstants.ARM_VISUAL_LINKS.__members__.items():
            if link_name not in self.points:
                continue
            pc = transform_point_cloud(
                self.points[link_name].float().repeat((poses.shape[0], 1, 1)),
                poses[:, link_idx],
                in_place=in_place,
            )
            fk_points.append(
                torch.cat(
                    (
                        pc,
                        link_idx * torch.ones((pc.size(0), pc.size(1), 1)).type_as(pc),
                    ),
                    dim=-1,
                )
            )
        pc = torch.cat(fk_points, dim=1)
        if num_points is None:
            return pc
        random_idxs = np.random.choice(pc.shape[1], num_points, replace=False)
        return pc[:, random_idxs, :]

    # ------------------------------------------------------------------
    # EEF sampling (internal)
    # ------------------------------------------------------------------

    def _sample_end_effector(
        self,
        with_normals,
        poses,
        num_points=None,
        frame="wrist3_Link",
        in_place=True,
    ):
        assert frame in AuboConstants.EEF_LINKS.__members__, (
            f"frame must be in {list(AuboConstants.EEF_LINKS.__members__.keys())}"
        )
        if poses.ndim == 2:
            poses = poses.unsqueeze(0)

        # Convert from the requested frame to wrist3_Link frame:
        #   T_world_wrist3 = T_world_frame @ T_frame_wrist3
        #                  = poses @ inv(T_wrist3_frame)
        if frame != "wrist3_Link":
            t = torch.as_tensor(
                AuboConstants.EEF_T_LIST[("wrist3_Link", frame)].inverse.matrix,
                dtype=poses.dtype,
                device=poses.device,
            )
            poses = poses @ t.unsqueeze(0)

        # Compute visual FK at the zero configuration so we know where
        # wrist3_Link's visual geometry sits relative to its link frame.
        default_cfg = torch.zeros((1, 6), device=poses.device)
        visual_fk = self.robot.visual_geometry_fk_batch(default_cfg, use_names=True)
        link_fk = self.robot.link_fk_batch(default_cfg, use_names=True)

        # Inverse of wrist3_Link at zero config (link FK, not visual FK, because
        # the input `poses` represents a link-frame pose).
        wrist3_at_zero = link_fk["wrist3_Link"]  # (1, 4, 4)
        wrist3_T_world = torch.zeros_like(wrist3_at_zero)
        wrist3_T_world[:, -1, -1] = 1
        wrist3_T_world[:, :3, :3] = wrist3_at_zero[:, :3, :3].transpose(1, 2)
        wrist3_T_world[:, :3, -1] = -torch.matmul(
            wrist3_T_world[:, :3, :3], wrist3_at_zero[:, :3, -1].unsqueeze(-1)
        ).squeeze(-1)

        # eef_transform brings the zero-config visual frame to the desired pose.
        eef_transform = poses @ wrist3_T_world  # (B, 4, 4)

        eef_key = "eef_wrist3_Link"
        link_idx = AuboConstants.EEF_VISUAL_LINKS["wrist3_Link"]
        B = poses.shape[0]

        pc = transform_point_cloud(
            self.points[eef_key].float().repeat((B, 1, 1)),
            eef_transform @ visual_fk["wrist3_Link"],
            in_place=in_place,
        )
        pc = torch.cat(
            (pc, link_idx * torch.ones((B, pc.size(1), 1)).type_as(pc)),
            dim=-1,
        )

        if with_normals:
            normals = transform_point_cloud(
                self.normals[eef_key].float().repeat((B, 1, 1)),
                eef_transform @ visual_fk["wrist3_Link"],
                vector=True,
                in_place=in_place,
            )
            normals = torch.cat(
                (normals, link_idx * torch.ones((B, normals.size(1), 1)).type_as(normals)),
                dim=-1,
            )

        if num_points is None:
            if with_normals:
                return pc, normals
            return pc
        sample_idxs = np.random.choice(pc.shape[1], num_points, replace=False)
        if with_normals:
            return pc[:, sample_idxs, :], normals[:, sample_idxs, :]
        return pc[:, sample_idxs, :]

    # ------------------------------------------------------------------
    # EEF sampling (public)
    # ------------------------------------------------------------------

    def sample_end_effector(
        self, poses, num_points=None, frame="wrist3_Link", in_place=True
    ):
        """
        Sample points from the EEF surface given desired EEF pose(s).

        :param poses: (4, 4) or (B, 4, 4) desired pose in ``frame``
        :param num_points: number of points to return; None returns all
        :param frame: EEF frame; must be in AuboConstants.EEF_LINKS
        :return: (B, N, 4) tensor  [x, y, z, link_idx]
        """
        return self._sample_end_effector(
            with_normals=False,
            poses=poses,
            num_points=num_points,
            frame=frame,
            in_place=in_place,
        )

    def sample_end_effector_with_normals(
        self, poses, num_points=None, frame="wrist3_Link"
    ):
        """
        Like ``sample_end_effector`` but also returns surface normals.

        :return: tuple ((B, N, 4) points, (B, N, 4) normals)
        """
        return self._sample_end_effector(
            with_normals=True,
            poses=poses,
            num_points=num_points,
            frame=frame,
        )
