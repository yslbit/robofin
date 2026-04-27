from collections import namedtuple

import numpy as np
import torch
import urchin
from geometrout import SE3, Sphere
from geometrout.maths import transform_in_place

from robofin.robot_constants_aubo import AuboConstants
from robofin.kinematics.numba_aubo import aubo_arm_link_fk as numba_aubo_arm_link_fk
from robofin.kinematics.torch_aubo import aubo_arm_link_fk as torch_aubo_arm_link_fk

SphereInfo = namedtuple("SphereInfo", "radii centers")


class AuboCollisionSpheres:
    def __init__(self, margin=0.0):
        self._init_collision_spheres(margin)
        self._init_self_collision_spheres()

    def _init_collision_spheres(self, margin):
        spheres = {}
        for r, centers in AuboConstants.SPHERES:
            for k, c in centers.items():
                spheres[k] = spheres.get(k, [])
                spheres[k].append(
                    SphereInfo(radii=r * np.ones((c.shape[0])) + margin, centers=c)
                )
        self.cspheres = {}
        for k, v in spheres.items():
            radii = np.concatenate([ix.radii for ix in v])
            radii.setflags(write=False)
            centers = np.concatenate([ix.centers for ix in v])
            centers.setflags(write=False)
            self.cspheres[k] = SphereInfo(radii=radii, centers=centers)

    def _init_self_collision_spheres(self):
        link_names = []
        centers = {}
        for s in AuboConstants.SELF_COLLISION_SPHERES:
            if s[0] not in centers:
                link_names.append(s[0])
                centers[s[0]] = [s[1]]
            else:
                centers[s[0]].append(s[1])
        self.sc_points = [(name, np.asarray(centers[name])) for name in link_names]

        self.collision_matrix = -np.inf * np.ones(
            (
                len(AuboConstants.SELF_COLLISION_SPHERES),
                len(AuboConstants.SELF_COLLISION_SPHERES),
            )
        )
        link_ids = {link_name: idx for idx, link_name in enumerate(link_names)}
        for idx1, (link_name1, center1, radius1) in enumerate(
            AuboConstants.SELF_COLLISION_SPHERES
        ):
            for idx2, (link_name2, center2, radius2) in enumerate(
                AuboConstants.SELF_COLLISION_SPHERES
            ):
                # Skip pairs that are 0, 1, or 2 link-hops apart.
                # Aubo i3H's compact geometry causes false positives at
                # threshold < 2 (e.g. base_link vs upperArm_Link at neutral).
                if abs(link_ids[link_name1] - link_ids[link_name2]) < 3:
                    continue
                self.collision_matrix[idx1, idx2] = radius1 + radius2

    def _get_fk(self, config):
        poses = numba_aubo_arm_link_fk(np.asarray(config, dtype=np.double), np.eye(4))
        return poses

    def has_self_collision(self, config, buffer=0.0):
        fk = self._get_fk(config)
        fk_points = []
        for link_name, centers in self.sc_points:
            mat = fk[AuboConstants.ARM_LINKS[link_name]]
            pts = np.copy(centers)
            transform_in_place(pts, mat)
            fk_points.append(pts)
        transformed_centers = np.concatenate(fk_points, axis=0)
        points_matrix = np.tile(
            transformed_centers, (transformed_centers.shape[0], 1, 1)
        )
        distances = np.linalg.norm(
            points_matrix - points_matrix.transpose((1, 0, 2)), axis=2
        )
        return np.any(distances < self.collision_matrix + buffer)

    def csphere_info(self, config, with_base_link=False):
        fk = self._get_fk(config)
        radii = []
        centers = []
        for link_name, info in self.cspheres.items():
            if not with_base_link and link_name == "base_Link":
                continue
            mat = fk[AuboConstants.ARM_LINKS[link_name]]
            pts = transform_in_place(np.copy(info.centers), mat)
            centers.append(pts)
            radii.append(info.radii)
        return SphereInfo(radii=np.concatenate(radii), centers=np.concatenate(centers))
    
    def collision_spheres(self, config, with_base_link=False):
        info = self.csphere_info(config, with_base_link=with_base_link)
        return [
            Sphere(c,r) for c, r in zip(info.centers, info.radii)
        ]
        
    def eef_csphere_info(self, pose, frame="wrist3_Link"):
        """
        Return collision sphere info for the EEF (wrist3_Link) given a desired pose.

        :param pose: SE3 or (4, 4) np.ndarray pose expressed in ``frame``
        :param frame: EEF frame name; must be a member of AuboConstants.EEF_LINKS
        :return: SphereInfo with world-frame sphere centers and radii
        """
        assert frame in AuboConstants.EEF_LINKS.__members__, (
            f"frame must be in {list(AuboConstants.EEF_LINKS.__members__.keys())}"
        )
        if isinstance(pose, SE3):
            pose = pose.matrix
        # Convert pose from eff_frame to wrist3_Link frame:
        # T_world_wrist3 = T_world_frame @ T_frame_wrist3
        #                = pose @ T_frame_wrist3
        if frame != "wrist3_Link":
            t = AuboConstants.EEF_T_LIST[("wrist3_Link", frame)]
            pose = pose @ t.inverse.matrix
        info = self.cspheres["wrist3_Link"]
        centers = transform_in_place(np.copy(info.centers), pose)
        return SphereInfo(radii=info.radii.copy(), centers=centers)
    
    def eef_collision_spheres(self, pose, frame="wrist3_Link"):
        info = self.eef_csphere_info(pose, frame)
        return [
            Sphere(c,r) for c, r in zip(info.centers, info.radii)
        ]
        
    def aubo_arm_collides(
        self,
        q,
        primitives,
        *,
        scene_buffer=0.0,
        self_collision_buffer=0.0,
        check_self=True,
        with_base_link=False,
    ):
        if check_self and self.has_self_collision(q, self_collision_buffer):
            return True
        cspheres = self.csphere_info(q, with_base_link=with_base_link)
        for p in primitives:
            if np.any(p.sdf(cspheres.centers) < cspheres.radii + scene_buffer):
                return True
        return False
    
    def aubo_arm_collides_fast(
        self,
        q,
        primitive_arrays,
        *,
        scene_buffer=0.0,
        self_collision_buffer=0.0,
        check_self=True,
        with_base_link=False,
    ):
        if check_self and self.has_self_collision(q, self_collision_buffer):
            return True
        cspheres = self.csphere_info(q, with_base_link=with_base_link)
        for arr in primitive_arrays:
            if np.any(arr.scene_sdf(cspheres.centers) < cspheres.radii + scene_buffer):
                return True
        return False

    def aubo_eef_collides(
        self, pose, primitives, frame="wrist3_Link", scene_buffer=0.0
    ):
        cspheres = self.eef_csphere_info(pose, frame)
        for p in primitives:
            if np.any(p.sdf(cspheres.centers) < cspheres.radii + scene_buffer):
                return True
        return False

    def aubo_eef_collides_fast(
        self, pose, primitive_arrays, frame="wrist3_Link", scene_buffer=0.0
    ):
        cspheres = self.eef_csphere_info(pose, frame)
        for arr in primitive_arrays:
            if np.any(arr.scene_sdf(cspheres.centers) < cspheres.radii + scene_buffer):
                return True
        return False


class TorchAuboCollisionSpheres:
    """
    Batched, GPU-compatible collision sphere checker for the Aubo i3H.

    Mirrors the Franka ``TorchFrankaCollisionSpheres`` interface but omits the
    ``prismatic_joint`` argument everywhere (AUBO has no gripper joints).
    """

    def __init__(self, margin=0.0, device="cpu"):
        self._device = device
        self._init_collision_spheres(margin, device)
        self._init_self_collision_spheres(device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _transform_in_place(point_cloud, transformation_matrix):
        """Transform (B, N, 3) points by (B, 4, 4) matrices in-place."""
        pc_T = torch.transpose(point_cloud, -2, -1)  # (B, 3, N)
        ones_shape = list(pc_T.shape)
        ones_shape[-2] = 1
        ones = torch.ones(ones_shape).type_as(point_cloud)
        hom = torch.cat((pc_T, ones), dim=-2)  # (B, 4, N)
        out = torch.matmul(transformation_matrix, hom)  # (B, 4, N)
        point_cloud[..., :3] = torch.transpose(out[..., :3, :], -2, -1)
        return point_cloud

    def _init_collision_spheres(self, margin, device):
        spheres = {}
        for r, centers in AuboConstants.SPHERES:
            for k, c in centers.items():
                spheres[k] = spheres.get(k, [])
                spheres[k].append(
                    SphereInfo(
                        radii=r * torch.ones(c.shape[0], device=device) + margin,
                        centers=torch.as_tensor(c, dtype=torch.float32, device=device),
                    )
                )
        self.cspheres = {}
        for k, v in spheres.items():
            self.cspheres[k] = SphereInfo(
                radii=torch.cat([ix.radii for ix in v]),
                centers=torch.cat([ix.centers for ix in v]),
            )

    def _init_self_collision_spheres(self, device):
        link_names = []
        centers = {}
        for link_name, center, _ in AuboConstants.SELF_COLLISION_SPHERES:
            if link_name not in centers:
                link_names.append(link_name)
                centers[link_name] = [
                    torch.as_tensor(center, dtype=torch.float32, device=device)
                ]
            else:
                centers[link_name].append(
                    torch.as_tensor(center, dtype=torch.float32, device=device)
                )
        self.sc_points = [
            (name, torch.stack(centers[name])) for name in link_names
        ]
        self.collision_matrix = -np.inf * torch.ones(
            (
                len(AuboConstants.SELF_COLLISION_SPHERES),
                len(AuboConstants.SELF_COLLISION_SPHERES),
            ),
            device=device,
        )
        link_ids = {name: idx for idx, name in enumerate(link_names)}
        for idx1, (ln1, _, r1) in enumerate(AuboConstants.SELF_COLLISION_SPHERES):
            for idx2, (ln2, _, r2) in enumerate(AuboConstants.SELF_COLLISION_SPHERES):
                if abs(link_ids[ln1] - link_ids[ln2]) < 3:
                    continue
                self.collision_matrix[idx1, idx2] = r1 + r2

    # ------------------------------------------------------------------
    # FK helper
    # ------------------------------------------------------------------

    def _arm_link_fk(self, config, base_pose=None):
        """
        :param config: (B, 6) joint angles
        :param base_pose: (4, 4) or (B, 4, 4) base transform; defaults to identity
        :return: (B, 7, 4, 4)
        """

        if base_pose is None:
            base_pose = torch.eye(4, dtype=config.dtype, device=config.device)
        return torch_aubo_arm_link_fk(config, base_pose)  # (B, 7, 4, 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_self_collision(self, config, buffer=0.0):
        """
        :param config: (6,) or (B, 6)
        :param buffer: extra clearance added to sphere radii sum threshold
        :return: bool or (B,) bool tensor
        """
        squeeze = False
        if config.ndim == 1:
            config = config.unsqueeze(0)
            squeeze = True
        B = config.size(0)
        fk = self._arm_link_fk(config)  # (B, 7, 4, 4)
        fk_points = []
        for link_name, centers in self.sc_points:
            link_idx = AuboConstants.ARM_LINKS[link_name]
            fk_points.append(
                self._transform_in_place(
                    centers[None].expand(B, -1, -1).clone().type_as(fk),
                    fk[:, link_idx],
                )
            )
        transformed = torch.cat(fk_points, dim=1)  # (B, N, 3)
        t = transformed.unsqueeze(2)  # (B, N, 1, 3)
        s = transformed.unsqueeze(1)  # (B, 1, N, 3)
        distances = torch.linalg.norm(t - s, dim=3)  # (B, N, N)
        self_collisions = torch.any(
            (distances < self.collision_matrix[None] + buffer).reshape(B, -1),
            dim=1,
        )
        if squeeze:
            return self_collisions.squeeze(0)
        return self_collisions

    def csphere_info(self, config, base_pose=None):
        """
        :param config: (6,) or (B, 6)
        :param base_pose: optional (4, 4) or (B, 4, 4) base transform
        :return: SphereInfo with (B, N) radii and (B, N, 3) centers;
                 batch dim squeezed when input was 1-D
        """
        squeeze = False
        if config.ndim == 1:
            config = config.unsqueeze(0)
            squeeze = True
        B = config.size(0)
        fk = self._arm_link_fk(config, base_pose)  # (B, 7, 4, 4)
        radii_list = []
        centers_list = []
        for link_name, info in self.cspheres.items():
            link_idx = AuboConstants.ARM_LINKS[link_name]
            centers_list.append(
                self._transform_in_place(
                    info.centers[None].expand(B, -1, -1).clone().type_as(fk),
                    fk[:, link_idx],
                )
            )
            radii_list.append(info.radii[None].expand(B, -1))
        radii = torch.cat(radii_list, dim=1)
        centers = torch.cat(centers_list, dim=1)
        if squeeze:
            return SphereInfo(radii=radii.squeeze(0), centers=centers.squeeze(0))
        return SphereInfo(radii=radii, centers=centers)

    def eef_csphere_info(self, pose, frame="wrist3_Link"):
        """
        Collision spheres for the EEF at a given pose.

        :param pose: (4, 4) or (B, 4, 4) pose in ``frame``
        :param frame: must be in AuboConstants.EEF_LINKS
        :return: SphereInfo; batch dim squeezed when input was (4, 4)
        """
        assert frame in AuboConstants.EEF_LINKS.__members__, (
            f"frame must be in {list(AuboConstants.EEF_LINKS.__members__.keys())}"
        )
        squeeze = False
        if pose.ndim == 2:
            pose = pose.unsqueeze(0)
            squeeze = True
        if frame != "wrist3_Link":
            t = torch.as_tensor(
                AuboConstants.EEF_T_LIST[(frame, "wrist3_Link", )].matrix,
                dtype=torch.float32,
                device=pose.device,
            )
            pose = pose @ t.unsqueeze(0)
        B = pose.size(0)
        info = self.cspheres["wrist3_Link"]
        centers = self._transform_in_place(
            info.centers[None].expand(B, -1, -1).clone().type_as(pose),
            pose,
        )
        radii = info.radii[None].expand(B, -1)
        if squeeze:
            return SphereInfo(radii=radii.squeeze(0), centers=centers.squeeze(0))
        return SphereInfo(radii=radii, centers=centers)

    def aubo_arm_collides(
        self,
        q,
        primitives,
        *,
        scene_buffer=0.0,
        self_collision_buffer=0.0,
        check_self=True,
    ):
        """
        Batched arm collision check against geometric primitives.

        :param q: (6,) or (B, 6) joint configurations
        :param primitives: single primitive or list thereof (must expose .sdf())
        :return: bool or (B,) bool tensor
        """
        if not isinstance(primitives, list):
            primitives = [primitives]
        squeeze = False
        if q.ndim == 1:
            q = q.unsqueeze(0)
            squeeze = True
        collisions = torch.zeros(q.size(0), dtype=torch.bool, device=q.device)
        if check_self:
            collisions = torch.logical_or(
                self.has_self_collision(q, self_collision_buffer), collisions
            )
        cspheres = self.csphere_info(q)
        for p in primitives:
            p_collisions = torch.any(
                p.sdf(cspheres.centers) < cspheres.radii + scene_buffer, dim=1
            )
            collisions = torch.logical_or(p_collisions, collisions)
        if squeeze:
            return collisions.squeeze(0)
        return collisions
    
    def aubo_arm_collides_sequence(
        self,
        q,
        primitives,
        *,
        scene_buffer=0.0,
        self_collision_buffer=0.0,
        check_self=True,
        with_base_link=False,
    ):
        """
        Sequence method works with squences if configurations [B,T,6].
        """
        if not isinstance(primitives, list):
            primitives = [primitives]
        assert q.ndim == 3, "Input must be (B, T, 6)"
        B, T, _ = q.shape
        collisions = torch.zeros(B, T, dtype=torch.bool, device=q.device)
        flat_q = q.view(-1, 6)
        if check_self:
            collisions = torch.logical_or(
                self.has_self_collision(flat_q, self_collision_buffer).view(B, T),
                collisions,
            )
        cspheres = self.csphere_info(flat_q, with_base_link=with_base_link)
        centers = cspheres.centers.view(B, T, cspheres.centers.size(1), cspheres.centers.size(2))
        radii = cspheres.radii.unsqueeze(0)
        
        for p in primitives:
            p_collisions = torch.any(
                p.sdf_sequence(centers) < radii + scene_buffer,
                dim=2,
            )
            collisions = torch.logical_or(p_collisions, collisions)
        
        return collisions

    def aubo_eef_collides(
        self, pose, primitives, frame="wrist3_Link", scene_buffer=0.0
    ):
        """
        Batched EEF collision check against geometric primitives.

        :param pose: (4, 4) or (B, 4, 4) EEF pose in ``frame``
        :param primitives: single primitive or list thereof (must expose .sdf())
        :return: bool or (B,) bool tensor
        """
        if not isinstance(primitives, list):
            primitives = [primitives]
        squeeze = False
        if pose.ndim == 2:
            pose = pose.unsqueeze(0)
            squeeze = True
        collisions = torch.zeros(pose.size(0), dtype=torch.bool, device=pose.device)
        cspheres = self.eef_csphere_info(pose, frame)
        for p in primitives:
            p_collisions = torch.any(
                p.sdf(cspheres.centers) < cspheres.radii + scene_buffer, dim=1
            )
            collisions = torch.logical_or(p_collisions, collisions)
        if squeeze:
            return collisions.squeeze(0)
        return collisions
