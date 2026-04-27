import numpy as np
import pybullet as p
from geometrout import SE3

from ikfast_aubo_i3H import get_fk, get_ik


from robofin.robot_constants_aubo import AuboConstants


class AuboRobot:
    constants = AuboConstants

    @classmethod
    def within_limits(cls, config):
        # We have to add a small buffer because of float math
        return np.all(config >= cls.constants.JOINT_LIMITS[:, 0] - 1e-5) and np.all(
            config <= cls.constants.JOINT_LIMITS[:, 1] + 1e-5
        )

    @classmethod
    def random_neutral(cls, method="normal"):
        if method == "normal":
            return np.clip(
                cls.constants.NEUTRAL + np.random.normal(0, 0.25, 6),
                cls.constants.JOINT_LIMITS[:, 0],
                cls.constants.JOINT_LIMITS[:, 1],
            )
        if method == "uniform":
            return cls.constants.NEUTRAL + np.random.uniform(0, 0.25, 6)
        assert False, "method must be either normal or uniform"

    @classmethod
    def fk(cls, config, eff_frame="wrist3_Link"):
        """Returns SE3 pose of the end effector (wrist3_Link) given a 6D config.
            get_fk eff_frame  now is only use wrist3_Link
        """
        assert (
            eff_frame in cls.constants.EEF_LINKS.__members__
        ), "Default FK only calculated for a valid end effector frame"

        pos, rot = get_fk(config)
        mat =np.eye(4)
        mat[:3, :3] = np.asarray(rot)
        mat[:3, 3] = np.asarray(pos)
        return SE3.from_matrix(mat)

    @classmethod
    def ik(cls, pose, joint_range_scalar=1.0, eff_frame="wrist3_Link"):
        """
        Returns a list of valid 6D configs using ikfast_aubo_i3H.
        ikfast returns angles in [0, 2π); we shift each joint into its limit range.
        """
        joint_limits = cls.constants.JOINT_LIMITS * joint_range_scalar
        assert (
            eff_frame in cls.constants.EEF_LINKS.__members__
        ), "Default IK only calculated for a valid end effector frame"
        
        rot = pose.so3.matrix.tolist()
        pos = pose.xyz.tolist()
        solutions =[np.asarray(sol) for sol in get_ik(pos, rot, [])]
        return [
            s
            for s in solutions
            if (np.all(s >= joint_limits[:, 0]) and np.all(s <= joint_limits[:, 1]))
        ]
    
    @classmethod
    def random_configuration(cls, joint_range_scalar=1.0):
        limits = cls.constants.JOINT_LIMITS * joint_range_scalar
        return (limits[:, 1] - limits[:, 0]) * np.random.rand(6) + limits[:, 0]

    @classmethod
    def random_ik(cls, pose, eff_frame="wrist3_Link", joint_range_scalar=1.0):
        try:
            return cls.ik(pose, joint_range_scalar=joint_range_scalar, eff_frame=eff_frame)
        except Exception as e:
            raise ValueError(f"IK failed for pose {pose} with error {e}")

    @classmethod
    def collision_free_ik(
        cls,
        pose,
        cooo,
        primitive_arrays,
        eff_frame="wrist3_Link",
        scene_buffer=0.0,
        self_collision_buffer=0.0,
        joint_range_scalar=1.0,
        retries=1000,
        bad_state_callback=lambda x: False,
        choose_close_to=None,
    ):
        # Aubo IKFast has no free joint; it returns the complete discrete solution
        # set for a pose, so retrying just rechecks the same candidates. Keep the
        # signature for API parity with Franka but evaluate each IK solution once.
        _ = retries
        samples = list(
            cls.random_ik(
                pose,
                eff_frame=eff_frame,
                joint_range_scalar=joint_range_scalar,
            )
        )
        if len(samples) == 0:
            return None
        if choose_close_to is None:
            order = np.random.permutation(len(samples))
        else:
            order = np.argsort(
                [np.linalg.norm(sample - choose_close_to) for sample in samples]
            )
        for idx in order:
            sample = samples[int(idx)]
            if not (
                cooo.aubo_arm_collides_fast(
                    sample,
                    primitive_arrays,
                    scene_buffer=scene_buffer,
                    self_collision_buffer=self_collision_buffer,
                )
                or bad_state_callback(sample)
            ):
                return sample
        return None
