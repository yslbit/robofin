from enum import IntEnum
from pathlib import Path

import numpy as np
from geometrout import SE3


class _AuboConstants:
    @property
    def urdf(cls):
        return cls._URDF

    @property
    def point_cloud_cache(cls):
        return cls._POINT_CLOUD_CACHE

    @property
    def JOINT_LIMITS(cls):
        return cls._JOINT_LIMITS

    @property
    def VELOCITY_LIMIT(cls):
        return cls._VELOCITY_LIMIT

    @property
    def ACCELERATION_LIMIT(cls):
        return cls._ACCELERATION_LIMIT

    @property
    def EEF_T_LIST(cls):
        return cls._EEF_T_LIST

    @property
    def DOF(cls):
        return cls._DOF

    @property
    def NEUTRAL(cls):
        return cls._NEUTRAL

    @property
    def SPHERES(cls):
        return cls._SPHERES

    @property
    def SELF_COLLISION_SPHERES(cls):
        return cls._SELF_COLLISION_SPHERES

    @property
    def EEF_LINKS(cls):
        return cls._EEF_LINKS

    @property
    def EEF_VISUAL_LINKS(cls):
        return cls._EEF_VISUAL_LINKS

    @property
    def ARM_LINKS(cls):
        return cls._ARM_LINKS

    @property
    def ARM_VISUAL_LINKS(cls):
        return cls._ARM_VISUAL_LINKS

    _URDF = str(
        Path(__file__).parent / "urdf" / "aubo_i3H" / "aubo_i3H.urdf"
    )

    _POINT_CLOUD_CACHE = Path(__file__).parent / "cache" / "point_cloud" / "aubo_i3H"

    # Joint order: shoulder, upperArm, foreArm, wrist1, wrist2, wrist3
    _JOINT_LIMITS = np.array(
        [
            (-6.283185307179586, 6.283185307179586),   # shoulder_joint
            (-3.054326190990076, 3.054326190990076),   # upperArm_joint
            (-3.054326190990076, 3.054326190990076),   # foreArm_joint
            (-3.054326190990076, 3.054326190990076),   # wrist1_joint
            (-3.054326190990076, 3.054326190990076),   # wrist2_joint
            (-6.283185307179586, 6.283185307179586),   # wrist3_joint
        ]
    )

    _VELOCITY_LIMIT = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])

    _ACCELERATION_LIMIT = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

    _DOF = 6

    # Identity transform: wrist3_Link is the EEF frame (T:2->1)
    _EEF_T_LIST = {
        ("wrist3_Link", "wrist3_Link"): SE3(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
    }

    _NEUTRAL = np.array([-0.03, -0.46, 1.5, 0.92, 0.2, 0.0])

    # Approximate collision spheres for each link (radius, {link_name: centers})
    # Centers are in the local link frame, units in meters
    # Synchronized with urdf/aubo_i3H/aubo_i3H.yaml collision_spheres
    _SPHERES = [
        # base_link intentionally excluded: the robot is mounted on the table,
        # so base-link spheres always overlap with the mount table obstacle.
        (
            0.07804,
            {
                "shoulder_Link": np.array([[0.0, -0.001, -0.005]]),
            },
        ),
        (
            0.06868,
            {
                "upperArm_Link": np.array([[0.0, -0.0, -0.007]]),
            },
        ),
        (
            0.06253,
            {
                "upperArm_Link": np.array([[0.111, -0.0, 0.008]]),
            },
        ),
        (
            0.05484,
            {
                "upperArm_Link": np.array([[0.175, -0.0, 0.008]]),
            },
        ),
        (
            0.05561,
            {
                "upperArm_Link": np.array([[0.273, -0.002, -0.027]]),
            },
        ),
        (
            0.06561,
            {
                "upperArm_Link": np.array([[0.001, 0.003, 0.021]]),
            },
        ),
        (
            0.0533,
            {
                "upperArm_Link": np.array([[0.09, -0.002, 0.007]]),
            },
        ),
        (
            0.06253,
            {
                "upperArm_Link": np.array([[0.267, -0.0, 0.002]]),
            },
        ),
        (
            0.06176,
            {
                "upperArm_Link": np.array([[0.005, 0.005, -0.023]]),
            },
        ),
        (
            0.05407,
            {
                "upperArm_Link": np.array([[0.146, -0.001, 0.007]]),
            },
        ),
        (
            0.05715,
            {
                "upperArm_Link": np.array([[0.006, -0.01, 0.034]]),
            },
        ),
        (
            0.04638,
            {
                "upperArm_Link": np.array([[0.197, -0.008, 0.005]]),
            },
        ),
        (
            0.04715,
            {
                "upperArm_Link": np.array([[0.252, -0.006, -0.035]]),
            },
        ),
        (
            0.04946,
            {
                "upperArm_Link": np.array([[-0.014, -0.014, -0.035]]),
            },
        ),
        (
            0.05773,
            {
                "foreArm_Link": np.array([[0.257, -0.0, 0.096]]),
            },
        ),
        (
            0.05701,
            {
                "foreArm_Link": np.array(
                    [
                        [0.044, -0.001, 0.102],
                        [0.123, -0.0, 0.103],
                        [0.179, 0.0, 0.103],
                    ]
                ),
            },
        ),
        (
            0.05773,
            {
                "foreArm_Link": np.array([[0.257, -0.0, 0.122]]),
            },
        ),
        (
            0.05838,
            {
                "wrist1_Link": np.array([[0.0, 0.014, -0.0], [0.0, -0.003, -0.0]]),
                "wrist2_Link": np.array([[-0.0, 0.003, -0.0], [0.0, -0.01, -0.0]]),
            },
        ),
        (
            0.05645,
            {
                "wrist1_Link": np.array([[-0.001, -0.023, -0.002]]),
                "wrist2_Link": np.array([[0.001, 0.023, -0.002]]),
            },
        ),
        (
            0.05516,
            {
                "wrist1_Link": np.array([[0.003, 0.025, -0.001]]),
                "wrist2_Link": np.array([[-0.003, -0.025, -0.001]]),
            },
        ),
        (
            0.05258,
            {
                "wrist2_Link": np.array([[0.005, -0.028, 0.002]]),
            },
        ),
        (
            0.04913,
            {
                "wrist3_Link": np.array([[-0.001, -0.0, -0.019]]),
            },
        ),
    ]

    # Self-collision spheres: (link_name, center, radius)
    # Synchronized with urdf/aubo_i3H/aubo_i3H.yaml collision_spheres
    _SELF_COLLISION_SPHERES = [
        ("base_link", [0.0, 0.0, 0.08], 0.08),
        ("shoulder_Link", [0.0, -0.001, -0.005], 0.07804),
        ("shoulder_Link", [0.0, 0.06, 0.0], 0.07),
        ("upperArm_Link", [0.0, -0.0, -0.007], 0.06868),
        ("upperArm_Link", [0.111, -0.0, 0.008], 0.06253),
        ("upperArm_Link", [0.266, 0.001, 0.028], 0.06253),
        ("upperArm_Link", [0.175, -0.0, 0.008], 0.05484),
        ("upperArm_Link", [0.273, -0.002, -0.027], 0.05561),
        ("upperArm_Link", [0.001, 0.003, 0.021], 0.06561),
        ("upperArm_Link", [0.09, -0.002, 0.007], 0.0533),
        ("upperArm_Link", [0.267, -0.0, 0.002], 0.06253),
        ("upperArm_Link", [0.005, 0.005, -0.023], 0.06176),
        ("upperArm_Link", [0.146, -0.001, 0.007], 0.05407),
        ("upperArm_Link", [0.006, -0.01, 0.034], 0.05715),
        ("upperArm_Link", [0.197, -0.008, 0.005], 0.04638),
        ("upperArm_Link", [0.252, -0.006, -0.035], 0.04715),
        ("upperArm_Link", [-0.014, -0.014, -0.035], 0.04946),
        ("foreArm_Link", [0.257, -0.0, 0.096], 0.05773),
        ("foreArm_Link", [0.044, -0.001, 0.102], 0.05701),
        ("foreArm_Link", [0.123, -0.0, 0.103], 0.05701),
        ("foreArm_Link", [0.179, 0.0, 0.103], 0.05701),
        ("foreArm_Link", [0.257, -0.0, 0.122], 0.05773),
        ("wrist1_Link", [0.0, 0.014, -0.0], 0.05838),
        ("wrist1_Link", [-0.001, -0.023, -0.002], 0.05645),
        ("wrist1_Link", [0.003, 0.025, -0.001], 0.05516),
        ("wrist1_Link", [0.0, -0.003, -0.0], 0.05838),
        ("wrist2_Link", [0.0, -0.01, -0.0], 0.05838),
        ("wrist2_Link", [0.001, 0.023, -0.002], 0.05645),
        ("wrist2_Link", [-0.003, -0.025, -0.001], 0.05516),
        ("wrist2_Link", [0.005, -0.028, 0.002], 0.05258),
        ("wrist2_Link", [-0.0, 0.003, -0.0], 0.05838),
        ("wrist3_Link", [-0.001, -0.0, -0.019], 0.04913),
    ]

    _EEF_LINKS = IntEnum(
        "AuboEefLinks",
        ["wrist3_Link"],
        start=0,
    )

    _EEF_VISUAL_LINKS = IntEnum(
        "AuboEefVisuals",
        ["wrist3_Link"],
        start=0,
    )

    _ARM_VISUAL_LINKS = IntEnum(
        "AuboArmVisuals",
        [
            "base_link",
            "shoulder_Link",
            "upperArm_Link",
            "foreArm_Link",
            "wrist1_Link",
            "wrist2_Link",
            "wrist3_Link",
        ],
        start=0,
    )

    _ARM_LINKS = IntEnum(
        "AuboArmLinks",
        [
            "base_link",
            "shoulder_Link",
            "upperArm_Link",
            "foreArm_Link",
            "wrist1_Link",
            "wrist2_Link",
            "wrist3_Link",
        ],
        start=0,
    )


AuboConstants = _AuboConstants()
