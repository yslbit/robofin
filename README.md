# robofin

A collection of robotics tools tailored to machine learning, with a focus on fast geometry, kinematics, collision checks, and sampling utilities.

## Repository Relationship

- Original repository (upstream): robofin (original project)
- Current repository (fork): https://github.com/fishbotics/robofin
- This fork is based on the original repository and adds Aubo robot support (including Aubo i3H).

## What Is Included

- Robot models and constants
- Kinematics backends (Numba and Torch)
- Collision utilities and geometry tools
- Point-cloud sampling and related utilities
- URDF assets for supported robots

## Supported Robots

- Franka Panda (existing support)
- Aubo i3H (newly added in this fork)

## Aubo i3H Update In This Fork

This fork adds Aubo i3H support across:

- robot constants and robot wrappers
- kinematics backends
- collision utilities
- samplers
- URDF and mesh assets
- cached point-cloud assets

Main files include:

- `robofin/robot_constants_aubo.py`
- `robofin/robots_aubo.py`
- `robofin/kinematics/numba_aubo.py`
- `robofin/kinematics/torch_aubo.py`
- `robofin/collision_aubo.py`
- `robofin/samplers_aubo.py`
- `robofin/urdf/aubo_i3H/`
- `robofin/cache/point_cloud/aubo_i3H/`

## Installation

From source:

```bash
pip install -e .
```

Package metadata:

- Name: `robofin`
- Version: `0.0.4.4`
- Python: `>=3.7`

## Development Notes

- This repository currently ignores `tests/` and `tools/` in `.gitignore`.
- If files under those paths were previously force-added, they must be removed from Git tracking with `git rm --cached` before they stop appearing in remote commits.

## License

MIT. See `LICENSE`.
