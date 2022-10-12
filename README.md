# PyBullet Simulator for Soma Cube Assembly
This is a simulation environment for Soma Cube Assembly based on PyBullet. Below is a right assembly sequence for Soma cubes.
![Cube Assembly](docs/imgs/cube_assembly.gif)

## URDF
Given STL files of blocks, URDF files are generated with [object2urdf](https://pypi.org/project/object2urdf/).


## PyBullet
In Pybullet, we implement pairwise collision detection and subassembly stability check. The resulting  feasibility matrices are further used to assist the learning of generalizable sequence planners.