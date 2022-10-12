import pybullet as p
import time
import pybullet_data
import numpy as np

from numpy.linalg import norm
from pybullet_utils import bullet_client


def initializeGUI():
    # pb_client = bullet_client.BulletClient(connection_mode=p.DIRECT)
    pb_client = bullet_client.BulletClient(connection_mode=p.GUI)

    pb_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    pb_client.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    pb_client.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    pb_client.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb_client.setGravity(0, 0, -10)
    pb_client.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=20, cameraPitch=-40,
                                         cameraTargetPosition=(0, 0, 1))
    return pb_client


class IterRegistry(type):

    def __len__(cls):
        return len(cls._registry)

    def __iter__(cls):
        return iter(cls._registry)

    def __reversed__(cls):
        return reversed(cls._registry)


class Cube(metaclass=IterRegistry):
    _registry = []

    def __init__(self, assemblePos, assembleOrn, physics_client):
        self._registry.append(self)
        self.assemblyPos = assemblePos
        self.assemblyOrn = p.getQuaternionFromEuler(assembleOrn)
        self.pb_client = physics_client
        self.startPos = None
        self.startOrn = None
        self.boxId = None

    def load(self, filename, startPos, startOrn):
        self.startPos = startPos
        self.startOrn = p.getQuaternionFromEuler(startOrn)
        self.boxId = p.loadURDF(filename, self.startPos, self.startOrn)

    def get_pose(self):
        return self.pb_client.getBasePositionAndOrientation(self.boxId)

    def set_assembly_pose(self):
        self.pb_client.resetBasePositionAndOrientation(
            self.boxId,
            self.assemblyPos,
            self.assemblyOrn)

    def set_target_pose(self, target_pos, target_orn):
        self.pb_client.resetBasePositionAndOrientation(
            self.boxId,
            target_pos,
            target_orn)

    def reset_start_pose(self):
        self.pb_client.resetBasePositionAndOrientation(
            self.boxId,
            self.startPos,
            self.startOrn)

    def move_cube(self):
        alpha = 4
        force = alpha*np.array([0, 0, 3])
        current_pos, _ = self.get_pose()
        self.pb_client.applyExternalForce(
            objectUniqueId=self.boxId, linkIndex=-1,
            forceObj=force, posObj=current_pos,
            flags=p.WORLD_FRAME)



def contact_detection(cube_class, max_distance):
    num_cubes = len(cube_class)
    distances = []
    liaison_graph = np.zeros((num_cubes, num_cubes))

    for cube in cube_class:
        time.sleep(1.)
        cube.set_assembly_pose()

    for _ in range(100):
        time.sleep(1./24.)
        p.stepSimulation()

    for cube1 in cube_class:
        for cube2 in cube_class:
            closest_points = p.getClosestPoints(
                cube1.boxId,
                cube2.boxId,
                distance=max_distance
            )
            if len(closest_points) > 0:
                liaison_graph[cube1.boxId-2, cube2.boxId-2] = 1
                liaison_graph[cube2.boxId-2, cube1.boxId-2] = 1

    for cube in reversed(cube_class):
        time.sleep(1.)
        cube.reset_start_pose()

    for _ in range(100):
        time.sleep(1./24.)
        p.stepSimulation()

    return liaison_graph


def collision_detection(cube_class):
    # TODO: move cubes in other directions
    """Detects pairwise collision during disassembly"""
    num_cubes = len(cube_class)
    collision_map = np.zeros((num_cubes, num_cubes))

    for cube1 in cube_class:
        for cube2 in cube_class:

            if cube1.boxId == cube2.boxId:
                continue
            cube1.set_assembly_pose()
            cube2.set_assembly_pose()
            for _ in range(100):
                cube1.set_assembly_pose()
                cube2.move_cube()
                time.sleep(1./50.)
                p.stepSimulation()

            finalPos, _ = cube2.get_pose()
            posDiff = np.array(finalPos) - np.array(cube2.assemblyPos)
            if (posDiff[0]**2 + posDiff[1]**2) > 0.1 or posDiff[2] < 1:
                collision_map[cube1.boxId - 2, cube2.boxId - 2] = 1
                print(f"cube[boxId:{cube2.boxId}] is blocked by",
                      f"cube[boxId:{cube1.boxId}]")

            cube2.reset_start_pose()
        cube1.reset_start_pose()

    return collision_map


def stability_analysis(cube_class):
    """Analyzes the self and pairwise stability of cubes"""
    num_cubes = len(cube_class)
    stability_map = np.zeros((num_cubes, num_cubes))

    for cube1 in cube_class:
        for cube2 in cube_class:

            if cube1.boxId == cube2.boxId:
                cube2.set_assembly_pose()
                for _ in range(200):
                    time.sleep(1./240.)
                    p.stepSimulation()
            else:
                cube1.set_assembly_pose()
                cube2.set_assembly_pose()
                for _ in range(200):
                    cube1.set_assembly_pose()
                    time.sleep(1./240.)
                    p.stepSimulation()

            finalPos, _ = cube2.get_pose()
            posDiff = np.array(finalPos) - np.array(cube2.assemblyPos)
            if not norm(posDiff) > 0.03:
                stability_map[cube2.boxId - 2, cube1.boxId - 2] = 1
                print(f"cube[boxId:{cube2.boxId}] is stable with",
                      f"cube[boxId:{cube1.boxId}]'s support")

            cube2.reset_start_pose()
        cube1.reset_start_pose()

    return stability_map


def main():
    pb_client = initializeGUI()

    cubeT = Cube([0.00, 0.00, 0.97], [0, 0, -np.pi/2], pb_client)
    cubeL = Cube([-0.28, 0.05, 0.97], [0, 0, np.pi/2], pb_client)
    cubeV = Cube([0.04, -0.11, 1.21], [0, np.pi/2, np.pi], pb_client)
    cubeZ = Cube([-0.14, -0.10, 1.15], [0, np.pi/2, 0], pb_client)
    cubeB = Cube([-0.29, -0.15, 1.25], [-np.pi/2, 0, -np.pi/2], pb_client)
    cubeA = Cube([0.00, 0.15, 1.24], [0, np.pi/2, -np.pi/2], pb_client)
    cubeP = Cube([-0.28, 0.15, 1.31], [np.pi/2, 0, np.pi/2], pb_client)

    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0, 0, -1], globalScaling=3)
    cubeT.load("cubes/cube1.urdf", [-1.6, 1.0, 0.97], [0, 0, 0])
    cubeL.load("cubes/cube2.urdf", [-0.9, 1.0, 0.97], [0, 0, 0])
    cubeV.load("cubes/cube3.urdf", [-0.4, 1.0, 0.97], [0, 0, 0])
    cubeZ.load("cubes/cube4.urdf", [0.2, 1.0, 0.97], [0, 0, 0])
    cubeB.load("cubes/cube6.urdf", [0.8, 1.0, 1.0], [0, 0, 0])
    cubeA.load("cubes/cube7.urdf", [1.4, 1.0, 1.0], [0, 0, 0])
    cubeP.load("cubes/cube5.urdf", [2.0, 1.0, 1.0], [0, 0, 0])

    # stabilize the cubes
    for _ in range(100):
        time.sleep(1./240.)
        p.stepSimulation()

    # liaison_graph = contact_detection(Cube, 0.01)
    collision_map = collision_detection(Cube)
    stability_map = stability_analysis(Cube)

    # for cube in Cube:
    #     cube.set_assembly_pose()

    # for _ in range(1000):
    #     time.sleep(1./240.)
        # p.stepSimulation()

    p.disconnect()


if __name__ == "__main__":
    main()