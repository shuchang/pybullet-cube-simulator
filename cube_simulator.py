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
    pb_client.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=20, cameraPitch=-30,
                                         cameraTargetPosition=(0, 0, 1))

    return pb_client


class IterRegistry(type):

    def __iter__(cls):
        return iter(cls._registry)


class Cube(metaclass=IterRegistry):
    _registry = []

    def __init__(self, assemblePos, assembleOrn, physics_client):
        self._registry.append(self)
        self.assemblePos = assemblePos
        self.assembleOrn = assembleOrn
        self.phy_client = physics_client
        self.boxId = None
        self.startPos = None
        self.startOrn = None

    def load(self, filename, startPos, startOrn):
        self.boxId = p.loadURDF(filename, startPos, startOrn)
        self.startPos = startPos
        self.startOrn = startOrn

    def get_pose(self):
        return self.phy_client.getBasePositionAndOrientation(self.boxId)

    def set_assembly_pose(self):
        if self.boxId != None:
            self.phy_client.resetBasePositionAndOrientation(
                self.boxId,
                self.assemblePos,
                self.assembleOrn)

    def set_target_pose(self, target_pos, target_orn):
        self.phy_client.resetBasePositionAndOrientation(
            self.boxId,
            target_pos,
            target_orn)

    def reset_start_pose(self):
        self.phy_client.resetBasePositionAndOrientation(
            self.boxId,
            self.startPos,
            self.startOrn)

    def move_cube(self, direction):
        assert direction in ["+x", "-x", "+y", "-y", "+z"]
        alpha = 4
        if direction == "+x":
            force = alpha*np.array([1, 0, 0])
        elif direction == "-x":
            force = alpha*np.array([-1, 0, 0])
        elif direction == "+y":
            force = alpha*np.array([0, 1, 0])
        elif direction == "-y":
            force = alpha*np.array([0, -1, 0])
        elif direction == "+z":
            force = alpha*np.array([0, 0, 3])

        current_pos = self.get_pose()[0]
        self.phy_client.applyExternalForce(objectUniqueId=self.boxId, linkIndex=-1,
                                           forceObj=force, posObj=current_pos,
                                           flags=p.WORLD_FRAME)



def contact_detection(cube_class, max_distance):
    num_cubes = len(cube_class._registry)
    distances = []
    liaison_graph = np.zeros((num_cubes, num_cubes))

    for cube in Cube:
        cube.set_assembly_pose()

    for _ in range(100):
        time.sleep(1./240.)
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

    for cube in Cube:
        cube.reset_start_pose()

    for _ in range(100):
        time.sleep(1./240.)
        p.stepSimulation()

    return liaison_graph


def collision_detection(cube_class):
    """Detects collision one by one during disassembly"""
    num_cubes = len(cube_class._registry)
    collision_map = np.zeros((num_cubes, num_cubes))

    for cube1 in cube_class:
        for cube2 in cube_class:
            if cube1.boxId == cube2.boxId:
                continue

            cube1.set_assembly_pose()
            cube2.set_assembly_pose()

            for _ in range(100):
                cube1.set_assembly_pose()
                cube2.move_cube(direction="+z")
                # time.sleep(1./120.)
                p.stepSimulation()

            finalPos, _ = cube2.get_pose()
            posDiff = np.array(finalPos) - np.array(cube2.assemblePos)
            if posDiff[0]**2+posDiff[1]**2 > 0.1 or posDiff[2] < 1:
                print("cube [boxId:{}] blocked by cube [boxId:{}]".
                format(cube2.boxId, cube1.boxId))
                collision_map[cube1.boxId - 2, cube2.boxId - 2] = 1

            cube2.reset_start_pose()
        cube1.reset_start_pose()

    return collision_map


def main():
    pb_client = initializeGUI()

    cubeT = Cube([0.00, 0.00, 0.95], p.getQuaternionFromEuler([0, 0, -np.pi/2]), pb_client)
    cubeL = Cube([-0.28, 0.05, 0.95], p.getQuaternionFromEuler([0, 0, np.pi/2]), pb_client)
    cubeV = Cube([0.04, -0.11, 1.20], p.getQuaternionFromEuler([0, np.pi/2, np.pi]), pb_client)
    cubeZ = Cube([-0.14, -0.10, 1.14], p.getQuaternionFromEuler([0, np.pi/2, 0]), pb_client)
    cubeP = Cube([-0.28, 0.14, 1.31], p.getQuaternionFromEuler([np.pi/2, 0, np.pi/2]), pb_client)
    cubeB = Cube([-0.28, -0.15, 1.25], p.getQuaternionFromEuler([-np.pi/2, 0, -np.pi/2]), pb_client)
    cubeA = Cube([0.00, 0.15, 1.24], p.getQuaternionFromEuler([0, np.pi/2, -np.pi/2]), pb_client)

    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0, 0, -1], globalScaling=3)
    cubeT.load("cubes/cube1.urdf", [-1.6, 1., 0.98], p.getQuaternionFromEuler([0, 0, 0]))
    cubeL.load("cubes/cube2.urdf", [-0.9, 1., 0.98], p.getQuaternionFromEuler([0, 0, 0]))
    cubeV.load("cubes/cube3.urdf", [-0.4, 1., 0.98], p.getQuaternionFromEuler([0, 0, 0]))
    cubeZ.load("cubes/cube4.urdf", [0.2, 1., 0.98], p.getQuaternionFromEuler([0, 0, 0]))
    cubeB.load("cubes/cube6.urdf", [0.8, 1., 1.], p.getQuaternionFromEuler([0, 0, 0]))
    cubeA.load("cubes/cube7.urdf", [1.4, 1., 1.], p.getQuaternionFromEuler([0, 0, 0]))
    cubeP.load("cubes/cube5.urdf", [2.0, 1., 1.], p.getQuaternionFromEuler([0, 0, 0]))

    # stabilize the cubes
    for _ in range(100):
        time.sleep(1./240.)
        p.stepSimulation()

    collision_map = collision_detection(Cube)
    liaison_graph = contact_detection(Cube, 0.01)

    p.disconnect()


if __name__ == "__main__":
    main()