import pybullet as p
import time
import pybullet_data
import numpy as np

from numpy.linalg import norm
from pybullet_utils import bullet_client
from tqdm import tqdm
from itertools import combinations


def initializeGUI():
    # pb_client = bullet_client.BulletClient(connection_mode=p.DIRECT)
    pb_client = bullet_client.BulletClient(connection_mode=p.GUI)
    pb_client.setAdditionalSearchPath(pybullet_data.getDataPath())

    pb_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    pb_client.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    pb_client.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    pb_client.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    pb_client.resetDebugVisualizerCamera(
        cameraDistance=1.8,
        cameraYaw=0, cameraPitch=-89,
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
        if self.boxId == None:
            print("cube not loaded ...")
            return
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


def initialCube(direction, pb_client):
    Cube._registry = []

    if direction == "+z":
        cubeT = Cube([0.00, 0.00, 0.97], [0, 0, -np.pi/2], pb_client)
        cubeL = Cube([-.28, 0.05, 0.97], [0, 0, np.pi/2], pb_client)
        cubeZ = Cube([-.14, -.10, 1.15], [0, np.pi/2, 0], pb_client)
        cubeV = Cube([0.04, -.11, 1.21], [0, np.pi/2, np.pi], pb_client)
        cubeB = Cube([-.29, -.15, 1.25], [-np.pi/2, 0, -np.pi/2], pb_client)
        cubeA = Cube([0.00, 0.15, 1.24], [0, np.pi/2, -np.pi/2], pb_client)
        cubeP = Cube([-.28, 0.15, 1.31], [np.pi/2, 0, np.pi/2], pb_client)
    elif direction =="-z":
        cubeT = Cube([0.03, -.00, 1.35], [0, 0, -np.pi/2], pb_client)
        cubeL = Cube([-.28, -.06, 1.35], [0, np.pi, np.pi/2], pb_client)
        cubeZ = Cube([-.13, 0.11, 1.17], [0, np.pi/2, 0], pb_client)
        cubeV = Cube([0.07, 0.11, 1.10], [0, -np.pi/2, 0], pb_client)
        cubeB = Cube([-.29, 0.15, 1.07], [np.pi/2, 0, np.pi/2], pb_client)
        cubeA = Cube([0.01, -.16, 1.07], [-np.pi/2, -np.pi/2, 0], pb_client)
        cubeP = Cube([-.28, -.15, 1.01], [0, 0, np.pi], pb_client)
    elif direction =="+y":
        cubeT = Cube([0.010, 0.148, 1.156], [0, -np.pi/2, -np.pi/2], pb_client)
        cubeL = Cube([-.267, 0.148, 1.210], [0, -np.pi/2, np.pi/2], pb_client)
        cubeZ = Cube([-.142, -.048, 1.070], [np.pi/2, 0, np.pi/2], pb_client)
        cubeV = Cube([0.043, -.110, 1.030], [0, np.pi/2, 0], pb_client)
        cubeB = Cube([-.290, -.150, 1.010], [0, 0, np.pi], pb_client)
        cubeA = Cube([-.002, -.148, 1.300], [0, np.pi, -np.pi/2], pb_client)
        cubeP = Cube([-.294, -.203, 1.300], [np.pi/2, -np.pi/2, np.pi/2], pb_client)
    elif direction == "-y":
        cubeT = Cube([-.005, -.190, 1.153], [0, -np.pi/2, -np.pi/2], pb_client)
        cubeL = Cube([-.282, -.190, 1.100], [0, np.pi/2, np.pi/2], pb_client)
        cubeZ = Cube([-.141, -.001, 1.260], [np.pi/2, 0, np.pi/2], pb_client)
        cubeV = Cube([0.043, 0.050, 1.280], [np.pi/2, 0, -np.pi/2], pb_client)
        cubeB = Cube([-.290, 0.115, 1.300], [0, np.pi, 0], pb_client)
        cubeA = Cube([-.002, 0.090, 1.000], [0, 0, -np.pi/2], pb_client)
        cubeP = Cube([-.294, 0.160, 1.000], [0, 0, np.pi/2], pb_client)
    elif direction == "+x":
        cubeT = Cube([0.060, 0.030, 1.30], [np.pi/2, 0, np.pi/2], pb_client)
        cubeL = Cube([0.052, 0.125, 1.00], [-np.pi/2, 0, np.pi/2], pb_client)
        cubeZ = Cube([-.150, -.080, 1.16], [0, 0, 0], pb_client)
        cubeV = Cube([-.195, -.088, 1.34], [0, 0, -np.pi/2], pb_client)
        cubeB = Cube([-.243, -.130, 1.00], [0, 0, -np.pi/2], pb_client)
        cubeA = Cube([-.260, 0.190, 1.30], [0, np.pi, 0], pb_client)
        cubeP = Cube([-.305, 0.175, 1.00], [0, 0, np.pi/2], pb_client)
    elif direction == "-x":
        cubeT = Cube([-.20, 0.00, 1.01], [-np.pi/2, 0, np.pi/2], pb_client)
        cubeL = Cube([-.21, 0.06, 1.29], [np.pi/2, 0, np.pi/2], pb_client)
        cubeZ = Cube([0.01, -.12, 1.16], [0, np.pi, 0], pb_client)
        cubeV = Cube([0.06, -.16, 0.97], [0, 0, np.pi], pb_client)
        cubeB = Cube([0.11, -.19, 1.30], [0, -np.pi/2, -np.pi/2], pb_client)
        cubeA = Cube([0.13, 0.13, 1.01], [0, 0, 0], pb_client)
        cubeP = Cube([0.18, 0.11, 1.30], [np.pi/2, 0, 0], pb_client)

    cubeT.load("cubes/cube1.urdf", [-1.8, 0.9, .97], [0, 0, 0])
    cubeL.load("cubes/cube2.urdf", [-1.0, 0.9, .97], [0, 0, 0])
    cubeZ.load("cubes/cube4.urdf", [-0.4, 0.9, .97], [0, 0, 0])
    cubeV.load("cubes/cube3.urdf", [0.2, 0.9, .97], [0, 0, 0])
    cubeB.load("cubes/cube6.urdf", [0.8, 0.9, 1.0], [0, 0, 0])
    cubeA.load("cubes/cube7.urdf", [1.4, 0.9, 1.0], [0, 0, 0])
    cubeP.load("cubes/cube5.urdf", [2.0, 0.9, 1.0], [0, 0, 0])
    return Cube


def test_assembly(direction, pb_client):
    pb_client.setGravity(0, 0, -10)
    pb_client.loadURDF("plane.urdf")
    pb_client.loadURDF("table/table.urdf", [0,0,-1], globalScaling=3)
    cube_class = initialCube(direction, pb_client)

    # stabilize the cubes
    for _ in range(100):
        time.sleep(1./240.)
        pb_client.stepSimulation()

    for cube in cube_class:
        cube.set_assembly_pose()

    for _ in range(100):
        time.sleep(1./80.)
        pb_client.stepSimulation()

    pb_client.resetSimulation()


def collision_detection(direction, pb_client):
    """Detects pairwise collision during disassembly"""
    print(f"\n***** collison check in {direction} direction *****")
    pb_client.setGravity(0, 0, -10)
    pb_client.loadURDF("plane.urdf")
    pb_client.loadURDF("table/table.urdf", [0,0,-1], globalScaling=3)
    cube_class = initialCube(direction, pb_client)

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
                time.sleep(1./120.)
                pb_client.stepSimulation()

            finalPos, _ = cube2.get_pose()
            posDiff = np.array(finalPos) - np.array(cube2.assemblyPos)
            if (posDiff[0]**2 + posDiff[1]**2) > 0.1 or posDiff[2] < 1:
                collision_map[cube2.boxId - 2, cube1.boxId - 2] = 1
                print(f"cube[boxId:{cube2.boxId}] collides with",
                      f"cube[boxId:{cube1.boxId}]")

            cube2.reset_start_pose()
        cube1.reset_start_pose()
    pb_client.resetSimulation()
    return collision_map


def stability_analysis(pb_client):
    # TODO: generate node level stability matrix
    """Analyzes the self and pairwise stability of cubes"""
    pb_client.setGravity(0, 0, -10)
    pb_client.loadURDF("plane.urdf")
    pb_client.loadURDF("table/table.urdf", [0,0,-1], globalScaling=3)
    cube_class = initialCube("+z", pb_client)

    # stabilize the cubes
    for _ in range(100):
        pb_client.stepSimulation()
    stable_subset = []

    for L in range(len(cube_class) + 1):
        for subset in combinations(cube_class._registry, L):
            subsetId = []

            for cube in subset:
                cube.set_assembly_pose()
                subsetId.append(cube.boxId - 2)

            for _ in range(200):
                time.sleep(1./240.)
                pb_client.stepSimulation()

            is_stable = True
            for cube in subset:
                finalPos, _ = cube.get_pose()
                posDiff = np.array(finalPos) - np.array(cube.assemblyPos)
                if norm(posDiff) > 0.03:
                    is_stable = False

            if is_stable and subsetId != []:
                stable_subset.append(subsetId)

            for cube in subset:
                cube.reset_start_pose()

    return stable_subset


def main():
    pb_client = initializeGUI()

    # test_assembly("+z", pb_client)
    # test_assembly("-z", pb_client)
    # test_assembly("+y", pb_client)
    # test_assembly("-y", pb_client)
    # test_assembly("+x", pb_client)
    # test_assembly("-x", pb_client)

    # # CM_{ijk} = 1 if cube i collides with cube j in direction k
    # collision_matrix = np.zeros((7, 7, 6))
    # collision_matrix[:,:,0] = collision_detection("+z", pb_client)
    # collision_matrix[:,:,1] = collision_detection("-z", pb_client)
    # collision_matrix[:,:,2] = collision_detection("+y", pb_client)
    # collision_matrix[:,:,3] = collision_detection("-y", pb_client)
    # collision_matrix[:,:,4] = collision_detection("+x", pb_client)
    # collision_matrix[:,:,5] = collision_detection("-x", pb_client)

    # AM_{ik} = U_{j=1}^{n} I_{ijk}
    # disassembly_matrix = np.any(collision_matrix, axis=1)

    stable_subset = stability_analysis(pb_client)

    pb_client.disconnect()


if __name__ == "__main__":
    main()