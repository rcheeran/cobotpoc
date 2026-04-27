from contextlib import contextmanager
from pymycobot.mycobot280 import MyCobot280
import time

import cv2
import numpy as np

@contextmanager
def printVerbosely(codeBlockDescription):
    """Print to console before/after block, report outcome and duration"""

    # Print before message
    print(f"{codeBlockDescription}...")
    start = time.time()
    success = True

    try:
        yield
    except Exception:
        success = False
        raise

    # Print after message in green or red

    finally:
        elapsed = time.time() - start
        COLORS = {
            "G" : "\033[92m", # Green
            "R" : "\033[91m", # Red
            "W" : "\033[0m", # Default, usually white
        }
        if success:
            print(f"{COLORS['G']}✔ {codeBlockDescription} succeeded in {elapsed:.2f}s{COLORS['W']}")
        else:
            print(f"{COLORS['R']}✗ {codeBlockDescription} failed after {elapsed:.2f}s{COLORS['W']}")

# Vision

def calibrateColorRanges():
    """Open a live camera preview. Click any pixel to print its BGR and HSV values.
    Use the printed HSV values to tune the ranges in findCubeByColor(). Press 'q' to quit."""

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {CAMERA_INDEX}")

    windowName = "Color Calibration - click a cube, press q to quit"
    clickedHsv = [None]  # mutable container so the callback can write into it

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = param["frame"]
            hsv = param["hsv"]
            bgrPixel = frame[y, x].tolist()
            hsvPixel = hsv[y, x].tolist()
            print(f"Pixel ({x}, {y})  BGR: {bgrPixel}  HSV: {hsvPixel}  "
                  f"  -> H={hsvPixel[0]}, S={hsvPixel[1]}, V={hsvPixel[2]}")

    cv2.namedWindow(windowName)
    sharedData = {"frame": None, "hsv": None}
    cv2.setMouseCallback(windowName, onMouse, sharedData)

    print("calibrateColorRanges: click on each cube to read its HSV values. Press 'q' to quit.")
    while True:
        success, frame = cap.read()
        if not success:
            print("Warning: failed to grab frame")
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sharedData["frame"] = frame
        sharedData["hsv"] = hsv

        cv2.imshow(windowName, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyWindow(windowName)

def findCubeByColor(color):
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {CAMERA_INDEX}")

    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError("Failed to grab frame from camera")

    # Convert to HSV color space (better for color detection)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    colorRanges = {
        "red": [
            # Measured H=178; wraps near 180, single range sufficient
            (np.array([168, 100, 100]), np.array([180, 255, 255])),
        ],
        "green": [
            # Measured H=80
            (np.array([70, 150, 40]), np.array([90, 255, 255])),
        ],
        "blue": [
            # Measured H=109
            (np.array([99, 150, 40]), np.array([119, 255, 255])),
        ],
        "yellow": [
            # Measured H=31
            (np.array([21, 80, 100]), np.array([41, 255, 255])),
        ],
    }

    selectedRanges = colorRanges.get(color.lower())
    if selectedRanges is None:
        raise ValueError(f"Unsupported cube color: {color}")

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lowerBound, upperBound in selectedRanges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lowerBound, upperBound))

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # No matching contour means no candidate object in the frame.
    if not contours:
        return None

    # Use the largest matching blob as the cube candidate.
    largest = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest)
    # m00 is contour area in moment form; zero would make centroid undefined.
    if moments["m00"] == 0:
        return None

    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])
    z = 0 # Not available from single camera, assume on table

    # Fit a rotated rectangle to estimate object size and tabletop yaw
    (_, _), (width, height), angleDeg = cv2.minAreaRect(largest)

    # TODO: In future, change gripper grasp size based on object size
    sizeInPx = int((width + height) / 2)

    # For cubes, orientation is periodic every 90 degrees
    rZ = int(angleDeg % 90.0)

    return x, y, z, rZ

# Helpers

def transformCoords(coords):
    """Transform coordinates from camera frame to robot base frame using homography and angle offset."""
    inputX, inputY, inputZ, inputRz = coords
    # Verify not None
    if None in (inputX, inputY, inputZ, inputRz):
        return None, None, None, None
    # POSITION: Compute homography (see calibrate.py)
    robotCorners = np.array(
        [
            [-50, 200],
            [200, 200],
            [200, -200],
            [-50, -200],
        ],
        dtype=np.float32,
    )
    cameraCorners = np.array(
        [
            [486, 73],
            [486, 307],
            [118, 316],
            [106, 84],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(cameraCorners, robotCorners)
    point = np.array([[[float(inputX), float(inputY)]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(point, homography)
    outputX, outputY = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
    # ANGLE: Just shift 45 degrees, measured empirically
    outputRz = -(inputRz - 45)
    # Return
    return outputX, outputY, inputZ, outputRz

def checkIfCoordsAreSafe(coords):
    """Verify coordinates are within safe reachable area: between 100 mm and 280 mm distance from base, and x >= -100 mm."""
    inputX, inputY, inputZ, inputRz = coords
    if inputX < -100:
        return False
    distance = (inputX ** 2 + inputY ** 2) ** 0.5
    if distance < 100 or distance > 280:
        return False
    return True

# Foundational robot movements, using simplified coords = [x, y, z rz]

def robotCompleteMove(thresholdSeconds=0.2):
    """Wait for move to complete: repeatedly check is_moving() until you get thresholdSeconds of not moving."""
    checkEvery = 0.01 # Seconds
    requiredCount = int(thresholdSeconds / checkEvery)
    count = 0
    while count < requiredCount:
        if ROBOT.is_moving() == 1:
            count = 0
        else:
            count += 1
        time.sleep(checkEvery)

def robotMoveToStraightPose(speed=60):
    """Straight pose: all joints at zero degrees."""
    ROBOT.send_angles([0, 0, 0, 0, 0, 0], speed)
    robotCompleteMove()

def robotMoveToYieldPose(speed=60):
    """Yield pose: bent back, over its own base, to maximize camera view."""
    ROBOT.send_angles([-30, 45, -45, 0, 30, -45], speed)
    robotCompleteMove()

def robotMoveToReadyPose(speed=60):
    """Ready pose: bent slighty forward, with gripper forward."""
    ROBOT.send_angles([0, 0, -45, 45, 0, -45], speed)
    robotCompleteMove()

def robotMoveToDropPose(speed=60):
    """Drop pose: bent forward, with gripper pointing downwards, to release objects."""
    ROBOT.send_angles([0, -20, -30, -40, 0, -45], speed)
    robotCompleteMove()

def robotOpenGripper(speed=100):
    """Open the gripper."""
    ROBOT.set_gripper_value(100, speed)
    robotCompleteMove()

def robotCloseGripper(speed=100):
    """Close the gripper about halfway to grasp typical object."""
    ROBOT.set_gripper_value(40, speed)
    robotCompleteMove()

def robotMoveSimple(coords, speed=60):
    """Move directly to coords without interpolation."""
    x, y, z, rz = coords
    ROBOT.send_coords([x, y, z, 180, 0, rz], speed)
    robotCompleteMove()

def robotMoveSmooth(coords0, coords1, speed=60):
    """Interpolate movement from coords0 to coords1 in incremental Cartesian steps, keeping orientation constant."""
    # Unpack
    x0, y0, z0, rz0 = coords0
    x1, y1, z1, rz1 = coords1
    # Compute number of steps based on distance and speed
    stepDelay = 0.05
    distanceInMilimeters = ((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) ** 0.5
    durationInSeconds = (distanceInMilimeters / speed) * 0.45 # Adjusted empirically to match observed robot speed
    numberOfSteps = max(1, int((durationInSeconds / stepDelay)))
    # Let settle then read current orientation
    time.sleep(stepDelay)
    current = ROBOT.get_coords()
    rx, ry, rz = current[3], current[4], current[5]
    # Interpolate
    for i in range(1, numberOfSteps + 1):
        t = i / numberOfSteps
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        z = z0 + (z1 - z0) * t
        ROBOT.send_coords([x, y, z, rx, ry, rz], speed)
        time.sleep(stepDelay)

# Higher level robot actions

def robotPickAndDrop(pickCoords, dropCoords, speed=60):
    # Input coordinates in x-y plane
    pickX, pickY, pickZ, pickRz = pickCoords
    dropX, dropY, dropZ, dropRz = dropCoords

    # Fixed heights for picking up, dropping off, and moving with the object
    pickH = 105
    dropH = 125
    moveH = 210

    # Open gripper
    ROBOT.set_gripper_value(100, speed)
    time.sleep(0.1)
    # Move to over pickup location
    ROBOT.send_coords([pickX, pickY, moveH, 180, 0, pickRz], speed)
    robotCompleteMove()
    # Descend to pickup height
    robotMoveSmooth([pickX, pickY, moveH, pickRz], [pickX, pickY, pickH, pickRz], speed)
    time.sleep(0.1)
    # Close gripper about halfway to grasp object
    ROBOT.set_gripper_value(40, speed)
    time.sleep(0.5)
    # Raise to move height
    robotMoveSmooth([pickX, pickY, pickH, pickRz], [pickX, pickY, moveH, pickRz], speed)
    time.sleep(0.1)
    # Move to over dropoff location
    ROBOT.send_coords([dropX, dropY, moveH, 180, 0, dropRz], speed)
    robotCompleteMove()
    # Descend to drop height
    robotMoveSmooth([dropX, dropY, moveH, dropRz], [dropX, dropY, dropH, dropRz], speed)
    time.sleep(0.1)
    # Drop object by opening gripper
    ROBOT.set_gripper_value(100, speed)
    time.sleep(0.5)

def robotPickUp(pickCoords, speed=60):
    # Input coordinates in x-y plane
    pickX, pickY, pickZ, pickRz = pickCoords

    # Fixed heights for positioning over and picking up object
    overH = 210
    pickH = 105

    # Open gripper
    ROBOT.set_gripper_value(100, speed)
    time.sleep(0.1)
    # Move to over pickup location
    ROBOT.send_coords([pickX, pickY, overH, 180, 0, pickRz], speed)
    robotCompleteMove()
    # Descend to pickup height
    robotMoveSmooth([pickX, pickY, overH, pickRz], [pickX, pickY, pickH, pickRz], speed)
    time.sleep(0.1)
    # Close gripper about halfway to grasp object
    ROBOT.set_gripper_value(40, speed)
    time.sleep(0.5)
    # Raise
    robotMoveSmooth([pickX, pickY, pickH, pickRz], [pickX, pickY, overH, pickRz], speed)
    time.sleep(0.1)
    # Go to ready pose
    robotMoveToDropPose(speed)

# Example usage

CAMERA_INDEX = 0

ROBOT = None
ROBOT_HAS_OBJECT = None
ROBOT_LAST_PICKUP_LOCATION = None

# INITIALIZE

with printVerbosely("Prepare robot arm"):
    robotPort = "/dev/ttyACM0"
    robotBaud = 115200
    ROBOT = MyCobot280(robotPort, robotBaud)
    # Wait briefly to ensure connection is established
    time.sleep(1.5)

# EXAMPLE: PICK UP, WAIT, DROP IT

robotMoveToYieldPose(30)

rawCoords = findCubeByColor("red")
if rawCoords:
    cubeCoords = transformCoords(rawCoords)
    if checkIfCoordsAreSafe(cubeCoords):
        robotPickUp(cubeCoords, 60)
        time.sleep(2)
        robotOpenGripper()

time.sleep(2)

robotMoveToStraightPose()