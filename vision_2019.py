import math
import cv2
from networktables import NetworkTables
from threading import Thread
from time import sleep

IS_NEW_CAMERA = True  # True for Axis M1045, False for Axis M1011

if IS_NEW_CAMERA:
    from grip_2019_newcamera import GripPipeline
else:
    from grip_2019_oldcamera import GripPipeline

vision_thread = None

FRAME_FETCH_INTERVAL = 1  # how long to wait between frames, in milliseconds

# Set up image stream
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
IM_WIDTH, IM_HEIGHT = 640, 480
if not IS_NEW_CAMERA:
    IM_WIDTH, IM_HEIGHT = 320, 240
COMPRESSION = 50  # from 0 to 100

NetworkTables.initialize(server='roborio-4373-frc.local')
sd = NetworkTables.getTable('SmartDashboard')
cap = cv2.VideoCapture(f'http://axis-camera.local/mjpg/video.mjpg?resolution={IM_WIDTH}x{IM_HEIGHT}'
                       + f'&compression={COMPRESSION}')

# constants in inches
VISION_TARGET_WIDTH = 2  # TODO: is this more accurate as 2.25?
INTER_VISION_TARGET_DIST = 8
FOCAL_LENGTH = 460.70837688446045 if IS_NEW_CAMERA else 382.8186340332031  # precomputed

# constants in pixels
Y_CROP_START = 0
Y_CROP_END = IM_HEIGHT
# X_CROP_START = 50
# X_CROP_END = 270
X_CROP_START = round(IM_WIDTH / 6) if IS_NEW_CAMERA else 0
X_CROP_END = round(IM_WIDTH * 5 / 6) if IS_NEW_CAMERA else IM_WIDTH

X_CENTER = (X_CROP_END - X_CROP_START) / 2
# FOV is roughly 52° with 2/3 frame crop on new camera; about 47 on old
DEGREES_PER_PIXEL = (52 if IS_NEW_CAMERA else 47) / (X_CROP_END - X_CROP_START)

X = 0
Y = 1

pipeline = GripPipeline()


def extra_processing(contours):
    """
    Performs extra calculations on the contours from the GRIP pipeline.

    :param contours: convex hulls from the GRIP pipeline.
    """
    print(f'{len(contours)} contours')
    if len(contours) > 3:
        sorted_contours = sorted(contours, key=lambda l: l[0][0][X])
        contour_pairs = []

        # collect all of the contours into their pairs
        for i in range(1, len(sorted_contours)):
            if is_correct_contour_pair(sorted_contours[i], sorted_contours[i - 1]):
                contour_pairs.append([sorted_contours[i], sorted_contours[i - 1]])

        # determine the centermost contour pair
        closest_mdpt = 0
        center_pair = []
        for pair in contour_pairs:
            pair_mdpt = find_contour_pair_midpoint_x(pair[0], pair[1])
            if math.fabs(pair_mdpt - X_CENTER) < math.fabs(closest_mdpt - X_CENTER):
                closest_mdpt = pair_mdpt
                center_pair = pair

        # publish midpoint
        if closest_mdpt != 0:
            publish_contour_distance(center_pair)
            sd.putString('vision_error', 'none')
        else:
            sd.putString('vision_error', 'no_valid_found')
        return

    elif len(contours) == 3:
        sorted_contours = sorted(contours, key=lambda l: l[0][0][X])
        for i in range(1, len(sorted_contours)):
            if is_correct_contour_pair(sorted_contours[i], sorted_contours[i - 1]):
                publish_contour_distance([sorted_contours[i], sorted_contours[i - 1]])
                sd.putString('vision_error', 'none')
                return
        sd.putString('vision_error', 'no_valid_found')

    elif len(contours) == 2:
        if is_correct_contour_pair(contours[0], contours[1]):
            publish_contour_distance(contours)
            sd.putString('vision_error', 'none')
            return
        sd.putString('vision_error', 'no_valid_found')

    else:
        print('too few')
        sd.putString('vision_error', 'too_few')
        return


def is_correct_contour_pair(contour1, contour2):
    """
    Returns whether a pair of contours is a correct vision target.
    """
    if contour1[0][0][X] < contour2[0][0][X]:
        leftmost_contour = contour1
        rightmost_contour = contour2
    else:
        leftmost_contour = contour2
        rightmost_contour = contour1
    return is_left_contour(leftmost_contour) and not is_left_contour(rightmost_contour)


def is_left_contour(contour):
    """
    Returns whether the detected contour is the one that belongs on the left-hand side of a contour pair.
    """
    contour_pts = [cnt_pnt[0] for cnt_pnt in contour]
    leftmost_point = min(contour_pts, key=lambda e: e[X])
    rightmost_point = max(contour_pts, key=lambda e: e[X])
    return leftmost_point[Y] > rightmost_point[Y]


def find_innermost_contour_point(contour):
    """
    Finds the point on a contour that is closest to the center of the vision target pair.

    :param contour: the contour whose inner point to find.
    :return: the innermost point of that contour (top right for the right target, top left for the left).
    """
    contour_pts = [cnt_pnt[0] for cnt_pnt in contour]
    if is_left_contour(contour):
        return max(contour_pts, key=lambda e: e[X])
    else:
        return min(contour_pts, key=lambda e: e[X])


def find_contour_pair_midpoint_x(contour1, contour2):
    """
    Determines if a pair of contours is a correct vision target pair.

    :return: True if the contours form a correct pair, or False if they do not.
    """
    inner_pt1 = find_innermost_contour_point(contour1)
    inner_pt2 = find_innermost_contour_point(contour2)
    contour_mdpt = (inner_pt2[X] + inner_pt1[X]) / 2
    return contour_mdpt


def find_lateral_distance_to_contour_mdpt(contour1, contour2):
    """
    Finds the horizontal distance to the midpoint of the specified contours,
    using the known horizontal distance between targets as a basis. The passed contours must be a valid pair.

    :return: the numerical lateral distance, in inches (the units of the inter-target distance), to the midpoint.
    """
    inner1 = find_innermost_contour_point(contour1)[X]
    inner2 = find_innermost_contour_point(contour2)[X]
    inter_contour_dist_px = math.fabs(inner2 - inner1)
    if inter_contour_dist_px == 0:
        return None
    px_to_in_ratio = INTER_VISION_TARGET_DIST / inter_contour_dist_px
    contour_mdpt = (inner2 + inner1) / 2
    dist_center_to_mdpt = contour_mdpt - X_CENTER
    return dist_center_to_mdpt * px_to_in_ratio


def publish_contour_distance(contours):
    """
    Publishes the lateral and forward distances to the target, in inches, to the Smart Dashboard.
    Also publishes the angle, in degrees, to the target.

    :param contours: two contours comprising the target contour pair.
    """
    left_contour = contours[0] if is_left_contour(contours[0]) else contours[1]
    rect = cv2.minAreaRect(left_contour)
    forward_distance = (VISION_TARGET_WIDTH * FOCAL_LENGTH) / min(rect[1][0], rect[1][1])
    lateral_distance = find_lateral_distance_to_contour_mdpt(contours[0], contours[1])
    angle_offset = (find_contour_pair_midpoint_x(contours[0], contours[1]) - X_CENTER) * DEGREES_PER_PIXEL

    sd.putNumber('forward_distance_to_target', forward_distance)
    if lateral_distance is not None:
        sd.putNumber('lateral_distance_to_target', lateral_distance)
    sd.putNumber('angle_to_target', angle_offset)
    print(f'forward_distance: {forward_distance}\tlateral_distance: {lateral_distance}\tangle_offset: {angle_offset}')


def calibrate_focal_length(known_distance_to_target, contours):
    """
    Determines the focal length of the camera based on a known distance to the target and the width of the target
    (defined as a constant) and prints it out.

    :param known_distance_to_target: the distance, in inches, to the target being used to calibrate.
    :param contours: a contour pair whose left contour will be used for calibration.
    """
    left_contour_rect = cv2.minAreaRect(contours[0] if is_left_contour(contours[0]) else contours[1])
    perimeter_width = min(left_contour_rect[1][0], left_contour_rect[1][1])
    print(perimeter_width * known_distance_to_target / VISION_TARGET_WIDTH)


shared_frame = None


def do_background_vision_computation():
    """
    Maintains an infinitely iterating thread that does vision computations on the shared frame.
    """
    global shared_frame
    while True:
        if shared_frame is None:
            return
        vision_frame = shared_frame[Y_CROP_START:Y_CROP_END, X_CROP_START:X_CROP_END]
        pipeline.process(vision_frame)
        extra_processing(pipeline.convex_hulls_output)


def main():
    global vision_thread, shared_frame
    vision_thread = Thread(target=do_background_vision_computation, args=()).start()
    while cap.isOpened():
        success, shared_frame = cap.read()
        if not success:
            print('Connection failed; will retry in 1 second...')
            sleep(1)
            continue
        cv2.imshow('camera', shared_frame)
        cv2.waitKey(FRAME_FETCH_INTERVAL)


if __name__ == "__main__":
    main()
