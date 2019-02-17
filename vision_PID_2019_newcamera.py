import math
import cv2
import numpy
from networktables import NetworkTables
from grip_2019_newcamera import GripPipeline
from threading import Thread

root = lmain = vision_thread = None

FRAME_FETCH_INTERVAL = 1  # how long to wait between frames, in milliseconds

# Set up image stream
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
IM_WIDTH, IM_HEIGHT = 640, 480
COMPRESSION = 50  # from 0 to 100

try:
    from PIL import Image, ImageTk
    import tkinter as tk
    root = tk.Tk()
    root.wm_attributes("-topmost", 1)
    root.bind('<Escape>', lambda e: root.quit())

    lmain = tk.Label(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    lmain.pack()
except ImportError:
    print('Could not import PIL; OpenCV imshow debugging mode will be used instead')

NetworkTables.initialize(server='roborio-4373-frc.local')
sd = NetworkTables.getTable('SmartDashboard')
cap = cv2.VideoCapture(f'http://axis-camera.local/mjpg/video.mjpg?resolution={IM_WIDTH}x{IM_HEIGHT}'
                        + f'&compression={COMPRESSION}')

# constants in inches
VISION_TARGET_WIDTH = 2  # TODO: is this might accurate as 2.25?
INTER_VISION_TARGET_DIST = 8
FOCAL_LENGTH = 558.1091213226318  # precomputed

# constants in pixels
Y_CROP_START = 0
Y_CROP_END = IM_HEIGHT
# X_CROP_START = 50
# X_CROP_END = 270
X_CROP_START = round(IM_WIDTH / 6)
X_CROP_END = round(IM_WIDTH * 5 / 6)

X_CENTER = (X_CROP_END - X_CROP_START) / 2
DEGREES_PER_PIXEL = 52 / (X_CROP_END - X_CROP_START)  # FOV is roughly 52° with 2/3 frame crop

X = 0
Y = 1

pipeline = GripPipeline()


def extra_processing(contours):
    print(f'{len(contours)} contours')
    if len(contours) > 3:
        sorted_contours = sorted(contours, key = lambda l: l[0][0][X])
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
                mdpt = find_contour_pair_midpoint_x(sorted_contours[i], sorted_contours[i - 1])
                publish_contour_distance([sorted_contours[i], sorted_contours[i - 1]])
                sd.putString('vision_error', 'none')
                return
        sd.putString('vision_error', 'no_valid_found')

    elif len(contours) == 2:
        if is_correct_contour_pair(contours[0], contours[1]):
            mdpt = find_contour_pair_midpoint_x(contours[0], contours[1])
            publish_contour_distance(contours)
            sd.putString('vision_error', 'none')
            return
        sd.putString('vision_error', 'no_valid_found')

    else:
        print('too few')
        sd.putString('vision_error', 'too_few')
        return


# returns whether a pair of contours is a correct vision target
def is_correct_contour_pair(contour1, contour2):
    if contour1[0][0][X] < contour2[0][0][X]:
        leftmost_contour = contour1
        rightmost_contour = contour2
    else:
        leftmost_contour = contour2
        rightmost_contour = contour1
    return is_left_contour(leftmost_contour) and not is_left_contour(rightmost_contour)


# returns whether the detected contour is the one that belongs on the left-hand side of a contour pair
def is_left_contour(contour):
    contour_pts = [cnt_pnt[0] for cnt_pnt in contour]
    leftmost_point = min(contour_pts, key=lambda e: e[X])
    rightmost_point = max(contour_pts, key=lambda e: e[X])
    return leftmost_point[Y] > rightmost_point[Y]


def find_innermost_contour_point(contour):
    contour_pts = [cnt_pnt[0] for cnt_pnt in contour]
    if is_left_contour(contour):
        return max(contour_pts, key=lambda e: e[X])
    else:
        return min(contour_pts, key=lambda e: e[X])


def find_contour_pair_midpoint_x(contour1, contour2):
    inner_pt1 = find_innermost_contour_point(contour1)
    inner_pt2 = find_innermost_contour_point(contour2)
    contour_mdpt = (inner_pt2[X] + inner_pt1[X]) / 2
    return contour_mdpt


def find_lateral_distance_to_contour_mdpt(contour1, contour2):
    inner1 = find_innermost_contour_point(contour1)[X]
    inner2 = find_innermost_contour_point(contour2)[X]
    inter_contour_dist_px = math.fabs(inner2 - inner1)
    px_to_in_ratio = INTER_VISION_TARGET_DIST / inter_contour_dist_px
    contour_mdpt = (inner2 + inner1) / 2
    dist_center_to_mdpt =  contour_mdpt - (X_CROP_END - X_CROP_START) / 2  # (XCE-XCS)/2 is the x midpoint of the frame
    return dist_center_to_mdpt * px_to_in_ratio


# publishes the distance to the target, in inches, to the Smart Dashboard
def publish_contour_distance(contours):
    left_contour = contours[0] if is_left_contour(contours[0]) else contours[1]
    rect = cv2.minAreaRect(left_contour)
    forward_distance = (VISION_TARGET_WIDTH * FOCAL_LENGTH) / min(rect[1][0], rect[1][1])
    lateral_distance = find_lateral_distance_to_contour_mdpt(contours[0], contours[1])
    sd.putNumber('forward_distance_to_target', forward_distance)
    sd.putNumber('lateral_distance_to_target', lateral_distance)
    print(f'forward_distance: {forward_distance}\tlateral_distance: {lateral_distance}')


# determines the focal length of tåhe camera with a known distance to the target
def calibrate_focal_length(known_distance_to_target, contours):
    left_contour_rect = cv2.minAreaRect(contours[0] if is_left_contour(contours[0]) else contours[1])
    perimeter_width = min(left_contour_rect[1][0], left_contour_rect[1][1])
    print(perimeter_width * known_distance_to_target / VISION_TARGET_WIDTH)


shared_frame = None


# renders a frame and does processing—used on Windows for simultaneous streaming and processing
def show_frame():
    global shared_frame
    _, shared_frame = cap.read()
    cv2image = cv2.cvtColor(shared_frame, cv2.COLOR_BGR2RGBA)
    cv2image = cv2.resize(cv2image, (0, 0), fx=WINDOW_WIDTH / IM_WIDTH, fy=WINDOW_HEIGHT / IM_HEIGHT)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(FRAME_FETCH_INTERVAL, show_frame)


def do_background_vision_computation():
    while True:
        frame = shared_frame[Y_CROP_START:Y_CROP_END, X_CROP_START:X_CROP_END]
        pipeline.process(frame)
        extra_processing(pipeline.convex_hulls_output)


if root is None:
    while cap.isOpened():
        _, frame = cap.read()
        frame = frame[Y_CROP_START:Y_CROP_END, X_CROP_START:X_CROP_END]
        pipeline.process(frame)
        extra_processing(pipeline.convex_hulls_output)
        contour_frame = cv2.drawContours(frame, pipeline.convex_hulls_output, -1, (0, 0, 0), 1)
        cv2.imshow('contours', contour_frame)
        cv2.waitKey(FRAME_FETCH_INTERVAL)
else:
    show_frame()
    vision_thread = Thread(target=do_background_vision_computation, args=()).start()
    root.mainloop()
