import math
import numpy as np
import cv2
from networktables import NetworkTables
from grip_2019 import GripPipeline
from threading import Thread

root = lmain = vision_thread = None

# Set up image stream
window_width, window_height = 640, 480
im_width, im_height = 320, 240

try:
    from PIL import Image, ImageTk
    import tkinter as tk
    root = tk.Tk()
    root.wm_attributes("-topmost", 1)
    root.bind('<Escape>', lambda e: root.quit())

    lmain = tk.Label(root, width=window_width, height=window_height)
    lmain.pack()
except ImportError:
    print('Could not import PIL, so no image stream will be displayed')

NetworkTables.initialize(server='roborio-4373-frc.local')
sd = NetworkTables.getTable('SmartDashboard')
# cap = cv2.VideoCapture("http://axis-camera.local/mjpg/video.mjpg?resolution=320x240")
cap = cv2.VideoCapture("http://10.43.73.74/mjpg/video.mjpg?resolution=320x240")

VISION_TARGET_WIDTH = 2
FOCAL_LENGTH = 382.8186340332031  # precomputed - previously 432

Y_CROP_START = 0
Y_CROP_END = 240
# X_CROP_START = 50
# X_CROP_END = 270
X_CROP_START = 0
X_CROP_END = 320

X_CENTER = (X_CROP_END - X_CROP_START) / 2
DEGREES_PER_PIXEL = 47 / (X_CROP_END - X_CROP_START)

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
            publish_contour_midpoint(closest_mdpt)
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
                publish_contour_midpoint(mdpt)
                publish_contour_distance([sorted_contours[i], sorted_contours[i - 1]])
                sd.putString('vision_error', 'none')
                return
        sd.putString('vision_error', 'no_valid_found')

    elif len(contours) == 2:
        if is_correct_contour_pair(contours[0], contours[1]):
            mdpt = find_contour_pair_midpoint_x(contours[0], contours[1])
            publish_contour_midpoint(mdpt)
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


# publishes the distance to the target, in inches, to the Smart Dashboard
def publish_contour_distance(contours):
    left_contour = contours[0] if is_left_contour(contours[0]) else contours[1]
    rect = cv2.minAreaRect(left_contour)
    distance = (VISION_TARGET_WIDTH * FOCAL_LENGTH) / min(rect[1][0], rect[1][1])
    sd.putNumber('distance_to_target', distance)
    print(f'distance: {distance}')


def publish_contour_midpoint(contour_mdpt):
    offset = contour_mdpt * DEGREES_PER_PIXEL - X_CENTER * DEGREES_PER_PIXEL
    if math.fabs(offset) < 2.5:  # roughly 5% of FOV
        sd.putString('vision_lateral_correction', 'none')
        print('none', end=' ')
    elif offset < 0:
        sd.putString('vision_lateral_correction', 'left')
        print('left', end=' ')
    else:
        sd.putString('vision_lateral_correction', 'right')
        print('right', end=' ')

    print(offset)
    sd.putNumber('vision_angle_offset', offset)


# determines the focal length of tåhe camera with a known distance to the target
def calibrate_focal_length(known_distance_to_target, contours):
    left_contour_rect = cv2.minAreaRect(contours[0] if is_left_contour(contours[0]) else contours[1])
    perimeter_width = min(left_contour_rect[1][0], left_contour_rect[1][1])
    print(perimeter_width * known_distance_to_target / VISION_TARGET_WIDTH)


shared_frame = None
new_frame = True


# renders a frame and does processing—used on Windows for simultaneous streaming and processing
def show_frame():
    global shared_frame, new_frame
    _, shared_frame = cap.read()
    new_frame = True
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2image = cv2.resize(cv2image, (0, 0), fx=2, fy=2)  # if we ever need to resize, this is how
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    # frame = frame[Y_CROP_START:Y_CROP_END, X_CROP_START:X_CROP_END]
    lmain.after(1, show_frame)


def do_background_vision_computation():
    global new_frame
    while True:
        if new_frame:
            pipeline.process(shared_frame)
            extra_processing(pipeline.convex_hulls_output)
            new_frame = False


if root is None:
    while cap.isOpened():
        _, frame = cap.read()
        frame = frame[Y_CROP_START:Y_CROP_END, X_CROP_START:X_CROP_END]
        pipeline.process(frame)
        extra_processing(pipeline.convex_hulls_output)
else:
    show_frame()
    vision_thread = Thread(target=do_background_vision_computation, args=()).start()
    root.mainloop()
