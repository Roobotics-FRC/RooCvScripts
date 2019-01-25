import math
import numpy as np
import cv2
from networktables import NetworkTables
from grip_2019 import GripPipeline
NetworkTables.initialize(server='roborio-4373-frc.local')
sd = NetworkTables.getTable('SmartDashboard')
cap = cv2.VideoCapture("http://10.43.73.74/mjpg/video.mjpg?resolution=320x240")
X_CENTER = 160
DEGREES_PER_PIXEL = 47 / 320

X = 0
Y = 1

pipeline = GripPipeline()


def extra_processing(contours):
    if len(contours) > 3:
        sd.putString('vision_error', 'too_many')
        return
    elif len(contours) == 3:
        sorted_contours = sorted(contours, key=lambda l: l[0][0][X])
        for i in range(1, len(sorted_contours)):
            if is_correct_contour_pair(sorted_contours[i], sorted_contours[i - 1]):
                publish_contour_midpoint(sorted_contours[i], sorted_contours[i - 1])
                sd.putString('vision_error', 'none')
                return
        print("No valid contours found")
    elif len(contours) == 2:
        if is_correct_contour_pair(contours[0], contours[1]):
            publish_contour_midpoint(contours[0], contours[1])
            sd.putString('vision_error', 'none')
            return
    else:
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


def publish_contour_midpoint(contour1, contour2):
    inner_pt1 = find_innermost_contour_point(contour1)
    inner_pt2 = find_innermost_contour_point(contour2)
    contour_mdpt = (inner_pt2[X] + inner_pt1[X]) / 2
    offset = contour_mdpt * DEGREES_PER_PIXEL - X_CENTER * DEGREES_PER_PIXEL
    if math.fabs(offset) < 2.5:  # roughly 5% of FOV
        sd.putString('vision_lateral_correction', 'none')
    elif offset < 0:
        sd.putString('vision_lateral_correction', 'left')
    else:
        sd.putString('vision_lateral_correction', 'right')

    sd.putNumber('vision_angle_offset', offset)
    sd.putNumber('distance_to_target', math.tan(offset) * (contour_mdpt - X_CENTER))


while cap.isOpened():
    have_frame, frame = cap.read()
    pipeline.process(frame)
    extra_processing(pipeline.convex_hulls_output)
