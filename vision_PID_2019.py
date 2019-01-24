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
        print("Too many contours—move closer")
        return
    elif len(contours) == 3:
        sorted_contours = sorted(contours, key=lambda l: l[0][0][X])
        print(sorted_contours)
        print("******")
        for i in range(1, len(sorted_contours)):
            if is_correct_contour_pair(sorted_contours[i], sorted_contours[i - 1]):
                publish_contour_midpoint(sorted_contours[i], sorted_contours[i - 1])
                return
            print("No valid contours found")
    elif len(contours) == 2:
        if is_correct_contour_pair(contours[0], contours[1]):
            publish_contour_midpoint(contours[0], contours[1])
    else:
        print("Too few contours found")
        return


def is_correct_contour_pair(contour1, contour2):
    low_pt1, high_pt1 = find_inner_contour_points(contour1)
    low_pt2, high_pt2 = find_inner_contour_points(contour2)
    ratio = math.fabs(high_pt2[X] - high_pt1[X]) / math.sqrt(pow(low_pt2[X] - high_pt2[X], 2) + pow(low_pt2[Y] - high_pt2[Y], 2))
    # ideal ratio is 8 / 5.5—being between two would yield 12/5.5
    return ratio < 2


def real_is_correct_contour_pair(contour1, contour2):
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


def find_inner_contour_points(contour):
    contour_pts = [cnt_pnt[0] for cnt_pnt in contour]
    lower_inner_point = min(contour_pts, key=lambda e: e[Y])
    # remove inner low point from array
    contour_pts.remove(lower_inner_point)
    # get point from array with lowest y-value (will be the other bottom point)
    other_low_point = min(contour_pts, key=lambda e: e[Y])
    # get the difference of x-vals of low points (is the other low point to the left or right?)
    low_pt_difference = lower_inner_point[X] - other_low_point[X]
    # if difference is positive, we want the rightmost x; if it is negative, we want the leftmost x
    if low_pt_difference > 0:
        higher_inner_pt = max(contour_pts, key=lambda e: e[X])
    else:
        higher_inner_pt = min(contour_pts, key=lambda e: e[X])
    return lower_inner_point, higher_inner_pt


def publish_contour_midpoint(contour1, contour2):
    _, high_pt1 = find_inner_contour_points(contour1)
    _, high_pt2 = find_inner_contour_points(contour2)
    contour_mdpt = (high_pt2[X] + high_pt1[X]) / 2
    offset = contour_mdpt * DEGREES_PER_PIXEL - X_CENTER * DEGREES_PER_PIXEL
    if math.fabs(offset) < 2.5: # roughly 5% of FOV
        sd.putString('vision_lateral_correction', 'none')
    elif offset < 0:
        sd.putString('vision_lateral_correction', 'left')
    else:
        sd.putString('vision_lateral_correction', 'right')

    sd.putNumber('vision_angle_offset', offset)
    avg_contour_area = (cv2.contourArea(contour1) + cv2.contourArea(contour2)) / 2
    # TODO: do computations on area


while cap.isOpened():
    have_frame, frame = cap.read()
    pipeline.process(frame)
    extra_processing(pipeline.filter_contours_output)
    is_left_contour(pipeline.convex_hulls_output[0])
    print(is_left_contour(pipeline.filter_contours_output[1]))
