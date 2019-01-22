import numpy as np
import cv2
from networktables import NetworkTables
NetworkTables.initialize(server='roborio-4373-frc.local')
sd = NetworkTables.getTable('SmartDashboard')
cam = cv2.VideoCapture("http://10.43.73.74/mjpg/video.mjpg")
X_CENTER = 320
DEGREES_PER_PIXEL = 47 / 640
while True:
    ret, im = cam.read()
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(im)
    for keypoint in keypoints:
        print("d= ",keypoint.pt[0] - X_CENTER," (px), theta= ", keypoint.pt[0]*DEGREES_PER_PIXEL - X_CENTER * DEGREES_PER_PIXEL, " (degs)")
        # networktables stuff
        sd.putNumber('Camera_Pixel_Offset', keypoint.pt[0] - X_CENTER)
        sd.putNumber('Camera_Angle_Offset', keypoint.pt[0]*DEGREES_PER_PIXEL - X_CENTER * DEGREES_PER_PIXEL)
#        sd.putNumber('Received_Gyro_Value', sd.getNumber('Gyro value'))
        sd.putNumber('PID Setpoint', sd.getNumber('Gyro value') + (keypoint.pt[0]*DEGREES_PER_PIXEL - X_CENTER*DEGREES_PER_PIXEL))
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    break
