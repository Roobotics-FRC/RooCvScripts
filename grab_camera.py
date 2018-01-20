from sys import argv

if len(argv) != 2:
  print("Usage: " + argv[0] + " <full address to http stream>")
  exit(1)


import cv2
#cam = cv2.VideoCapture("http://10.43.73.74/mjpg/video.mjpg")
cam = cv2.VideoCapture(argv[1])
while True:
    ret, im = cam.read()
    cv2.imshow("Camera Stream", im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
