import tkinter as tk
import cv2
from PIL import Image, ImageTk

window_width, window_height = 640, 480
im_width, img_height = 320, 240
cap = cv2.VideoCapture('http://10.43.73.74/mjpg/video.mjpg?resolution=320x240')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, im_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, im_height)

root = tk.Tk()
root.wm_attributes("-topmost", 1)
root.bind('<Escape>', lambda e: root.quit())

lmain = tk.Label(root, width=window_width, height=window_height)
lmain.pack()

def show_frame():
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2image = cv2.resize(cv2image, (0,0), fx=2, fy=2) # if we ever need to resize, this is how
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, show_frame)

show_frame()
root.mainloop()
