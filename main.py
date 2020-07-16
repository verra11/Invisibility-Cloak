from imutils.video import VideoStream
import cv2
import imutils
import argparse
import time
import os
import numpy as np

path = str(os.getcwd())
print(path)

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default=path,
	help="Path where the output is to be saved.")
ap.add_argument("-f", "--fps", type=int, default=60,
	help="FPS of the output video.")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
	help="Video Codec format.")

args = vars(ap.parse_args())

vs = VideoStream().start()
print("[INFO] warming up camera...")
time.sleep(2.0)

prev = None
fourcc = cv2.VideoWriter_fourcc(*args["codec"])

writer = None

while True:

	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	orig = frame.copy()


	if prev is None:
		print("[INFO] Background captured.")
		prev = frame
		time.sleep(1.0)

	if writer is None:
		(h, w) = frame.shape[:2]
		writer = cv2.VideoWriter(args["output"], fourcc, args["fps"],
			(w*2, h), True)

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower = np.array([0, 50, 50], dtype="uint8")
	upper = np.array([20, 255, 255], dtype="uint8")

	mask1 = cv2.inRange(hsv, lower, upper)

	lower = np.array([170, 50, 50], dtype="uint8")
	upper = np.array([180, 255, 255], dtype="uint8")

	mask2 = cv2.inRange(hsv, lower, upper)

	mask = mask1 + mask2

	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=3)

	inv_mask = cv2.bitwise_not(mask, mask=None)
	frame = cv2.bitwise_and(frame, frame, mask=inv_mask)
	cloak = cv2.bitwise_or(frame, prev, mask=mask)
	output = cv2.bitwise_or(frame, cloak, mask=None)


	final = np.zeros((h, w*2, 3), dtype="uint8")
	final[0:h,0:w] = orig
	final[0:h,w:w*2] = output
	writer.write(final)

	cv2.imshow("Output", np.hstack([orig, output]))

	key = cv2.waitKey(1) & 0xFF

	if key==ord("q"):
		break;

vs.stop()
cv2.destroyAllWindows()
writer.release()