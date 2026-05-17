"""
Runs quality labeling using a simple OpenCV UI.
"""

import cv2
from quality_estimation import labeling


SCREEN_W, SCREEN_H = 1920, 1080


if __name__ == "__main__":
    labeler = labeling.Labeler()

    window_name = "Frame"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        img = labeler.draw_image_ui(SCREEN_W, SCREEN_H)
        cv2.imshow(window_name, img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        elif key == ord("a"):
            labeler.prev()
        elif key == ord("d"):
            labeler.next()
        elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
            rating = int(chr(key))
            labeler.set_label(rating)
    labeler.close()
