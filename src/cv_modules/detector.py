import cv2 as cv
import numpy as np

from imutils.object_detection import non_max_suppression

class Detector: # Detektor
    def __init__(self): # Hog-Deskriptor
        self.hog = cv.HOGDescriptor()
        self.hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())


    def detect(self, frame, fgmask):  # Detektiert Boxen und filtert die beste

        # Frame Down Samplen für schnellere Berechnung
        frame_down_sample = cv.resize(frame, ((frame.shape[1] // 40) * 10, (frame.shape[0] // 40) * 10))
        boxes, weights = self.hog.detectMultiScale(frame_down_sample, winStride=(2, 2), padding=(4, 4),
                                                   scale=1.07, useMeanshiftGrouping=True)  # Je geringer das Scaling, desto weniger Boxen aber schneller

        #rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        pick = non_max_suppression(boxes, probs=None, overlapThresh=0.6) # 0.3 TODO Kann dazu führen das bei nur 1 person die box detektiert wird wenn nahe aneinander
        boxes = np.divide(boxes * 40, 10).astype(int)  # Up-sampling der Koordinaten
        return boxes


