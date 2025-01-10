import cv2 as cv
import numpy as np

from helpers import compute_iou
from imutils.object_detection import non_max_suppression

class Detector: # Detektor
    def __init__(self): # Hog-Deskriptor
        self.hog = cv.HOGDescriptor()
        self.hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())


    def detect(self, frame, fgmask):  # Detektiert Boxen und filtert die beste

        # Frame Down Samplen für schnellere Berechnung
        frame_down_sample = cv.resize(frame, ((frame.shape[1] // 40) * 10, (frame.shape[0] // 40) * 10))
        boxes, weights = self.hog.detectMultiScale(frame_down_sample, winStride=(4, 4), padding=(8, 8),
                                                   scale=1.1, useMeanshiftGrouping=True)  # Je geringer das Scaling, desto weniger Boxen aber schneller

        #rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        pick = non_max_suppression(boxes, probs=None, overlapThresh=0.3)
        boxes = np.divide(pick * 40, 10).astype(int)  # Up-sampling der Koordinaten
        return boxes

    def filter_boxes2(self, boxes, fgmask, min_area=1000):
        valid_boxes = [box for box in boxes if
                       np.count_nonzero(fgmask[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]) > min_area]

        filtered_boxes = []

        print("BOXEN", valid_boxes)

        while valid_boxes:
            # Berechne den Anteil der Fläche der Person in der Box
            valid_boxes.sort(key=lambda b: np.count_nonzero(fgmask[b[1]:b[1] + b[3], b[0]:b[0] + b[2]]) / (b[2] * b[3]),
                             reverse=True)
            best_box = valid_boxes.pop(0)  # Box mit dem größten Anteil der Person
            filtered_boxes.append(best_box)

            # Boxen die stark mit der besten Box überlappen raus
            valid_boxes = [box for box in valid_boxes if compute_iou(best_box, box) < 0.3]

        print("GEFILTERT", filtered_boxes)
        return filtered_boxes

