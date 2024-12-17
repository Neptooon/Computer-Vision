import cv2 as cv
import numpy as np

from src.cv_modules.helpers import compare_histograms


class Detector: # Detektor
    def __init__(self): # Hog-Deskriptor
        self.hog = cv.HOGDescriptor()
        self.hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
        self.ids = []

    def create_id(self, hist, features):
        for id in self.ids:
            if compare_histograms(hist, id["hist"]) < 0.5:
                return id["id"]
        self.ids.append({
            "id": len(self.ids) + 1,
            "hist": hist,
            "features": features
        })
        return len(self.ids)

    def get_histogram_by_id(self, search_id):
        for entry in self.ids:
            if entry["id"] == search_id:
                return entry["hist"]
        return None  # Falls die ID nicht gefunden wurde

    def get_features_by_id(self, search_id):
        for entry in self.ids:
            if entry["id"] == search_id:
                return entry["features"]
        return None  # Falls die ID nicht gefunden wurde

    def detect(self, frame, fgmask):  # Detektiert Boxen und filtert die beste

        # Frame Down Samplen für schnellere Berechnung
        frame_down_sample = cv.resize(frame, ((frame.shape[1] // 25) * 10, (frame.shape[0] // 25) * 10))
        boxes, weights = self.hog.detectMultiScale(frame_down_sample, winStride=(8, 8), padding=(8, 8),
                                                   scale=1.07)  # 1.06 bzw 1.07
        boxes = np.divide(boxes * 25, 10).astype(int)  # Up-sampling der Koordinaten
        return self.filter_boxes(boxes, fgmask)

    @staticmethod
    def _inside(r, q):  # Wirft Boxen raus die innerhalb von anderen Boxen sind
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q
        return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

    def filter_boxes(self, boxes, fgmask, min_area=1000):  # Filtert die Boxen

        filtered_boxes = []
        for ri, r in enumerate(boxes):
            for qi, q in enumerate(boxes):
                if ri != qi and self._inside(r, q):
                    break
            else:
                filtered_boxes.append(r)
                print("FILTER")
        return [box for box in filtered_boxes if
                np.count_nonzero(fgmask[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]) > min_area]