import cv2 as cv
import numpy as np

from src.cv_modules.helpers import calculate_color_histogram, draw_boxes, draw_features, compare_histograms, compute_iou
from src.cv_modules.id import ID

class Tracker:
    def __init__(self):
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 15, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7)
        self.tracks = []  # Liste von Track-Objekten
        self.ids = []
        self.last_tracks = []
        #self.last_partial_virt_tracks = []
        #self.lock = 1
        #self.virt_frame_counter = 0 in Methode checkVirt

    def reinitialize_features(self, frame_gray, box, contours=None):
        x, y, w, h = box
        buffer = 20
        dynamic_box = [
            max(0, x - buffer),
            max(0, y - buffer),
            min(frame_gray.shape[1], x + w + buffer),
            min(frame_gray.shape[0], y + h + buffer)
        ]
        roi = frame_gray[dynamic_box[1]:dynamic_box[3], dynamic_box[0]:dynamic_box[2]]
        features = cv.goodFeaturesToTrack(roi, **self.feature_params)
        if features is None:
            return []
        return [(px + dynamic_box[0], py + dynamic_box[1]) for px, py in np.float32(features).reshape(-1, 2)]

    def init_tracks(self, boxes, frame_gray, vis, detector, contours, fgmask):
        boxes = detector.filter_boxes(boxes, fgmask)
        for (x, y, w, h) in boxes:
            roi = frame_gray[y:y + h, x:x + w]
            features = cv.goodFeaturesToTrack(roi, **self.feature_params)
            features = [(px + x, py + y) for px, py in np.float32(features).reshape(-1, 2)]
            hist = calculate_color_histogram(vis, [x, y, w, h], contours)
            track_id, valid = self.create_id(hist, features, (x, y, w, h))
            print(track_id, valid)
            if features is not None and valid:
                self.tracks.append(Track((x, y, w, h), features, hist, track_id))
                self.ids.append(ID(track_id, hist, features, (x, y, w, h)))  # TODO Ersetzen
            elif not valid:
                p = self.get_track(track_id)
                if p.lost:  #  TODO Ersetzen
                    i = self.get_id(track_id)
                    p.lost = False
                    p.box = (x, y, w, h)
                    p.features = features
                    i.features = features


    def update_tracks(self, prev_gray, frame_gray, fgmask, contours, vis=None):
        for track in self.tracks:
            if not track.lost:
                track.update(prev_gray, frame_gray, fgmask, contours, self.lk_params, vis)

    def create_id(self, hist, features, box):  # id Klasse notwendig ? TODO Ersetzen
        for key, track in zip(self.ids, self.tracks):
            hist_similarity = compare_histograms(hist, key.hist)
            feature_similarity = np.mean([
                np.linalg.norm(np.array(f) - np.array(kf)) for f, kf in zip(features, key.features)
            ]) if features and key.features else float('inf')

            iou = compute_iou(box, track.box)
            print(hist_similarity, feature_similarity, iou)
            if hist_similarity <= 0.6 and feature_similarity >= 50 and iou >= 0.3:
                return key.id, False
        return len(self.ids) + 1, True

    def get_id(self, search_id): # TODO Ersetzen
        for entry in self.ids:
            if entry.id == search_id:
                return entry
        return None  # Falls die ID nicht gefunden wurde

    def get_track(self, track_id): # TODO Ersetzen
        for entry in self.tracks:
            if entry.id == track_id:
                return entry
        return None


class Track:

    def __init__(self, box, features, hist, track_id):
        self.box = box  # (x, y, w, h)
        self.features = features  # Liste der Feature-Punkte
        self.hist = hist  # Farb-Histogramm der Box
        self.id = track_id  # Eindeutige ID
        self.center = [(box[0] + box[2] // 2), (box[1] + box[3] // 2)]  # Mittelpunkt der Box
        self.mean_shift = [[0, 0]]  # Bewegungsgeschichte (Verschiebung)
        self.trace = []  # Historie der Positionen
        self.skipped_frames = 0
        self.lost = False

    def update(self, prev_gray, frame_gray, fgmask, contours, lk_params, vis=None):
        buffer = 20
        alpha = 0.15
        min_height, min_width = 50, 50

        # Berechne optischen Fluss
        features = np.float32(self.features).reshape(-1, 1, 2)
        p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, features, None, **lk_params)
        p0, st0, err0 = cv.calcOpticalFlowPyrLK(frame_gray, prev_gray, p1, None, **lk_params)

        # Filtere valide Punkte
        valid_points, previous_points = self._filter_valid_points(p1, st, p0, fgmask) # features

        if len(valid_points) > 10:
            self.lost = False
            movement = valid_points - previous_points
            mean_shift = np.mean(movement, axis=0)

            # Box-Parameter aktualisieren
            x, y, w, h = self.box
            dx, dy = mean_shift
            box_center = np.array([x + dx + w // 2, y + dy + h // 2])

            # Neue dynamische Box berechnen
            valid_y_coords = valid_points[:, 1]
            valid_x_coords = valid_points[:, 0]
            min_y = int(np.min(valid_y_coords)) - buffer
            max_y = int(np.max(valid_y_coords)) + buffer
            min_x = int(np.min(valid_x_coords)) - buffer
            max_x = int(np.max(valid_x_coords)) + buffer
            dynamic_height = max(max_y - min_y, min_height)
            dynamic_width = max(max_x - min_x, min_width)

            if contours:
                contour_centers = []
                for contour in contours:
                    M = cv.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        if min_x <= cx <= min_x + dynamic_width and min_y <= cy <= min_y + dynamic_height:
                            contour_centers.append((cx, cy))

                if contour_centers:
                    distances = [np.linalg.norm(np.array(center) - box_center) for center in contour_centers]
                    weights = 1 / (np.array(distances) + 1e-5)
                    weights /= np.sum(weights)
                    weighted_center = np.sum(np.array(contour_centers) * weights[:, None], axis=0)

                    even_center = (1 - alpha) * box_center + alpha * weighted_center
                    new_x = int(even_center[0] - dynamic_width // 2)
                    new_y = int(even_center[1] - dynamic_height // 2)

                    new_box = (new_x, new_y, dynamic_width, dynamic_height)
                else:
                    new_box = (x + dx, y + dy, w, h)
            else:
                new_box = (x + dx, y + dy, w, h)

            # Update der Track-Eigenschaften
            self.box = new_box
            self.features = valid_points.tolist()
            self.mean_shift.append(mean_shift.tolist())
            self.center = [(new_box[0] + new_box[2] // 2), (new_box[1] + new_box[3] // 2)]
            self.trace.append(self.center)
        else:
            self.skipped_frames += 1
            self.lost = True

    @staticmethod
    def _filter_valid_points(p1, st, features, fgmask):  # Nur Punkte in Roi werden als valid eingestuft
        valid_points = p1[st == 1].reshape(-1, 2)
        previous_points = features[st == 1].reshape(-1, 2)
        mask_filter = [  # +25
            not np.all(fgmask[int(p[1]):int(p[1]) + 25, int(p[0]):int(p[0]) + 25] == 0) for p in valid_points
        ]
        return valid_points[mask_filter], previous_points[mask_filter]