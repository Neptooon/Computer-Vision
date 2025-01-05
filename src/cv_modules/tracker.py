import cv2 as cv
import numpy as np

from src.cv_modules.helpers import calculate_color_histogram, draw_boxes, draw_features, compare_histograms, compute_iou
from scipy.optimize import linear_sum_assignment


class Tracker:
    def __init__(self):
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 15, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7)
        self.tracks = []  # Liste von Track-Objekten
        self.id_count = 1

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


    def compute_track_features(self, detection, frame, vis, contours):
        (x, y, w, h) = detection
        roi = frame[y:y + h, x:x + w]
        features = cv.goodFeaturesToTrack(roi, **self.feature_params)
        features = [(px + x, py + y) for px, py in np.float32(features).reshape(-1, 2)]
        hist = calculate_color_histogram(vis, [x, y, w, h], contours)

        new_track = Track(box=(x, y, w, h), hist=hist, features=features, track_id=self.id_count)
        self.id_count += 1
        return new_track


    def update(self, detections, frame, vis, contours):

        # Tracks anlegen
        if len(self.tracks) == 0 and len(detections) != 0:
            for box in detections:
                new_track = self.compute_track_features(box, frame, vis, contours)
                self.tracks.append(new_track)

        cost_matrix = self.setup_matrix(detections, contours, vis)
        if len(cost_matrix) == 0 or len(cost_matrix[0]) == 0:
            return

        assignments = []
        unassigned = []

        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        # Zuordnung der Detektionen zu den Tracks
        for t_index, d_index in zip(track_indices, detection_indices):
            if cost_matrix[t_index][d_index] > 1.0:  # TODO Schwellwert
                #self.tracks[t_index].update()  # TODO HIER MUSS DER TRACK AKTUALISIERT WERDEN
                assignments.append(d_index)  # TODO: Weil hier wurde der detektion "d_index" der track self.tracks[t_index] zugeordnet
                self.tracks[t_index].lost = False
            else:
                self.tracks[t_index].lost = True
                self.tracks[t_index].skipped_frames += 1
                unassigned.append(d_index)

        # Welche Tracks können gelöscht werden?
        del_tracks = []
        for i, track in enumerate(self.tracks):
            if track.skipped_frames > 5:  # TODO bestimmte Frame Anzahl festlegen
                del_tracks.append(i)
        if len(del_tracks) > 0:
            for index in del_tracks:
                if index < len(self.tracks):
                    del self.tracks[index]
                    del assignments[index]

        # Welche Detektionen haben keine Zuordnung zu den Tracks bekommen
        if len(unassigned) != 0:
            for d_idx in unassigned:
                new_track = self.compute_track_features(detections[d_idx], frame, vis, contours)
                self.tracks.append(new_track)

        for track in self.tracks:
            if not track.updated:
                track.lost = True


    def update2(self, detections, frame, vis, contours):

        if len(self.tracks) == 0 and len(detections) != 0:
            for box in detections:
                new_track = self.compute_track_features(box, frame, vis, contours)
                self.tracks.append(new_track)

        cost_matrix = self.setup_matrix(detections, contours, vis)
        if len(cost_matrix) == 0 or len(cost_matrix[0]) == 0:
            return

        assignments = [-1 for _ in range(len(self.tracks))]
        unassigned = []

        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        for idx in range(len(track_indices)):
            assignments[track_indices[idx]] = detection_indices[idx]
            if self.tracks[track_indices[idx]].lost:

                self.tracks[track_indices[idx]].lost = False
                (x, y, w, h) = detections[detection_indices[idx]]
                roi = frame[y:y + h, x:x + w]
                features = cv.goodFeaturesToTrack(roi, **self.feature_params)
                features = [(px + x, py + y) for px, py in np.float32(features).reshape(-1, 2)]
                self.tracks[track_indices[idx]].features = features
                self.tracks[track_indices[idx]].box = (x, y, w, h)

        for idx in range(len(assignments)):
            if assignments[idx] != -1:
                if cost_matrix[idx][assignments[idx]] > 0.5:  # 50
                    assignments[idx] = -1
                    unassigned.append(idx)
            else:
                self.tracks[idx].skipped_frames += 1

        del_tracks = []
        for idx in range(len(self.tracks)):
            if self.tracks[idx].skipped_frames > 5:
                del_tracks.append(idx)
                self.tracks[idx].lost = True
            else:
                self.tracks[idx].lost = False
        if len(del_tracks) > 0:
            for idx in del_tracks:
                if idx < len(self.tracks):
                    del self.tracks[idx]
                    del assignments[idx]
                else:
                    pass

        unassigned_detections = []
        for idx in range(len(detections)):
            if idx not in assignments:
                unassigned_detections.append(idx)

        if len(unassigned_detections) != 0:
            for idx in range(len(unassigned_detections)):
                new_track = self.compute_track_features(detections[unassigned_detections[idx]], frame, vis, contours)
                self.tracks.append(new_track)
        print(self.tracks)
    def setup_matrix(self, detections, contours, vis):

        cost_matrix = []
        for track in self.tracks:
            costs = []
            for detection in detections:
                detection_center = [detection[0] + detection[2] // 2, detection[1] + detection[3] // 2]
                #iou = compute_iou(track.box, detection)
                hist_sim = compare_histograms(calculate_color_histogram(vis, detection, contours), track.hist)
                #print(np.array(track.trace[-1:]), "------------", np.array([detection_center]))
                center_dist = np.linalg.norm(np.array(track.trace[-1:]) - np.array([detection_center]))
                #print("TRACKER CENTER", track.trace[-1:], "-----------", "Detection Center", detection_center)
                #print("Distanz", center_dist)
                #flow = np.linalg.norm(track.mean_shift[-1:] - (np.array(track.trace[-1:]) - np.array([detection_center])))
                #print("FLOW", flow)
                cost = hist_sim

                costs.append(cost)
            cost_matrix.append(costs)

        #cost_matrix = (0.5) * np.array(cost_matrix)
        #print("Kostenmatrix", cost_matrix)
        if len(cost_matrix) == 0 or len(cost_matrix[0]) == 0:
            return []
        return cost_matrix

    def update_tracks(self, prev_gray, frame_gray, fgmask, contours, vis=None):

        for track in self.tracks:
            if not track.lost:
                track.update(prev_gray, frame_gray, fgmask, contours, self.lk_params, vis)

class Track:

    def __init__(self, box, features, hist, track_id):
        self.box = box  # (x, y, w, h)
        self.features = features  # Liste der Feature-Punkte
        self.hist = hist  # Farb-Histogramm der Box
        self.id = track_id  # Eindeutige ID
        self.center = [(box[0] + box[2] // 2), (box[1] + box[3] // 2)]  # Mittelpunkt der Box
        self.mean_shift = [[0, 0]]  # Bewegungsgeschichte (Verschiebung)
        self.trace = [[self.center]]  # Historie der Positionen
        self.skipped_frames = 0
        self.lost = False
        self.updated = True

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


            if contours and cv.contourArea(max(contours, key=cv.contourArea)) > 30000.0:
                self.hist = calculate_color_histogram(vis, new_box, contours)
            self.box = new_box
            self.features = valid_points.tolist()
            self.mean_shift.append(mean_shift)
            self.center = [(new_box[0] + new_box[2] // 2), (new_box[1] + new_box[3] // 2)]
            self.trace.append([self.center])
            self.skipped_frames = 0
            #self.lost = False
            #self.updated = True
        else:
            self.lost = True
            self.skipped_frames += 1

    @staticmethod
    def _filter_valid_points(p1, st, features, fgmask):  # Nur Punkte in Roi werden als valid eingestuft
        valid_points = p1[st == 1].reshape(-1, 2)
        previous_points = features[st == 1].reshape(-1, 2)
        mask_filter = [  # +25
            not np.all(fgmask[int(p[1]):int(p[1]) + 25, int(p[0]):int(p[0]) + 25] == 0) for p in valid_points
        ]
        return valid_points[mask_filter], previous_points[mask_filter]