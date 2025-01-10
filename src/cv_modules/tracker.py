import cv2 as cv
import numpy as np

from src.cv_modules.helpers import calculate_color_histogram, draw_boxes, draw_features, compare_histograms,\
    compute_iou, hog_descriptor_similarity, calculate_hog_descriptor, calculate_movement_similarity
from scipy.optimize import linear_sum_assignment


class Tracker:
    def __init__(self):
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 15, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7)
        self.tracks = []  # Liste von Track-Objekten
        self.id_count = 1

    def start_new_track(self, detection, frame, vis, contours):
        (x, y, w, h) = detection
        roi = frame[y:y + h, x:x + w]
        features = cv.goodFeaturesToTrack(roi, **self.feature_params)
        features = [(px + x, py + y) for px, py in np.float32(features).reshape(-1, 2)]
        hist = calculate_color_histogram(vis, [x, y, w, h], contours)
        hog = calculate_hog_descriptor(frame, detection)

        new_track = Track(box=(x, y, w, h), features=features, hist=hist, track_id=self.id_count, hog_deskriptor=hog)

        self.id_count += 1
        return new_track

    def update2(self, detections, frame, vis, contours):

        if len(self.tracks) == 0 and len(detections) != 0:
            for box in detections:
                new_track = self.start_new_track(box, frame, vis, contours)
                self.tracks.append(new_track)

        cost_matrix = self.setup_matrix(detections, contours, vis)
        if len(cost_matrix) == 0 or len(cost_matrix[0]) == 0:
            return

        assignments = [-1 for _ in range(len(self.tracks))]
        unassigned = []

        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        print(f"Track Indizes: {track_indices} und Detektion Indizes: {detection_indices}")

        for idx in range(len(track_indices)):
            print(f"Track: {track_indices[idx]} zu detektion: {detection_indices[idx]}")
            assignments[track_indices[idx]] = detection_indices[idx]
            if self.tracks[track_indices[idx]].lost:

                self.tracks[track_indices[idx]].lost = False
                (x, y, w, h) = detections[detection_indices[idx]]
                roi = frame[y:y + h, x:x + w]
                features = cv.goodFeaturesToTrack(roi, **self.feature_params)
                features = [(px + x, py + y) for px, py in np.float32(features).reshape(-1, 2)]
                self.tracks[track_indices[idx]].features = features
                self.tracks[track_indices[idx]].box = (x, y, w, h)
                self.tracks[track_indices[idx]].hog_descriptor = calculate_hog_descriptor(frame, self.tracks[track_indices[idx]].box)

        print(f"Assignments: {assignments}")

        cost_threshold = np.percentile(np.array(cost_matrix).flatten(), 80) # TODO

        for idx in range(len(assignments)):
            if assignments[idx] != -1:
                if cost_matrix[idx][assignments[idx]] > cost_threshold:  # 50
                    assignments[idx] = -1
                    unassigned.append(idx)
            else:
                self.tracks[idx].skipped_frames += 1


        del_tracks = []
        for idx in range(len(self.tracks)):
            if self.tracks[idx].skipped_frames > 5:
                del_tracks.append(idx)


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
                new_track = self.start_new_track(detections[unassigned_detections[idx]], frame, vis, contours)
                self.tracks.append(new_track)

        for i in range(len(assignments)):
            if assignments[i] != -1:
                self.tracks[assignments[i]].skipped_frames = 0
                #self.tracks[i].lost = False


    def setup_matrix(self, detections, contours, vis):

        print(f"ANZ T: {self.tracks}")
        print(f"ANZ D: {detections}")
        cost_matrix = []
        for track in self.tracks:
            costs = []
            for detection in detections:
                hist_sim = compare_histograms(calculate_color_histogram(vis, detection, contours), track.hist)
                hog_sim = hog_descriptor_similarity(track.hog_descriptor, calculate_hog_descriptor(vis, detection))
                movement_sim = calculate_movement_similarity(track, detection)

                total_cost = (0.4 * hist_sim) + (0.5 * hog_sim) + (0.1 * movement_sim)

                costs.append(total_cost)
            cost_matrix.append(costs)

        if len(cost_matrix) == 0 or len(cost_matrix[0]) == 0:
            return []
        print(f"Kostenmatrix: {cost_matrix}")
        return cost_matrix

    def update_tracks(self, prev_gray, frame_gray, fgmask, contours, vis=None):

        for track in self.tracks:
            if not track.lost:
                track.update(prev_gray, frame_gray, fgmask, contours, self.lk_params, vis)

class Track:

    def __init__(self, box, features, hist, track_id, hog_deskriptor):
        self.box = box  # (x, y, w, h)
        self.features = features  # Liste der Feature-Punkte
        self.hist = hist  # Farb-Histogramm der Box
        self.id = track_id  # Eindeutige ID
        self.center = [(box[0] + box[2] // 2), (box[1] + box[3] // 2)]  # Mittelpunkt der Box
        self.mean_shift = [np.array([0, 0])]  # Bewegungsgeschichte (Verschiebung)
        self.trace = [[self.center]]  # Historie der Positionen
        self.skipped_frames = 0
        self.lost = False
        self.hog_descriptor = hog_deskriptor

    def update(self, prev_gray, frame_gray, fgmask, contours, lk_params, vis=None):
        buffer = 20
        alpha = 0.15
        min_height, min_width = 50, 50

        # Berechne optischen Fluss
        features = np.float32(self.features).reshape(-1, 1, 2)
        print("LÄNGE:", len(features))
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

            if contours and cv.contourArea(max(contours, key=cv.contourArea)) > 40000.0:
                self.hist = calculate_color_histogram(vis, new_box, contours)
                self.hog_descriptor = calculate_hog_descriptor(frame_gray, new_box)

            self.box = new_box
            self.features = valid_points.tolist()
            self.mean_shift.append(mean_shift)
            self.center = [(new_box[0] + new_box[2] // 2), (new_box[1] + new_box[3] // 2)]
            self.trace.append([self.center])
            #self.skipped_frames = 0

        else:
            self.lost = True
            #self.skipped_frames += 1

    @staticmethod
    def _filter_valid_points(p1, st, features, fgmask):  # Nur Punkte in Roi werden als valid eingestuft
        valid_points = p1[st == 1].reshape(-1, 2)
        previous_points = features[st == 1].reshape(-1, 2)
        mask_filter = [  # +25
            not np.all(fgmask[int(p[1]):int(p[1]) + 25, int(p[0]):int(p[0]) + 25] == 0) for p in valid_points
        ]
        return valid_points[mask_filter], previous_points[mask_filter]