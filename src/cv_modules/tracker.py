import cv2 as cv
import numpy as np

from src.cv_modules.helpers import calculate_color_histogram, draw_boxes, draw_features, compare_histograms, \
    compute_iou, calculate_hog_descriptor, calculate_movement_similarity, match_feature_cost, \
    merge_boxes, merge_contours
from scipy.optimize import linear_sum_assignment


class Tracker:
    def __init__(self):
        self.lk_params = dict(winSize=(31, 31), maxLevel=4,  # (21,21), 3
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20, 0.03))  # 15
        self.feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=7, blockSize=7)  # 10 minDist
        self.tracks = []  # Liste von Track-Objekten
        self.id_count = [-1] * 2

    def start_new_track(self, detection, frame, vis, contours, mask):
        (x, y, w, h) = detection
        x = int(max(0, min(x, frame.shape[1] - 1)))
        y = int(max(0, min(y, frame.shape[0] - 1)))
        w = int(min(w, frame.shape[1] - x))
        h = int(min(h, frame.shape[0] - y))
        roi = frame[y:y + h, x:x + w]  # TODO 1. Nur auf die Maske beziehen ?

        features = cv.goodFeaturesToTrack(roi, **self.feature_params)
        if features is not None:
            features = [(px + x, py + y) for px, py in np.float32(features).reshape(-1, 2)]
            contours = merge_contours(contours)
            p, hist = calculate_color_histogram(vis, detection, contours, mask)
            hog = calculate_hog_descriptor(frame, detection, mask)

            set_id = 0
            for i, id in enumerate(self.id_count):
                if id == -1:
                    self.id_count[i] = 1
                    set_id = i + 1
                    break


            new_track = Track(box=(x, y, w, h), features=features, hist=hist, track_id=set_id,
                              hog_deskriptor=hog)
            new_track.ref_box = (x, y, w, h)

            return new_track
        return None

    def update2(self, detections, frame, vis, contours, mask):

        if len(self.tracks) == 0:
            for box in detections:
                new_track = self.start_new_track(box, frame, vis, contours, mask)
                if new_track:
                    self.tracks.append(new_track)
            return

        cost_matrix = self.setup_matrix(detections, contours, vis, frame, mask)
        if not cost_matrix:
            return

        print(f"Kostenmatrix: {cost_matrix}")

        assignments = [-1] * len(self.tracks)
        unassigned = []

        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        print(f"Track Indizes: {track_indices} und Detektion Indizes: {detection_indices}")

        for idx in range(len(track_indices)):
            print(f"Track: {track_indices[idx]} zu detektion: {detection_indices[idx]}")
            assignments[track_indices[idx]] = detection_indices[idx]
            print(
                f"TRACK {self.tracks[track_indices[idx]].id} -> {detection_indices[idx]} DETEKTION {detections[detection_indices[idx]]}")


        cost_threshold = np.percentile(np.array(cost_matrix).flatten(), 80)

        for idx in range(len(assignments)):
            if assignments[idx] != -1:
                if cost_matrix[idx][assignments[idx]] > cost_threshold:  # 50
                    assignments[idx] = -1
                    unassigned.append(idx)
            else:
                self.tracks[idx].skipped_frames += 1

        del_tracks = []
        for idx in range(len(self.tracks)):
            if self.tracks[idx].skipped_frames > 150:
                del_tracks.append(idx)


        if len(del_tracks) > 0:
            for idx in del_tracks:
                if idx < len(self.tracks):
                    del self.tracks[idx]
                    del assignments[idx]
                    self.id_count[idx] = -1
                else:
                    pass

        unassigned_detections = []
        for idx in range(len(detections)):
            if idx not in assignments:
                unassigned_detections.append(idx)

        if len(unassigned_detections) != 0:
            for idx in range(len(unassigned_detections)):
                if not any([True for track in self.tracks if
                            compute_iou(track.box, detections[unassigned_detections[idx]]) > 0.0]) and self.id_count.count(-1) >= 1:
                    new_track = self.start_new_track(detections[unassigned_detections[idx]], frame, vis, contours, mask)
                    self.tracks.append(new_track)

        for idx in range(len(assignments)):
            if assignments[idx] != -1:
                (x, y, w, h) = detections[assignments[idx]]
                roi = frame[y:y + h, x:x + w]
                features = cv.goodFeaturesToTrack(roi, **self.feature_params)
                features = [(px + x, py + y) for px, py in np.float32(features).reshape(-1, 2)]
                self.tracks[idx].features = features
                self.tracks[idx].box = (x, y, w, h)
                self.tracks[idx].lost = False
                self.tracks[idx].skipped_frames = 0
                self.tracks[idx].non_detects = 0

                if not any([True for track in self.tracks if self.tracks[idx] != track and compute_iou(track.box, detections[assignments[idx]]) > 0.0]):
                    self.tracks[idx].hog_descriptor = calculate_hog_descriptor(frame, self.tracks[idx].box, mask)
                    p, new_hist = calculate_color_histogram(vis, self.tracks[idx].box, contours, mask)
                    if p >= 36000 or p<= 10000:
                        self.tracks[idx].hist = new_hist
            else:
                self.tracks[idx].non_detects += 1
                if self.tracks[idx].non_detects >= 15:
                    self.tracks[idx].lost = True

    def setup_matrix(self, detections, contours, vis, frame_gray, mask):

        print(f"ANZ Tracks für Matrix: {self.tracks}")
        print(f"ANZ Deteks für Matrix: {detections}")

        cost_matrix = []
        for track in self.tracks:
            costs = []
            for detection in detections:
                p, d_hist = calculate_color_histogram(vis, detection, contours, mask)
                hist_sim = compare_histograms(d_hist, track.hist)
                # Histogrammähnlichkeit: kleiner ist besser

                hog_sim = np.linalg.norm(calculate_hog_descriptor(frame_gray, detection, mask) - track.hog_descriptor)

                movement_cost = calculate_movement_similarity(track, detection)

                # Gewichte anpassen 65 25 10
                total_cost = (
                        0.65 * hist_sim +
                        0.25 * min(hog_sim / 100, 1) +
                        0.10 * movement_cost
                )

                costs.append(total_cost)
            cost_matrix.append(costs)

        if len(cost_matrix) == 0 or len(cost_matrix[0]) == 0:
            return []
        # print(f"Kostenmatrix: {cost_matrix}")
        return cost_matrix

    def handle_collision(self, collisions, contours, vis, frame_gray, mask):
        Track.count += 1
        merged_box = merge_boxes(collisions)

        x, y, w, h = merged_box
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        #cv.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 5)

        foreground_track = self.setup_collision_matrix(collisions, merged_box, contours, vis, frame_gray, mask)

        for track in collisions:
            if track.id == foreground_track.id:
                track.lost = False
            else:
                track.lost = True  # Geht nicht mehr in update sondern muss neu detektiert werden
                track.box = None

    def update_tracks(self, prev_gray, frame_gray, fgmask, contours, vis=None, collisions=None, counter=None):

        #if len(collisions) > 0:
            #self.handle_collision(collisions, contours, vis, frame_gray, fgmask)

        # TODO
        for track in self.tracks:
            if not track.lost:
                if len(collisions) > 0:
                    track.update(prev_gray, frame_gray, fgmask, contours, self.lk_params, vis, collisions, counter)
                else:
                    track.update(prev_gray, frame_gray, fgmask, contours, self.lk_params, vis, [], counter)

    def check_collision(self):
        collisions = set()
        for i, track1 in enumerate(self.tracks):
            for j, track2 in enumerate(self.tracks):
                if i >= j:  # Avoid duplicate checks
                    continue
                if track1.box is not None and track2.box is not None and compute_iou(track1.box, track2.box) > 0.65:
                    collisions.add(track1)
                    collisions.add(track2)
        return list(collisions)

    def setup_collision_matrix(self, collisions, merged_box, contours, vis, frame_gray, mask):

        best_cost = float('inf')
        foreground_track = None
        contours = merge_contours(contours)

        # Berechne Eigenschaften für die Merged Box
        p, merged_hist = calculate_color_histogram(vis, merged_box, contours, mask)
        merged_hog_descriptor = calculate_hog_descriptor(frame_gray, merged_box, mask)

        for track in collisions:
            # Histogrammähnlichkeit: kleiner ist besser
            hist_sim = compare_histograms(merged_hist, track.hist)

            # HOG-Ähnlichkeit
            hog_sim = np.linalg.norm(merged_hog_descriptor - track.hog_descriptor)

            # Bewegungsähnlichkeit: basierend auf dem Zentrum und Bewegung des Tracks
            movement_cost = calculate_movement_similarity(track, merged_box)

            # Berechne Gesamtkosten
            total_cost = (
                    0.85 * hist_sim +  # Gewicht für Histogramm
                    0.10 * min(hog_sim / 100.0, 1.0) +  # Gewicht für HOG
                    0.05 * movement_cost
            )

            # Greedy: Überprüfen, ob diese Kosten die niedrigsten sind
            if total_cost < best_cost:
                best_cost = total_cost
                foreground_track = track

        return foreground_track


class Track:
    count = 0

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
        self.ref_box = None
        self.non_detects = 0


    def draw_track(self, frame):
        # Zeichne die Historie der Positionen der Person im Video
        for i in range(1, len(self.trace)):
            # Hole die Positionen der aktuellen und der vorherigen Frame
            prev_center = self.trace[i - 1][0]
            curr_center = self.trace[i][0]
            curr_center[0] = int(curr_center[0])
            curr_center[1] = int(curr_center[1])

            prev_center = tuple(prev_center) if isinstance(prev_center, list) else prev_center
            curr_center = tuple(curr_center) if isinstance(curr_center, list) else curr_center

            # Zeichne eine Linie von der vorherigen Position zur aktuellen Position
            cv.line(frame, prev_center, curr_center, (255 // self.id, 0, 255 // self.id), 2)  # Rot, Linie dick

        # Optional: Zeichne die letzten Position als Punkt (optional)
        # cv.circle(frame, tuple(self.center), 5, (0, 255, 0), -1)  # Grün, Mittelpunkt der letzten Position

    @staticmethod
    def draw_meanshift_vector(vis, box, shift):
        if vis is not None:
            x, y, w, h = box
            cv.arrowedLine(vis, (int(x + w // 2), int(y + h // 2)),
                           (int(x + w // 2 + shift[0]), int(y + h // 2 + shift[1])),
                           (0, 255, 0), 2)

    def update(self, prev_gray, frame_gray, fgmask, contours, lk_params, vis, collisions, counter):
        buffer = 20
        alpha = 0.15
        min_height, min_width = 50, 50
        x, y, w, h = self.box

        # Berechne optischen Fluss

        features = np.float32(self.features).reshape(-1, 1, 2)
        p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, features, None, **lk_params)
        p0, st0, err0 = cv.calcOpticalFlowPyrLK(frame_gray, prev_gray, p1, None, **lk_params)

        # Filtere valide Punkte
        valid_points, previous_points = self._filter_valid_points(p1, st, p0, fgmask, (x-40, y-40, w+80, h+80))  # features

        if len(valid_points) >= 10:

            movement = valid_points - previous_points

            mean_shift = np.mean(movement, axis=0)
            historical_mean_shift = np.mean(self.mean_shift[-3:], axis=0) if len(self.mean_shift) > 3 else mean_shift
            smooth_shift = 0.8 * mean_shift + 0.2 * historical_mean_shift  # 0.5 0.5

            #self.draw_meanshift_vector(vis, self.box, smooth_shift)

            # Box-Parameter aktualisieren
            x, y, w, h = self.box
            dx, dy = smooth_shift
            box_center = np.array([x + dx + w // 2, y + dy + h // 2])

            #self.draw_track(vis)

            # Neue dynamische Box berechnen
            valid_y_coords = valid_points[:, 1]
            valid_x_coords = valid_points[:, 0]
            min_y = int(np.min(valid_y_coords)) - buffer
            max_y = int(np.max(valid_y_coords)) + buffer
            min_x = int(np.min(valid_x_coords)) - buffer
            max_x = int(np.max(valid_x_coords)) + buffer
            dynamic_height = min(max_y - min_y, self.ref_box[3])
            dynamic_width = min(max_x - min_x, self.ref_box[2])# Boxen kleiner machen

            if contours and len(collisions) == 0:
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

            self.features = valid_points.tolist()
            self.mean_shift.append(smooth_shift)
            self.center = [(new_box[0] + new_box[2] // 2), (new_box[1] + new_box[3] // 2)]
            self.trace.append([self.center])

            if not self.lost and new_box[2] * new_box[3] > self.ref_box[2] * self.ref_box[3] // 4:
                self.box = new_box
        else:
            self.lost = True

    @staticmethod
    def _filter_valid_points(p1, st, features, fgmask, box):
        valid_points = p1[st == 1].reshape(-1, 2)
        previous_points = features[st == 1].reshape(-1, 2)

        x, y, w, h = box

        points_within_box = [
            (x <= p[0] <= x + w and y <= p[1] <= y + h) for p in valid_points
        ]
        # Filter: Punkte müssen innerhalb der Box liegen und im Vordergrund sein
        points_in_fg = [
            not np.all(fgmask[int(p[1]):int(p[1]) + 25, int(p[0]):int(p[0]) + 25] == 0) for p in valid_points
        ]

        mask_filter = np.array(points_within_box, dtype=bool) & np.array(points_in_fg, dtype=bool)  # np.array(points_within_box) &
        # TODO:  return valid_points[mask_filter], previous_points[mask_filter]
        # IndexError: arrays used as indices must be of integer (or boolean) type
        return valid_points[mask_filter], previous_points[mask_filter]
