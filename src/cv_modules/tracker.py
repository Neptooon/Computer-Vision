import cv2 as cv
import numpy as np

from src.cv_modules.helpers import calculate_color_histogram, draw_boxes, draw_features, compare_histograms, \
    compute_iou, hog_descriptor_similarity, calculate_hog_descriptor, calculate_movement_similarity, merge_contours
from scipy.optimize import linear_sum_assignment


class Tracker:
    def __init__(self):
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 15, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7)
        self.tracks = []  # Liste von Track-Objekten
        self.id_count = 1
        self.sift = cv.SIFT.create(nfeatures=500)
        self.orb = cv.ORB.create(nfeatures=500)

    def start_new_track(self, detection, frame, vis, contours, fgmask):
        (x, y, w, h) = detection
        roi = vis[y:y + h, x:x + w]
        keypoints, descriptors = self.sift.detectAndCompute(roi, fgmask)
        temp_points = []
        for kp in keypoints:
            global_x = kp.pt[0] + x
            global_y = kp.pt[1] + y
            global_kp = cv.KeyPoint(global_x, global_y, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            temp_points.append(global_kp)
        keypoints = temp_points
        #keypoints, descriptors = self.orb.detectAndCompute(roi, None)

        hist = calculate_color_histogram(vis, [x, y, w, h], contours)
        hog = calculate_hog_descriptor(frame, detection)
        new_track = Track(box=(x, y, w, h), keypoints=keypoints, descriptor=descriptors, hist=hist, track_id=self.id_count, hog_deskriptor=hog)

        self.id_count += 1
        return new_track

    def update2(self, detections, frame, vis, contours, fgmask):

        if len(self.tracks) == 0 and len(detections) != 0:
            for box in detections:
                new_track = self.start_new_track(box, frame, vis, contours, fgmask)
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
                    print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
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

    def __init__(self, box, keypoints, descriptor, hist, track_id, hog_deskriptor):
        self.box = box  # (x, y, w, h)
        self.features = []  # Liste der Feature-Punkte
        self.hist = hist  # Farb-Histogramm der Box
        self.id = track_id  # Eindeutige ID
        self.center = [(box[0] + box[2] // 2), (box[1] + box[3] // 2)]  # Mittelpunkt der Box
        self.mean_shift = [np.array([0, 0])]  # Bewegungsgeschichte (Verschiebung)
        self.trace = [[self.center]]  # Historie der Positionen
        self.skipped_frames = 0
        self.lost = False
        self.hog_descriptor = hog_deskriptor

        self.orb = cv.ORB.create(nfeatures=500)
        self.sift = cv.SIFT.create(nfeatures=500)
        self.bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        self.sift_descriptor = descriptor
        self.sift_keypoints = keypoints

    def update(self, prev_gray, frame_gray, fgmask, contours, lk_params, vis=None):
        buffer = 20
        roi_buffer = 40
        min_height, min_width = 500, 200
        alpha = 0.15

        x, y, w, h = self.box
        x = int(max(0, x - roi_buffer))
        y = int(max(0, y - roi_buffer))
        w = int(min(w + 2 * roi_buffer, vis.shape[1] - x))
        h = int(min(h + 2 * roi_buffer, vis.shape[0] - y))
        roi = vis[y:y + h, x:x + w]
        cv.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

        keypoints, descriptors = self.sift.detectAndCompute(roi, None)
        #keypoints, descriptors = self.orb.detectAndCompute(roi, fgmask)

        temp_points = []
        for kp in keypoints:
            global_x = kp.pt[0] + x
            global_y = kp.pt[1] + y
            global_kp = cv.KeyPoint(global_x, global_y, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            temp_points.append(global_kp)
        keypoints = temp_points

        box_center = np.array([x + w // 2, y + h // 2])
        if descriptors is None:
            descriptors = np.array([])

        matches = self.bf.match(self.sift_descriptor, descriptors)
        for match in matches:
            cv.circle(vis, (int(keypoints[match.trainIdx].pt[0]), int(keypoints[match.trainIdx].pt[1])), 2, (132 // self.id, 0, 255 // self.id), 2)
        matches = [m for m in matches if m.distance < 0.75 * max([m.distance for m in matches])]
        #matches = sorted(matches, key=lambda xx: xx.distance)

        #prev_pts = np.float32([self.sift_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        #curr_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        #points = np.array([kp.pt for kp in keypoints])
        points = np.array([keypoints[match.trainIdx].pt for match in matches])
        min_x_idx = np.argmin(points[:, 0])  # Index des minimalen X-Wertes
        max_x_idx = np.argmax(points[:, 0])  # Index des maximalen X-Wertes
        min_y_idx = np.argmin(points[:, 1])  # Index des minimalen Y-Wertes
        max_y_idx = np.argmax(points[:, 1])  # Index des maximalen Y-Wertes


        #movement = np.mean(curr_pts - prev_pts, axis=0)
        min_y = int(points[min_y_idx][1] - buffer)
        max_y = int(points[max_y_idx][1] + buffer)
        min_x = int(points[min_x_idx][0] - buffer)
        max_x = int(points[max_x_idx][0] + buffer)
        dynamic_height = min(max(max_y - min_y, min_height), 600)
        dynamic_width = min(max(max_x - min_x, min_width), 250)


        if contours:
            contours = merge_contours(contours)
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
                new_box = (min_x, min_y, dynamic_width, dynamic_height)
        else:
            new_box = (min_x, min_y, dynamic_width, dynamic_height)

        #self.sift_keypoints = np.array([keypoints[m.trainIdx] for m in matches] + list(keypoints))
        #self.sift_keypoints = np.array([keypoints[m.trainIdx] for m in matches])
        #self.sift_descriptor = np.array([descriptors[m.trainIdx] for m in matches] + list(descriptors))
        #self.sift_descriptor = np.array([descriptors[m.trainIdx] for m in matches])
        self.box = new_box
        self.center = [(new_box[0] + new_box[2] // 2), (new_box[1] + new_box[3] // 2)]

    @staticmethod
    def _filter_valid_points(p1, st, features, fgmask):  # Nur Punkte in Roi werden als valid eingestuft
        valid_points = p1[st == 1].reshape(-1, 2)
        previous_points = features[st == 1].reshape(-1, 2)
        mask_filter = [  # +25
            not np.all(fgmask[int(p[1]):int(p[1]) + 25, int(p[0]):int(p[0]) + 25] == 0) for p in valid_points
        ]
        return valid_points[mask_filter], previous_points[mask_filter]