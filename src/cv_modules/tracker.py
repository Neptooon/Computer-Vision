import cv2 as cv
import numpy as np

from src.cv_modules.helpers import calculate_color_histogram, draw_boxes, draw_features


class Tracker:
    def __init__(self):
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 15, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7)  # 21 bzw. 7
        self.box_tracks = []
        self.last_box_tracks = []
        self.updated_box_tracks = []
        self.last_partial_virt_box_tracks = []
        self.lock = 1
        self.virt_frame_counter = 0

    def reinitialize_features(self, frame_gray, box, contours=None):  # Feature werden alle X Frame reinitialisiert
        x, y, w, h = box

        if contours is not None and len(contours) > 0:
            # Initialisieren der kombinierten ROI mit der Box
            roi_x_start, roi_y_start = x, y
            roi_x_end, roi_y_end = x + w, y + h

            for contour in contours:
                if isinstance(contour, np.ndarray) and len(contour) > 0:
                    # Konturkoordinaten berechnen
                    contour_x, contour_y, contour_w, contour_h = cv.boundingRect(contour)
                    # ROI erweitern basierend auf Konturen
                    roi_x_start = min(roi_x_start, contour_x)
                    roi_y_start = min(roi_y_start, contour_y)
                    roi_x_end = max(roi_x_end, contour_x + contour_w)
                    roi_y_end = max(roi_y_end, contour_y + contour_h)

            # Begrenzungen
            roi_x_start = max(0, roi_x_start)
            roi_y_start = max(0, roi_y_start)
            roi_x_end = min(frame_gray.shape[1], roi_x_end)
            roi_y_end = min(frame_gray.shape[0], roi_y_end)
        else:
            # Wenn keine Kontur vorhanden ist nur Box verwenden
            roi_x_start, roi_y_start, roi_x_end, roi_y_end = x, y, x + w, y + h

        # Begrenzungen
        roi_x_start = max(0, roi_x_start)
        roi_y_start = max(0, roi_y_start)
        roi_x_end = min(frame_gray.shape[1], roi_x_end)
        roi_y_end = min(frame_gray.shape[0], roi_y_end)

        # ROI aus Graustufenbild extrahieren
        roi = frame_gray[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        # Neue Features innerhalb der neuen ROI berechnen
        features = cv.goodFeaturesToTrack(roi, **self.feature_params)

        # Features von ROI-Koordinaten in globale Bildkoordinaten umrechnen
        if features is not None:
            return [(px + roi_x_start, py + roi_y_start) for px, py in np.float32(features).reshape(-1, 2)]
        else:
            return []

    def init_tracks(self, boxes, frame_gray, vis, detector, contours):  # Tracks initialisieren
        self.box_tracks.clear()
        for (x, y, w, h) in boxes:
            roi = frame_gray[y:y + h, x:x + w]  # ROI um Feature-Suche einzugrenzen
            features = cv.goodFeaturesToTrack(roi, **self.feature_params)
            features = [(px + x, py + y) for px, py in np.float32(features).reshape(-1, 2)]
            hist = calculate_color_histogram(vis, [x, y, w, h], contours)
            id = detector.create_id(hist, features)

            if features is not None:
                self.box_tracks.append(
                    {
                        "box": (x, y, w, h),
                        "features": features,
                        "mean_shift": [[0, 0]],
                        "center": [(x + (w // 2)), (y + (h // 2))],
                        "id": id,
                        "hist": hist
                    }
                )

    def update_tracks(self, prev_gray, frame_gray, fgmask, contours, frame_counter, detector, vis=None):  # Tracks updaten

        if not contours:
            return

        points = []
        min_height = 50  # Mindesthöhe der bbox
        min_width = 50  # Mindestbreite der bbox
        buffer = 20  # Puffer, um Featurepunkte vollständig einzuschließen
        alpha = 0.15  # Glättungsfaktor um bbox Schwankung zu reduzieren


        for i, track in enumerate(self.box_tracks):
            box = track["box"]
            features = np.float32(track["features"]).reshape(-1, 1, 2)
            prev_mean_shift = track["mean_shift"]  # Vorherige durchsch. Verschiebung

            # Vorwärts- und Rückwärtsfluss
            p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, features, None, **self.lk_params)
            p0, st0, err0 = cv.calcOpticalFlowPyrLK(frame_gray, prev_gray, p1, None, **self.lk_params)
            valid_points, previous_points = self._filter_valid_points(p1, st, p0, fgmask)  # Punkte in Roi behalten

            if len(valid_points) > 0:
                # Durchschnittliche Verschiebung der Punkte

                points.append(valid_points)
                movement = valid_points - previous_points
                mean_shift = np.mean(movement, axis=0)

                x, y, w, h = box
                dx, dy = mean_shift
                # Berechne den neuen Mittelpunkt der bbox
                box_center = np.array([x + dx + w // 2, y + dy + h // 2])

                #Feature-Punkte zur Höhenanpassung
                valid_y_coords = valid_points[:, 1]
                valid_x_coords = valid_points[:, 0]
                min_y = int(np.min(valid_y_coords)) - buffer
                max_y = int(np.max(valid_y_coords)) + buffer
                min_x = int(np.min(valid_x_coords)) - buffer
                max_x = int(np.max(valid_x_coords)) + buffer

                # Berechne dynamische Höhe basierend auf Feature-Punkten
                dynamic_height = max(max_y - min_y, min_height)
                dynamic_width = max(max_x - min_x, min_width)  # Breite bleibt stabil oder wird angepasst

                # Konturen verwenden, um Mittelpunkt zu aktualisieren

                if contours:
                    self.lock = 1

                    # Todo Hier stand Virt


                    contour_centers = []

                    for contour in contours:
                        # Berechne jeden Schwerpunkt einer Kontur
                        M = cv.moments(contour)
                        cx = int(M["m10"] / (M["m00"] + 1e-5))  # x-Koordinate
                        cy = int(M["m01"] / (M["m00"] + 1e-5)) - buffer  # y-Koordinate
                        if min_x <= cx <= min_x + dynamic_width or min_y <= cy <= min_y + dynamic_height:
                            contour_centers.append((cx, cy))

                    if contour_centers:
                        # Euklidische Distanz zur aktuellen Box
                        # Abstand zwischen dem Mittelpunkt der Box (box_center) und den Schwerpunkten der Konturen
                        distances = [np.linalg.norm(np.array(center) - box_center) for center in contour_centers]

                        # Gewichte basierend auf der Distanz
                        weights = 1 / (np.array(distances) + 1e-5)  # Vermeidung von Division durch 0
                        weights /= np.sum(
                            weights)  # Normalisierung der Gewichte d.h immer 1 wenn nur 1 Kontur da ist

                        # Gewichteten Mittelwert der Schwerpunkte berechnen
                        weighted_center = np.sum(np.array(contour_centers) * weights[:, None], axis=0)

                        # Glätttung zwischen aktuellem Box-Mittelpunkt und gewichtetem Schwerpunkt
                        even_center = (1 - alpha) * box_center + alpha * weighted_center
                        new_x = int(even_center[0] - dynamic_width // 2)
                        new_y = int(even_center[1] - dynamic_height // 2)

                        # Box updaten und wenn notwendig, Featurepunkte neu berechnen
                        self.updated_box_tracks.append({
                            "box": (new_x, new_y, dynamic_width, dynamic_height),
                            "features": valid_points.tolist() if frame_counter % 5 != 0 else
                            self.reinitialize_features(
                                frame_gray, (new_x, new_y, w, h), contour),
                            "mean_shift": (prev_mean_shift := [[dx, dy]] if prev_mean_shift is None or len(
                                prev_mean_shift) > 25 else prev_mean_shift + [[dx, dy]]),
                            "center": [new_x + dynamic_width // 2, new_y + dynamic_height // 2],
                            "id": track["id"],
                            "hist": track["hist"]
                        })

        return points

    @staticmethod
    def _filter_valid_points(p1, st, features, fgmask):  # Nur Punkte in Roi werden als valid eingestuft
        valid_points = p1[st == 1].reshape(-1, 2)
        previous_points = features[st == 1].reshape(-1, 2)
        mask_filter = [  # +25
            not np.all(fgmask[int(p[1]):int(p[1]) + 5, int(p[0]):int(p[0]) + 5] == 0) for p in valid_points
        ]
        return valid_points[mask_filter], previous_points[mask_filter]

    def virtual_movement(self):
        print("VIRT")
        if len(self.last_partial_virt_box_tracks) > 0:
            for track in self.last_partial_virt_box_tracks:
                x, y, w, h = track["box"]
                x = np.int32(x + np.mean([shift[0] for shift in track["mean_shift"]]) * 2)
                track["box"] = (x, y, w, h)
                track["center"] = [(x + (w // 2)), (y + (h // 2))]
            self.last_box_tracks = self.last_partial_virt_box_tracks.copy()
            self.last_partial_virt_box_tracks = []
        else:
            for track in self.last_box_tracks:
                x, y, w, h = track["box"]
                x = np.int32(x + np.mean([shift[0] for shift in track["mean_shift"]]) * 2)
                # y = np.int32(y + track["mean_shift"][1]) # Y-nicht
                track["box"] = (x, y, w, h)
                track["center"] = [(x + (w // 2)), (y + (h // 2))]

    def virtual_movement_partial(self, track, index, filtered_contours, box_center, vis,
                                 valid_points, frame_counter, frame_gray, prev_mean_shift, mean_shift):
        self.lock = 0
        x, y, w, h = track["box"]
        dx, dy, dw, dh = self.last_box_tracks[index]["box"]
        alpha = 0.15

        track["box"] = (x, dy, w, dh)
        track["center"] = [(x + (w // 2)), (dy + (dh // 2))]

        valid_y_coords = np.float32(self.last_box_tracks[index]["features"]).reshape(-1, 1, 2).reshape(-1, 2)[:, 1]
        valid_x_coords = np.float32(self.last_box_tracks[index]["features"]).reshape(-1, 1, 2).reshape(-1, 2)[:, 0]
        min_y = int(np.min(valid_y_coords)) - 20
        max_y = int(np.max(valid_y_coords)) + 20
        min_x = int(np.min(valid_x_coords)) - 20
        max_x = int(np.max(valid_x_coords)) + 20

        # Berechne dynamische Höhe basierend auf Feature-Punkten
        dynamic_height = max(max_y - min_y, 50)
        dynamic_width = max(max_x - min_x, 50)  # Breite bleibt stabil oder wird angepasst

        if filtered_contours:
            cv.drawContours(vis, filtered_contours, -1,
                            (0, 255, 0), 2)
            contour_centers = []
            for contour in filtered_contours:

                M = cv.moments(contour)
                cx = int(M["m10"] / (M["m00"] + 1e-5))  # x-Koordinate
                cy = self.last_box_tracks[index]["center"][1]  # y-Koordinate
                contour_centers.append((cx, cy))

            if contour_centers:

                distances = [np.linalg.norm(np.array(center) - box_center) for center in contour_centers]

                weights = 1 / (np.array(distances) + 1e-5)
                weights /= np.sum(weights)

                weighted_center = np.sum(np.array(contour_centers) * weights[:, None], axis=0)

                even_center = (1 - alpha) * box_center + alpha * weighted_center
                new_x = int(even_center[0] - dynamic_width // 2)
                new_y = int(even_center[1] - dynamic_height // 2)

                self.updated_box_tracks.append({
                    "box": (new_x, new_y, dynamic_width, dynamic_height),
                    "features": valid_points.tolist() if frame_counter % 5 != 0 else
                    self.reinitialize_features(
                        frame_gray, (new_x, new_y, w, h), filtered_contours),
                    "mean_shift": (
                        prev_mean_shift := [[mean_shift[0], mean_shift[1]]] if prev_mean_shift is None or len(
                            prev_mean_shift) > 25
                        else prev_mean_shift + [[mean_shift[0], mean_shift[1]]]),
                    "center": [new_x + dynamic_width // 2, new_y + dynamic_height // 2],
                    "id": track["id"],
                    "hist": track["hist"]
                })

                self.last_partial_virt_box_tracks = self.updated_box_tracks.copy()

    def check_virt(self, boxes, points, vis, height, width):
        if len(boxes) <= 0 < len(self.last_box_tracks) and self.virt_frame_counter <= 50:
            margin = 50  # Abstand vom Kamerarand in Pixeln
            valid_tracks = []
            for track in self.last_box_tracks:
                x, y, w, h = track["box"]
                # Bedingung: Box nicht in der Nähe des Randes
                if x > margin and y > margin and (x + w) < (width - margin) and (y + h) < (
                        height - margin):
                    valid_tracks.append(track)  # Box ist gültig

            if valid_tracks:  # Nur wenn es gültige Tracks gibt
                self.last_box_tracks = valid_tracks
                self.virtual_movement()
                draw_features(vis, points, self.last_box_tracks)
                draw_boxes(vis, self.last_box_tracks)
                self.virt_frame_counter += 1

    def init_new_tracks(self):  # Neue Tracks init

        self.box_tracks = self.updated_box_tracks
        if len(self.box_tracks) != 0 and self.lock:
            self.last_box_tracks = self.updated_box_tracks
        self.updated_box_tracks = []


class Track:

    def __init__(self):
        self.trace = []
