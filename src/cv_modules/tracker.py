import cv2 as cv
import numpy as np
from IoU import IoUMetrik


class BGS: # BGS
    def __init__(self):
        self.backgroundSubtraction = cv.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=75)  # 70
        self.backgroundSubtraction.setBackgroundRatio(0.7)
        self.backgroundSubtraction.setShadowValue(255)
        self.backgroundSubtraction.setShadowThreshold(0.2)  # 0.3
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    def bgs_apply(self, frame):
        fgmask = self.backgroundSubtraction.apply(frame)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, self.kernel)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, self.kernel)
        return cv.medianBlur(fgmask, 3)


class Detector: # Detektor
    def __init__(self): # Hog-Deskriptor
        self.hog = cv.HOGDescriptor()
        self.hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

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


def merge_contours(contours, max_gap=100):  # Merged gefundene Konturen zu einer konvexen Hülle zusammen
    if not contours:
        return []

    # Sortiert Konturen nach y-koordinaten der bbox
    contours = sorted(contours, key=lambda cnt: cv.boundingRect(cnt)[1])

    merged_contours = []
    used = [False] * len(contours)  # Speichert gemerged Konturen

    for i, contour_a in enumerate(contours):
        if used[i]:
            continue

        # Bbox der 1. Kontur
        x_a, y_a, w_a, h_a = cv.boundingRect(contour_a)
        merged = contour_a

        for j, contour_b in enumerate(contours[i + 1:], start=i + 1):
            if used[j]:
                continue

            # Bbox der 2. Kontur
            x_b, y_b, w_b, h_b = cv.boundingRect(contour_b)

            # Merge Kriterium: vertikaler Abstand und horz. Überlappung
            vertical_gap = y_b - (y_a + h_a)
            horizontal_overlap = min(x_a + w_a, x_b + w_b) - max(x_a, x_b)

            if vertical_gap <= max_gap and horizontal_overlap > 0:
                # Beide Konturen Mergen
                merged = np.vstack((merged, contour_b))
                # Bbox updaten der gemerged Kontur
                x_a, y_a, w_a, h_a = cv.boundingRect(merged)
                used[j] = True

        # Konvexe Hülle der gemerged Kontur
        hull = cv.convexHull(merged)
        merged_contours.append(hull)
        used[i] = True

    return merged_contours


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

    def init_tracks(self, boxes, frame_gray):  # Tracks initialisieren

        for (x, y, w, h) in boxes:
            roi = frame_gray[y:y + h, x:x + w]  # ROI um Feature-Suche einzugrenzen
            features = cv.goodFeaturesToTrack(roi, **self.feature_params)
            if features is not None:
                self.box_tracks.append(
                    {
                        "box": (x, y, w, h),
                        "features": [(px + x, py + y) for px, py in np.float32(features).reshape(-1, 2)],
                        "mean_shift": [[0, 0]],
                        "center": [(x + (w // 2)), (y + (h // 2))]
                    }
                )

    def update_tracks(self, prev_gray, frame_gray, fgmask, contours, frame_counter, vis=None):  # Tracks updaten

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
                    filtered_contours = [contour for contour in contours if cv.contourArea(contour) >= 1000]
                    self.lock = 1
                    filtered_contours = merge_contours(filtered_contours)

                    # Todo Hier stand Virt

                    cv.drawContours(vis, filtered_contours, -1, (0, 0, 255), 2)
                    if filtered_contours:
                        contour_centers = []

                        for contour in filtered_contours:
                            # Berechne jeden Schwerpunkt einer Kontur
                            M = cv.moments(contour)
                            cx = int(M["m10"] / M["m00"] + 1e-5)  # x-Koordinate
                            cy = int(M["m01"] / M["m00"] + 1e-5) - buffer  # y-Koordinate
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
                                    frame_gray, (new_x, new_y, w, h), filtered_contours),
                                "mean_shift": (prev_mean_shift := [[dx, dy]] if prev_mean_shift is None or len(
                                    prev_mean_shift) > 25 else prev_mean_shift + [[dx, dy]]),
                                "center": [new_x + dynamic_width // 2, new_y + dynamic_height // 2]
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
                cx = int(M["m10"] / M["m00"] + 1e-5)  # x-Koordinate
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
                    "center": [new_x + dynamic_width // 2, new_y + dynamic_height // 2]
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
                self.draw_features(vis, points, self.last_box_tracks)
                self.draw_boxes(vis, self.last_box_tracks)
                self.virt_frame_counter += 1

    def init_new_tracks(self):  # Neue Tracks init

        self.box_tracks = self.updated_box_tracks
        if len(self.box_tracks) != 0 and self.lock:
            self.last_box_tracks = self.updated_box_tracks
        self.updated_box_tracks = []

    def draw_boxes(self, vis, box_tracks):  # Boxen zeichnen
        for track in box_tracks:
            x, y, w, h = track["box"]
            cv.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def draw_features(self, vis, features, box_tracks):  # Feature zeichnen
        if features is not None:
            for feature_list in features:
                for i, point in enumerate(feature_list):
                    cv.circle(vis, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)

        if box_tracks is not None:
            for track in box_tracks:
                if track["center"] is not None:
                    cv.circle(vis, (int(track["center"][0]), int(track["center"][1])), 2, (0, 0, 255), 2)


class SingleObjectTrackingPipeline:
    def __init__(self, video_path):
        self.cap = cv.VideoCapture(video_path)
        self.bgs = BGS()
        self.detector = Detector()
        self.tracker = Tracker()
        self.iou_metrik = IoUMetrik(video_path)
        self.prev_gray = None
        self.width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)


        # Todo -- Nur für Metrik zwecke angelegt --
        self.frame_counter = 0
        self.detect_counter = 0
        self.tracking_counter = 0
        self.empty = 0

    def run(self):

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            vis = frame.copy()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            fgmask = self.bgs.bgs_apply(frame)
            points = []

            if len(self.tracker.box_tracks) < 1:

                boxes = self.detector.detect(frame, fgmask)
                self.tracker.init_tracks(boxes, frame_gray)

                # -------------------------------------- TODO Nur für Metrik
                if len(self.tracker.box_tracks) >= 1:
                    self.detect_counter += 1
                else:
                    self.empty += 1
                # --------------------------------------
                self.tracker.check_virt(boxes, points, vis, self.height, self.width)

            else:
                self.tracker.virt_frame_counter = 0
                self.tracking_counter += 1
                contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                points = self.tracker.update_tracks(self.prev_gray, frame_gray, fgmask, contours, self.frame_counter,
                                                    vis)
                self.tracker.init_new_tracks()

            self.tracker.draw_features(vis, points, self.tracker.box_tracks)
            self.tracker.draw_boxes(vis, self.tracker.box_tracks)
            self.iou_metrik.get_iou_info(self.tracker.box_tracks, self.frame_counter)
            self.frame_counter += 1
            self.prev_gray = frame_gray

            cv.imshow('HOG', vis)
            #cv.imshow('BG', fgmask)
            key = cv.waitKey(10)

            if key & 0xFF == 27:
                break

            if key == ord('p'):
                cv.waitKey(-1)

        self.cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    pipeline = SingleObjectTrackingPipeline('../../assets/videos/DS-Parkour-Tshirt-Hell-RL.mov')
    pipeline.run()

'''
In die Pipeline Einbinden, falls bestimmte Funktionen nur auf die BoundingBox beschränkt werden sollen.

# Erweiterungsparameter für die ROI
padding = 20  # Erweiterung um 20 Pixel in alle Richtungen
# Aktuelle Bounding-Box
x, y, w, h = self.tracker.box_tracks[0]["box"]
# Erweiterte ROI berechnen
roi_x_start = max(0, x - padding)
roi_y_start = max(0, y - padding)
roi_x_end = min(fgmask.shape[1], x + w + padding)
roi_y_end = min(fgmask.shape[0], y + h + padding)
# Erweiterte ROI auf Vordergrundmaske anwenden
roi = fgmask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
'''

"""max_K_height = cv.boundingRect(max(merge_contours(filtered_contours), key=cv.contourArea))[3]
                    max_K_width = cv.boundingRect(max(merge_contours(filtered_contours), key=cv.contourArea))[2]

                    if max_K_height + max_K_width <= 450:
                        filtered_contours.clear()

                    if filtered_contours:
                        max_K_height = cv.boundingRect(max(merge_contours(filtered_contours), key=cv.contourArea))[3]
                        max_K_width = cv.boundingRect(max(merge_contours(filtered_contours), key=cv.contourArea))[2]

                        if max_K_height + max_K_width <= 450:
                            # if max_K_height <= 75 or max_K_width <= 150:
                            if max_K_height + max_K_width <= 350 and (max_K_height * 2.5 < max_K_width or max_K_height > max_K_width * 2.5):
                                filtered_contours.clear()
                            self.virtual_movement_feet(track, i, filtered_contours, box_center, vis, valid_points,
                                                       frame_counter, frame_gray, prev_mean_shift, mean_shift)
                            continue"""

"""@staticmethod
    def _filter_valid_points(p1, st, features, fgmask, box):
        x, y, w, h = box
        valid_points = p1[st == 1].reshape(-1, 2)
        previous_points = features[st == 1].reshape(-1, 2)

        box_filter = [
            (x - 55 <= p[0] <= x + w + 55) and (y - 55 <= p[1] <= y + h + 55) for p in valid_points
        ]

        mask_filter = [  # +25
            not np.all(fgmask[int(p[1]):int(p[1]) + 5, int(p[0]):int(p[0]) + 5] == 0) for p in valid_points
        ]

        combined_filter = np.array(box_filter) & np.array(mask_filter)

        return valid_points[combined_filter], previous_points[combined_filter]"""
