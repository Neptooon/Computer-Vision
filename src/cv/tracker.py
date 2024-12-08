import cv2 as cv
import numpy as np
# TODO: Virtualisieren der Boxen fehlt

class BGS:
    def __init__(self):
        self.backgroundSubtraction = cv.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=50)
        self.backgroundSubtraction.setBackgroundRatio(0.7)
        self.backgroundSubtraction.setShadowValue(255)
        self.backgroundSubtraction.setShadowThreshold(0.3)
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    def bgs_apply(self, frame):

        fgmask = self.backgroundSubtraction.apply(frame)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, self.kernel)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, self.kernel)
        return cv.medianBlur(fgmask, 3)


class Detector:
    def __init__(self):
        self.hog = cv.HOGDescriptor()
        self.hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame, fgmask):

        frame_down_sample = cv.resize(frame, ((frame.shape[1] // 25) * 10, (frame.shape[0] // 25) * 10))
        boxes, weights = self.hog.detectMultiScale(frame_down_sample, winStride=(8, 8), padding=(8, 8), scale=1.06)
        boxes = np.divide(boxes * 25, 10).astype(int)  # Up-sampling der Koordinaten

        return self.filter_boxes(boxes, fgmask)

    @staticmethod
    def _inside(r, q):
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q
        return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

    def filter_boxes(self, boxes, fgmask, min_area=1000):

        filtered_boxes = []
        for ri, r in enumerate(boxes):
            for qi, q in enumerate(boxes):
                if ri != qi and self._inside(r, q):
                    break
            else:
                filtered_boxes.append(r)
                print("FILTER")
        return [box for box in filtered_boxes if np.count_nonzero(fgmask[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]) > min_area]



class Tracker:
    def __init__(self):
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 15, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7) # 21 bzw. 7
        self.box_tracks = []
        self.last_box_tracks = []
        self.updated_box_tracks = []
        self.min_width = 50

    def reinitialize_features(self, frame_gray, box, contours=None):
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

            # Begrenzungen des Bildes berücksichtigen
            roi_x_start = max(0, roi_x_start)
            roi_y_start = max(0, roi_y_start)
            roi_x_end = min(frame_gray.shape[1], roi_x_end)
            roi_y_end = min(frame_gray.shape[0], roi_y_end)
        else:
            # Wenn keine Kontur vorhanden ist, nur die Box verwenden
            roi_x_start, roi_y_start, roi_x_end, roi_y_end = x, y, x + w, y + h

        # Begrenzungen des Bildes berücksichtigen
        roi_x_start = max(0, roi_x_start)
        roi_y_start = max(0, roi_y_start)
        roi_x_end = min(frame_gray.shape[1], roi_x_end)
        roi_y_end = min(frame_gray.shape[0], roi_y_end)

        # ROI aus dem Graustufenbild extrahieren
        roi = frame_gray[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        # Neue Features innerhalb des erweiterten ROI berechnen
        features = cv.goodFeaturesToTrack(roi, **self.feature_params)

        # Features von ROI-Koordinaten in globale Bildkoordinaten umrechnen
        if features is not None:
            return [(px + roi_x_start, py + roi_y_start) for px, py in np.float32(features).reshape(-1, 2)]
        else:
            return []

    def init_tracks(self, boxes, frame_gray):

        for (x, y, w, h) in boxes:
            roi = frame_gray[y:y + h, x:x + w]  # ROI um Feature-Suche einzugrenzen
            features = cv.goodFeaturesToTrack(roi, **self.feature_params)
            self.min_width = w
            if features is not None:
                self.box_tracks.append(
                    {
                        "box": (x, y, w, h),
                        "features": [(px + x, py + y) for px, py in np.float32(features).reshape(-1, 2)],
                        "mean_shift": [0, 0],
                        "center": [(x + (w//2)), (y + (h // 2))]
                    }
                )

    def update_tracks(self, prev_gray, frame_gray, fgmask, contours, frame_counter, vis=None):

        points = []
        min_height = 50  # Mindesthöhe der Bounding-Box
        buffer = 20  # Puffer, um Featurepunkte vollständig einzuschließen
        alpha = 0.3  # Glättungsfaktor
        for track in self.box_tracks:
            box = track["box"]
            features = np.float32(track["features"]).reshape(-1, 1, 2)

            p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, features, None, **self.lk_params)
            p0, st0, err0 = cv.calcOpticalFlowPyrLK(frame_gray, prev_gray, p1, None, **self.lk_params)
            valid_points, previous_points = self._filter_valid_points(p1, st, p0, track, fgmask)

            if len(valid_points) > 0:
                # Berechne die durchschnittliche Verschiebung der Punkte
                points.append(valid_points)
                movement = valid_points - previous_points
                mean_shift = np.mean(movement, axis=0)

                x, y, w, h = box
                dx, dy = mean_shift
                # Berechne den neuen Mittelpunkt der Bounding-Box
                box_center = np.array([x + dx + w // 2, y + dy + h // 2])

                # Nutze Feature-Punkte zur Höhenanpassung
                valid_y_coords = valid_points[:, 1]
                valid_x_coords = valid_points[:, 0]
                min_y = int(np.min(valid_y_coords)) - buffer
                max_y = int(np.max(valid_y_coords)) + buffer
                min_x = int(np.min(valid_x_coords)) - buffer
                max_x = int(np.max(valid_x_coords)) + buffer

                # Berechne dynamische Höhe basierend auf Feature-Punkten
                dynamic_height = max(max_y - min_y, min_height)
                dynamic_width = max(max_x - min_x, self.min_width)  # Breite bleibt stabil oder wird angepasst

                # Konturen verwenden, um den Mittelpunkt zu aktualisieren
                if contours:
                    filtered_contours = [contour for contour in contours if cv.contourArea(contour) >= 900]

                    cv.drawContours(vis, filtered_contours, -1,
                                    (0, 255, 0), 2)

                    if filtered_contours:
                        contour_centers = []
                        for contour in filtered_contours:
                            # Berechne jeden Mittelpunkt einer Kontur
                            M = cv.moments(contour)
                            if M["m00"] > 0:
                                cx = int(M["m10"] / M["m00"]) # x-Koordinate des Mittelpunkts
                                cy = int(M["m01"] / M["m00"]) - buffer # y-Koordinate des Mittelpunkts
                                contour_centers.append((cx, cy))

                        if contour_centers:
                            # Berechnung der euklidische Distanz zur aktuellen Box
                            # Abstand zwischen dem Mittelpunkt der Box (box_center) und den Mittelpunkten der Konturen
                            distances = [np.linalg.norm(np.array(center) - box_center) for center in contour_centers]

                            # Gewichte basierend auf der Distanz berechnen
                            weights = 1 / (np.array(distances) + 1e-5)  # Vermeidung von Division durch Null
                            weights /= weights.sum()  # Normalisierung der Gewichte
                            # Gewichteten Mittelwert der Konturzentren berechnen
                            weighted_center = np.sum(np.array(contour_centers) * weights[:, None], axis=0)

                            # Glättung zwischen aktuellem Box-Mittelpunkt und gewichteter Mitte
                            smooth_center = (1 - alpha) * box_center + alpha * weighted_center
                            new_x = int(smooth_center[0] - dynamic_width // 2)
                            new_y = int(smooth_center[1] - dynamic_height // 2)

                            # Box updaten und wenn notwendig, Featurepunkte neu berechnen
                            self.updated_box_tracks.append({
                                "box": (new_x, new_y, dynamic_width, dynamic_height),
                                "features": valid_points.tolist() if frame_counter % 5 != 0 else
                                        self.reinitialize_features(
                                                frame_gray, (new_x, new_y, w, h), filtered_contours),
                                "mean_shift": mean_shift,
                                "center": [new_x + dynamic_width // 2, new_y + dynamic_height // 2]
                            })

        return points


    @staticmethod
    def _filter_valid_points(p1, st, features, box, fgmask):
        #p1[box["center"][1] - 5:box["center"][1] + 5, box["center"][0] - 5:box["center"][0] + 5]
        valid_points = p1[st == 1].reshape(-1, 2)
        previous_points = features[st == 1].reshape(-1, 2)
        mask_filter = [
            not np.all(fgmask[int(p[1]):int(p[1]) + 25, int(p[0]):int(p[0]) + 25] == 0)
            #and (int(p[1]) >= int(box["center"][1]) - 5) and (int(p[1]) <= int(box["center"][1]) + 5)
            #and (int(p[0]) >= int(box["center"][0]) - 5) and (int(p[0]) <= int(box["center"][0]) + 5)
            for p in valid_points
        ]
        return valid_points[mask_filter], previous_points[mask_filter]

    def virtual_movement(self):
        print("VIRT")
        for track in self.last_box_tracks:
            x, y, w, h = track["box"]
            x = np.int32(x + track["mean_shift"][0])
            #y = np.int32(y + track["mean_shift"][1]) # Y-nicht
            track["box"] = (x, y, w, h)
            track["center"] = [(x + (w // 2)), (y + (h // 2))]

    def init_new_tracks(self):

        self.box_tracks = self.updated_box_tracks
        self.last_box_tracks = self.updated_box_tracks
        self.updated_box_tracks = []

    def draw_boxes(self, vis, box_tracks):
        for track in box_tracks:
            x, y, w, h = track["box"]
            cv.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def draw_features(self, vis, features, box_tracks):
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
        self.prev_gray = None
        self.frame_counter = 0

    def run(self):

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            vis = frame.copy()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            fgmask = self.bgs.bgs_apply(frame)
            points = []

            self.frame_counter += 1
            if len(self.tracker.box_tracks) < 1:
                print("DETEKT")
                self.tracker.box_tracks.clear()
                boxes = self.detector.detect(frame, fgmask)
                self.tracker.init_tracks(boxes, frame_gray)
            else:
                print("TRACK", len(self.tracker.box_tracks[0]["features"]))

                contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                if contours:
                    points = self.tracker.update_tracks(self.prev_gray, frame_gray, fgmask, contours, self.frame_counter, vis)
                self.tracker.init_new_tracks()
                self.tracker.draw_features(vis, points, self.tracker.box_tracks)
                self.tracker.draw_boxes(vis, self.tracker.box_tracks)

            self.prev_gray = frame_gray

            cv.imshow('HOG', vis)
            cv.imshow('BG', fgmask)
            key = cv.waitKey(30)

            if key & 0xFF == 27:
                break

            if key == ord('p'):
                cv.waitKey(-1)

        self.cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    pipeline = SingleObjectTrackingPipeline('../../assets/videos/LL-Default-Tshirt-Hell-RL.mov')
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