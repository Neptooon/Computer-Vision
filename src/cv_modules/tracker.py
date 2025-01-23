import cv2 as cv
import numpy as np

from src.cv_modules.helpers import calculate_color_histogram, compare_histograms, \
    compute_iou, calculate_hog_descriptor, calculate_movement_similarity, \
    merge_contours
from scipy.optimize import linear_sum_assignment


class Tracker:
    """
    Die Tracker-Klasse verfolgt Objekte über Frames hinweg.
    """
    def __init__(self):
        """
        Initialisiert die Tracker-Klasse mit Standardparametern und enthält eine Liste der zu verfolgenden Objekte / Tracks
        """
        self.lk_params = dict(winSize=(31, 31), maxLevel=4,  # Parameter für optischen Fluss
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=7, blockSize=7)  # Feature-Erkennung
        self.tracks = [] # Liste der zu verfolgenden Objekte
        self.id_count = [-1] * 2 # IDs der Tracks. Die Multiplikation ist die Anzahl der maximalen Personen die zugelassen werden.
        # * 2 = 2 Personen und * 4 = 4 Personen ...

    def start_new_track(self, detection, frame, vis, contours):
        """
        Startet einen neuen Track für eine gegebene Detektion.

        Args:
            detection (tuple): Bounding-Box der neuen Detektion (x, y, Breite, Höhe).
            frame (numpy.ndarray): Aktueller Frame.
            vis (numpy.ndarray): Visualisierungsframe.
            contours (list): Konturen zur Maske.
        Returns:
            Track: Neuer Track, falls erfolgreich, sonst None.
        """
        (x, y, w, h) = detection
        x = int(max(0, min(x, frame.shape[1] - 1)))
        y = int(max(0, min(y, frame.shape[0] - 1)))
        w = int(min(w, frame.shape[1] - x))
        h = int(min(h, frame.shape[0] - y))
        roi = frame[y:y + h, x:x + w]

        features = cv.goodFeaturesToTrack(roi, **self.feature_params)
        if features is not None:
            features = [(px + x, py + y) for px, py in np.float32(features).reshape(-1, 2)]
            contours = merge_contours(contours)
            p, hist = calculate_color_histogram(vis, detection, contours)
            hog = calculate_hog_descriptor(frame, detection)

            # Überprüft welche ID frei zum Vergeben ist
            set_id = 0
            for i, id in enumerate(self.id_count):
                if id == -1:
                    self.id_count[i] = 1
                    set_id = i + 1
                    break

            # Legt einen neuen Track an
            new_track = Track(box=(x, y, w, h), features=features, hist=hist, track_id=set_id,
                              hog_deskriptor=hog)
            new_track.ref_box = (x, y, w, h)

            return new_track
        return None

    def associate_tracks(self, detections, frame, vis, contours):
        """
        Aktualisiert bestehende Tracks oder erstellt neue basierend auf Detektionen.

        Args:
            detections (list): Liste der Bounding-Boxen der Detektionen.
            frame (numpy.ndarray): Aktueller Frame.
            vis (numpy.ndarray): Visualisierungsframe.
            contours (list): Konturen zur Maske.

        Beschreibung:
        1. Wenn keine bestehenden Tracks vorhanden sind, werden neue Tracks für alle Detektionen erstellt.
        2. Eine Kostenmatrix wird basierend auf Ähnlichkeitsmetriken zwischen bestehenden Tracks und Detektionen erstellt.
        3. Das Zuordnungsproblem wird gelöst mithilfe des Hungarian-Algorithmus, um Tracks den Detektionen zuzuweisen.
        4. Nicht zugewiesene Tracks werden als "skipped" markiert und gegebenenfalls entfernt.
        5. Neue Tracks werden für Detektionen erstellt, die keinem bestehenden Track zugewiesen wurden.
        6. Tracks werden mit neuen Features, Positionen und Histogrammen aktualisiert.

        """

        if len(self.tracks) == 0:
            # Keine bestehenden Tracks, neue Tracks erstellen
            for box in detections:
                new_track = self.start_new_track(box, frame, vis, contours)
                if new_track:
                    self.tracks.append(new_track)
            return

        # Kostenmatrix für die Zuordnung zwischen Tracks und Detektionen erstellen
        cost_matrix = self.setup_matrix(detections, contours, vis, frame)
        if not cost_matrix:
            return

        assignments = [-1] * len(self.tracks)
        unassigned = []

        # Löse das Zuordnungsproblem
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        for idx in range(len(track_indices)):
            assignments[track_indices[idx]] = detection_indices[idx]

        # Schwellenwert für die Kosten bestimmen
        cost_threshold = np.percentile(np.array(cost_matrix).flatten(), 80)

        # Tracks zu Detektionen zuordnen
        for idx in range(len(assignments)):
            if assignments[idx] != -1:
                if cost_matrix[idx][assignments[idx]] > cost_threshold:
                    assignments[idx] = -1
                    unassigned.append(idx)
            else:
                self.tracks[idx].skipped_frames += 1

        # Entferne Tracks, die zu lange verschwunden sind
        del_tracks = []
        for idx in range(len(self.tracks)):
            if self.tracks[idx].skipped_frames > 100:
                del_tracks.append(idx)

        # Tracks löschen
        if len(del_tracks) > 0:
            for idx in del_tracks:
                if idx < len(self.tracks):
                    del self.tracks[idx]
                    del assignments[idx]
                    self.id_count[idx] = -1
                else:
                    pass

        # Neue Tracks für nicht zugewiesene Detektionen erstellen
        unassigned_detections = []
        for idx in range(len(detections)):
            if idx not in assignments:
                unassigned_detections.append(idx)

        if len(unassigned_detections) != 0:
            for idx in range(len(unassigned_detections)):
                if not any([True for track in self.tracks if
                            compute_iou(track.box, detections[unassigned_detections[idx]]) > 0.0]) and self.id_count.count(-1) >= 1:
                    new_track = self.start_new_track(detections[unassigned_detections[idx]], frame, vis, contours)
                    self.tracks.append(new_track)

        # Aktualisiere bestehende Tracks mit neuen Informationen
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
                    self.tracks[idx].hog_descriptor = calculate_hog_descriptor(frame, self.tracks[idx].box)
                    p, new_hist = calculate_color_histogram(vis, self.tracks[idx].box, contours)
                    if p >= 36000 or p <= 10000:
                        self.tracks[idx].hist = new_hist
            else:
                # Track als "nicht gefunden" markieren
                self.tracks[idx].non_detects += 1
                if self.tracks[idx].non_detects >= 15:
                    self.tracks[idx].lost = True

    def setup_matrix(self, detections, contours, vis, frame_gray):
        """
        Erstellt die Kostenmatrix für die Zuordnung von Tracks zu Detektionen.

        Args:
            detections (list): Liste der Bounding-Boxen der Detektionen.
            contours (list): Konturen
            vis (numpy.ndarray): Visualisierungsframe.
            frame_gray (numpy.ndarray): Graustufenbild des aktuellen Frames.

        Returns:
            list: Kostenmatrix, die die Ähnlichkeit zwischen Tracks und Detektionen beschreibt.
        """
        cost_matrix = []
        for track in self.tracks:
            costs = []
            for detection in detections:
                # Berechne Ähnlichkeit basierend auf Farb-Histogrammen
                _, d_hist = calculate_color_histogram(vis, detection, contours)
                hist_sim = compare_histograms(d_hist, track.hist)
                # Berechne Ähnlichkeit basierend auf HOG-Deskriptoren
                hog_sim = np.linalg.norm(calculate_hog_descriptor(frame_gray, detection) - track.hog_descriptor)
                # Berechne Bewegungsähnlichkeit
                movement_cost = calculate_movement_similarity(track, detection)
                # Kombiniere die Ähnlichkeiten zu einer Gesamtmetrik
                total_cost = (
                        0.65 * hist_sim +
                        0.25 * min(hog_sim / 100, 1) +
                        0.10 * movement_cost
                )

                costs.append(total_cost)
            cost_matrix.append(costs)

        if len(cost_matrix) == 0 or len(cost_matrix[0]) == 0:
            return []
        return cost_matrix

    def update_tracks(self, prev_gray, frame_gray, fgmask, contours):
        """
        Aktualisiert alle bestehenden Tracks

        Args:
            prev_gray (numpy.ndarray): Graustufenbild des vorherigen Frames.
            frame_gray (numpy.ndarray): Graustufenbild des aktuellen Frames.
            fgmask (numpy.ndarray): Vordergrundmaske.
            contours (list): Konturen, die im aktuellen Frame enthalten sind.
        """

        for track in self.tracks:
            if not track.lost:
                track.update(prev_gray, frame_gray, fgmask, contours, self.lk_params)


class Track:
    """
    Die Track-Klasse repräsentiert ein einzelnes Objekt, das über mehrere Frames hinweg verfolgt wird.
    Sie kapselt die Eigenschaften des Objekts
    """

    def __init__(self, box, features, hist, track_id, hog_deskriptor):
        self.box = box  # (x, y, w, h)
        self.features = features  # Liste der Feature-Punkte
        self.hist = hist  # Farb-Histogramm der Box
        self.id = track_id  # Eindeutige ID
        self.center = [(box[0] + box[2] // 2), (box[1] + box[3] // 2)]  # Mittelpunkt der Box
        self.mean_shift = [np.array([0, 0])]  # Bewegungsgeschichte (Verschiebung)
        self.trace = [[self.center]]  # Historie der Positionen
        self.skipped_frames = 0  # Anzahl der Frames, in denen das Objekt nicht detektiert wurde
        self.lost = False  # Status, ob das Objekt als verloren gilt
        self.hog_descriptor = hog_deskriptor  # HOG-Deskriptor
        self.ref_box = None  # Referenz-Bounding-Box
        self.non_detects = 0  # Anzahl der Nicht-Detektionen


    def draw_track(self, frame):
        """
        Zeichnet die Historie der Positionen des Objekts in das Bild ein.

        Args:
            frame (numpy.ndarray): Das Bild, in das die Historie eingezeichnet werden soll.
        """
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
            cv.line(frame, prev_center, curr_center, (255 // self.id, 0, 255 // self.id), 2)

    @staticmethod
    def draw_meanshift_vector(vis, box, shift):
        """
        Zeichnet den Mean-Shift-Vektor eines Tracks in das Bild ein.

        Args:
            vis (numpy.ndarray): Das Bild, in das der Vektor eingezeichnet werden soll.
            box (tuple): Die Bounding-Box des Tracks (x, y, Breite, Höhe).
            shift (numpy.ndarray): Der Mean-Shift-Vektor.
        """
        if vis is not None:
            x, y, w, h = box
            cv.arrowedLine(vis, (int(x + w // 2), int(y + h // 2)),
                           (int(x + w // 2 + shift[0]), int(y + h // 2 + shift[1])),
                           (0, 255, 0), 2)

    def update(self, prev_gray, frame_gray, fgmask, contours, lk_params):
        """
        Aktualisiert die Eigenschaften des Tracks basierend auf dem optischen Fluss und anderen Merkmalen.

        Args:
            prev_gray (numpy.ndarray): Graustufenbild des vorherigen Frames.
            frame_gray (numpy.ndarray): Graustufenbild des aktuellen Frames.
            fgmask (numpy.ndarray): Vordergrundmaske.
            contours (list): Liste der Konturen im aktuellen Frame.
            lk_params (dict): Parameter für den optischen Fluss (Lucas-Kanade).
        """

        buffer = 20
        alpha = 0.15
        x, y, w, h = self.box

        # Berechne optischen Fluss
        features = np.float32(self.features).reshape(-1, 1, 2)
        p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, features, None, **lk_params)
        p0, st0, err0 = cv.calcOpticalFlowPyrLK(frame_gray, prev_gray, p1, None, **lk_params)

        # Filtere valide Punkte
        valid_points, previous_points = self._filter_valid_points(p1, st, p0, fgmask, (x-40, y-40, w+80, h+80))

        if len(valid_points) >= 10:

            movement = valid_points - previous_points

            mean_shift = np.mean(movement, axis=0)
            historical_mean_shift = np.mean(self.mean_shift[-3:], axis=0) if len(self.mean_shift) > 3 else mean_shift
            smooth_shift = 0.8 * mean_shift + 0.2 * historical_mean_shift

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
            dynamic_width = min(max_x - min_x, self.ref_box[2])

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
        """
        Filtert gültige Punkte basierend auf der Position und der Vordergrundmaske.

        Args:
            p1 (numpy.ndarray): Neue Positionen der Punkte.
            st (numpy.ndarray): Status der Punkte (1 = valide, 0 = nicht valide).
            features (numpy.ndarray): Ursprüngliche Positionen der Punkte.
            fgmask (numpy.ndarray): Vordergrundmaske.
            box (tuple): Bounding-Box, um die Punkte zu überprüfen.

        Returns:
            tuple: Gefilterte gültige Punkte und die entsprechenden vorherigen Punkte.
        """

        valid_points = p1[st == 1].reshape(-1, 2)
        previous_points = features[st == 1].reshape(-1, 2)

        x, y, w, h = box

        points_within_box = [
            (x <= p[0] <= x + w and y <= p[1] <= y + h) for p in valid_points
        ]

        points_in_fg = [
            not np.all(fgmask[int(p[1]):int(p[1]) + 25, int(p[0]):int(p[0]) + 25] == 0) for p in valid_points
        ]

        # Filter: Punkte müssen innerhalb der Box liegen und im Vordergrund sein
        mask_filter = np.array(points_within_box, dtype=bool) & np.array(points_in_fg, dtype=bool)

        return valid_points[mask_filter], previous_points[mask_filter]
