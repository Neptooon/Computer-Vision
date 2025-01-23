import cv2 as cv
import numpy as np


def calculate_color_histogram(frame, box, contours):
    """
    Berechnet das Farb-Histogramm eines bestimmten Bereichs (ROI) im Bild.

    Args:
        frame (numpy.ndarray): Das Eingabebild.
        box (tuple): Die Bounding-Box, definiert durch (x, y, Breite, Höhe).
        contours (list): Liste der Konturen, die in der Maske gezeichnet werden sollen.
    Returns:
        tuple: Anzahl der weißen Pixel und das normalisierte HSV-Histogramm.
    """
    x, y, w, h = box
    x = int(max(0, min(x, frame.shape[1] - 1)))
    y = int(max(0, min(y, frame.shape[0] - 1)))
    w = int(min(w, frame.shape[1] - x))
    h = int(min(h, frame.shape[0] - y))

    roi = frame[y:y + h, x:x + w]

    # Erstellen einer Maske basierend auf den Konturen
    contours = merge_contours(contours)
    mask = cv.UMat(np.zeros(roi.shape[:2], dtype=np.uint8))
    if contours is not None:
        adjusted_contours = []
        for cnt in contours:
            cnt_adjusted = cnt - np.array([x, y])
            # Überprüfen, ob der angepasste Konturpunkt innerhalb der ROI liegt
            cnt_adjusted = np.clip(cnt_adjusted, 0, [w - 1, h - 1]).astype(int)
            adjusted_contours.append(cnt_adjusted)

        # Zeichnen der angepassten Konturen auf die Maske
        if adjusted_contours:
            cv.drawContours(mask, adjusted_contours, -1, 255, -1)

    num_white_pixels = cv.countNonZero(mask) # Anzahl der nicht-schwarzen Pixel in der Maske
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    if num_white_pixels > 36000:
        hist = cv.calcHist([hsv_roi], [0, 1], mask, [30, 32], [0, 180, 0, 256])
    else:
        hist = cv.calcHist([hsv_roi], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv.normalize(hist, hist, 0, 1, cv.NORM_MINMAX) # Normalisierung des Histogramms
    return num_white_pixels, hist


def compare_histograms(hist1, hist2):
    """
    Vergleicht zwei Farb-Histogramme mit der Bhattacharyya-Methode.

    Args:
        hist1 (numpy.ndarray): Das erste Histogramm.
        hist2 (numpy.ndarray): Das zweite Histogramm.

    Returns:
        float: Der Bhattacharyya-Abstand (niedriger ist besser).
    """
    return cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)


def merge_contours(contours, max_gap=100):
    """
    Verbindet mehrere Konturen, die nahe beieinander liegen, zu einer einzigen.

    Args:
        contours (list): Liste der Konturen.
        max_gap (int): Maximale vertikale Entfernung zwischen Konturen.

    Returns:
        list: Liste der gemergten Konturen.
    """
    if not contours:
        return []

    # Sortiere nach y-Koordinate
    contours = sorted(contours, key=lambda cnt: cv.boundingRect(cnt)[1])

    merged_contours = []
    used = [False] * len(contours)  # Speichert, ob eine Kontur bereits gemergt wurde

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

        merged_contours.append(merged)

        used[i] = True

    return merged_contours


def draw_boxes(vis, tracks):
    """
    Zeichnet Bounding-Boxen der Tracks auf ein Bild.

    Args:
        vis (numpy.ndarray): Das Bild, auf dem die Boxen gezeichnet werden.
        tracks (list): Liste der Tracks mit Box-Informationen.
    """
    for track in tracks:
        if not track.lost and track.box is not None:
            x, y, w, h = track.box
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            cv.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2) # Box Zeichnen
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(vis, str(track.id), (x, y), font, 1, (0, 0, 255), 2, cv.LINE_AA) # Track-ID anzeigen


def draw_features(vis, tracks):
    """
    Zeichnet Merkmale und Zentren der Tracks auf ein Bild.

    Args:
        vis (numpy.ndarray): Das Bild, auf dem die Merkmale gezeichnet werden.
        tracks (list): Liste der Tracks mit Merkmalen.
    """
    if tracks is not None:
        for track in tracks:
            if track.center is not None and not track.lost:
                cv.circle(vis, (int(track.center[0]), int(track.center[1])), 2, (0, 0, 255), 2) # Zentrum zeichnen
            if track.features is not None and not track.lost:
                for point in track.features:
                    cv.circle(vis, (int(point[0]), int(point[1])), 2, (132 // track.id, 0, 255 // track.id), 2) # Feature Zeichnen


def compute_iou(box_a, box_b):
    """
    Berechnet den Intersection-over-Union (IoU)-Wert zwischen zwei Boxen.

    Args:
        box_a (tuple): Erste Box (x, y, Breite, Höhe).
        box_b (tuple): Zweite Box (x, y, Breite, Höhe).

    Returns:
        float: Der IoU-Wert (0 bis 1).
    """

    if box_a is None or box_b is None:
        return 0

    if any(val is None for val in box_a) or any(val is None for val in box_b):
        return None

    # Boxen (x, y, w, h) in (x1, y1, x2, y2)
    x1_a, y1_a, w_a, h_a = box_a
    x2_a, y2_a = x1_a + w_a, y1_a + h_a

    x1_b, y1_b, w_b, h_b = box_b
    x2_b, y2_b = x1_b + w_b, y1_b + h_b

    # Schnittmenge
    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    # Vereinigungsmenge
    area_a = w_a * h_a
    area_b = w_b * h_b
    union = area_a + area_b - intersection

    # IoU berechnen
    return intersection / union if union > 0 else 0.0


def calculate_hog_descriptor(frame, box):
    """
    Berechnet den HOG-Deskriptor für einen Bereich.

    Args:
        frame (numpy.ndarray): Das Eingabebild.
        box (tuple): Die Bounding-Box (x, y, Breite, Höhe).

    Returns:
        numpy.ndarray: Der berechnete HOG-Deskriptor.
    """
    x, y, w, h = box
    x = int(max(0, min(x, frame.shape[1] - 1)))
    y = int(max(0, min(y, frame.shape[0] - 1)))
    w = int(min(w, frame.shape[1] - x))
    h = int(min(h, frame.shape[0] - y))
    roi = frame[y:y+h, x:x+w]

    roi = cv.resize(roi, (64, 128))  # Standardgröße für HOG
    hog = cv.HOGDescriptor()
    return hog.compute(roi)


def calculate_movement_similarity(track, detection):
    """
    Berechnet die Bewegungsähnlichkeit zwischen einem Track und einer neuen Detektion.

    Args:
        track (Track): Der bestehende Track.
        detection (tuple): Die neue Detektion, definiert durch (x, y, Breite, Höhe).

    Returns:
        float: Die berechnete Bewegungsähnlichkeit (0 bis 1) normalisiert.
    """
    track_center = np.array([track.center[0], track.center[1]])
    detection_center = np.array([detection[0] + detection[2] // 2, detection[1] + detection[3] // 2])

    distance = np.sqrt(
        (track_center[0] - detection_center[0]) ** 2 +
        (track_center[1] - detection_center[1]) ** 2
    )

    return min(distance / 100, 1)


def filter_contours(contours):
    """
    Filtert Konturen basierend auf ihrer Fläche und ihrem Seitenverhältnis.

    Args:
        contours (list): Liste der Konturen.

    Returns:
        list: Gefilterte Konturen.
    """
    filtered = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area < 1000:  # Ignoriere kleine Konturen
            continue

        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = h / w

        if 1.2 < aspect_ratio < 5.0:  # Filtere basierend auf dem Seitenverhältnis
            filtered.append(contour)
    return filtered
