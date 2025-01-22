import cv2 as cv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_color_histogram(frame, box, contours, fgmask):
    x, y, w, h = box
    x = int(max(0, min(x, frame.shape[1] - 1)))
    y = int(max(0, min(y, frame.shape[0] - 1)))
    w = int(min(w, frame.shape[1] - x))
    h = int(min(h, frame.shape[0] - y))

    roi = frame[y:y + h, x:x + w]

    # Erstellen einer Maske für die Konturen
    #mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    contours = merge_contours(contours)
    mask = cv.UMat(np.zeros(roi.shape[:2], dtype=np.uint8))
    # Anpassen der Konturen relativ zur ROI
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

    num_white_pixels = cv.countNonZero(mask)
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    if num_white_pixels > 36000:
        hist = cv.calcHist([hsv_roi], [0, 1], mask, [30, 32], [0, 180, 0, 256])
    else:
        hist = cv.calcHist([hsv_roi], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv.normalize(hist, hist, 0, 1, cv.NORM_MINMAX)
    return num_white_pixels, hist


def compare_histograms(hist1, hist2):
    return cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)  # Kleiner umso besser


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
        merged_contours.append(merged)

        used[i] = True

    return merged_contours


def draw_boxes(vis, tracks):  # Boxen zeichnen
    for track in tracks:
        if not track.lost and track.box is not None:
            x, y, w, h = track.box
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            cv.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(vis, str(track.id), (x, y), font, 1, (0, 0, 255), 2, cv.LINE_AA)


def draw_features(vis, tracks):  # Feature zeichnen
    if tracks is not None:
        for track in tracks:
            if track.center is not None and not track.lost:
                cv.circle(vis, (int(track.center[0]), int(track.center[1])), 2, (0, 0, 255), 2)
            if track.features is not None and not track.lost:
                for point in track.features:
                    cv.circle(vis, (int(point[0]), int(point[1])), 2, (132 // track.id, 0, 255 // track.id), 2)


def compute_iou(box_a, box_b):
    # Boxen (x, y, w, h) in (x1, y1, x2, y2)
    if box_a is None or box_b is None:
        return 0

    if any(val is None for val in box_a) or any(val is None for val in box_b):
        return None
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


def calculate_hog_descriptor(frame, box, mask):
    """Berechnet den HOG-Deskriptor für den aktuellen Track."""
    x, y, w, h = box
    x = int(max(0, min(x, frame.shape[1] - 1)))
    y = int(max(0, min(y, frame.shape[0] - 1)))
    w = int(min(w, frame.shape[1] - x))
    h = int(min(h, frame.shape[0] - y))
    roi = frame[y:y+h, x:x+w]
    #roi_mask = mask[y:y+h, x:x+w]

    #roi = cv.bitwise_and(roi, roi_mask)
    roi = cv.resize(roi, (64, 128))  # Standardgröße für HOG
    #cv.imshow("", roi)
    hog = cv.HOGDescriptor()
    return hog.compute(roi)

def calculate_movement_similarity(track, detection):

    track_center = np.array([track.center[0], track.center[1]])
    detection_center = np.array([detection[0] + detection[2] // 2, detection[1] + detection[3] // 2])

    # Euk. Diz

    distance = np.sqrt(
        (track_center[0] - detection_center[0]) ** 2 +
        (track_center[1] - detection_center[1]) ** 2
    )

    return min(distance / 100, 1)

def match_feature_cost(frame, detection, track, feature, matcher, mask, vis=None):
    x, y, w, h = detection
    x = int(max(0, min(x, frame.shape[1] - 1)))
    y = int(max(0, min(y, frame.shape[0] - 1)))
    w = int(min(w, frame.shape[1] - x))
    h = int(min(h, frame.shape[0] - y))

    roi = frame[y:y + h, x:x + w]

    keypoints, descriptors = feature.detectAndCompute(roi, mask[y:y + h, x:x + w])

    matches = matcher.match(track.orb_descriptors, descriptors)
    if len(matches) == 0:
        return 1, 0
    threshold = 0.75 * max(m.distance for m in matches)
    matches = [m for m in matches if m.distance < threshold]
    if vis is not None:
        temp_points = []
        for kp in keypoints:
            global_x = kp.pt[0] + x
            global_y = kp.pt[1] + y
            global_kp = cv.KeyPoint(global_x, global_y, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            temp_points.append(global_kp)
        keypoints = temp_points
        for match in matches:
            if track.id == 1:
                cv.circle(vis, (int(keypoints[match.trainIdx].pt[0]), int(keypoints[match.trainIdx].pt[1])), 2, (100 , 200, 100), 2)
            if track.id == 2:
                cv.circle(vis, (int(keypoints[match.trainIdx].pt[0]), int(keypoints[match.trainIdx].pt[1])), 2,
                          (0, 0, 255), 2)

    return 1-(len(matches)/(len(track.orb_descriptors) // 2)), len(matches)

def merge_boxes(collisions):
    valid_tracks = [track for track in collisions if track.box is not None]
    if not valid_tracks:
        return None
    x_min = min([track.box[0] for track in valid_tracks])
    y_min = min([track.box[1] for track in valid_tracks])
    x_max = max([track.box[0] + track.box[2] for track in valid_tracks])
    y_max = max([track.box[1] + track.box[3] for track in valid_tracks])

    return (x_min, y_min, x_max - x_min, y_max - y_min)
