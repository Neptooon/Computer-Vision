import cv2 as cv
import numpy as np


def calculate_color_histogram(frame, box, contours):
    x, y, w, h = box
    roi = frame[y:y + h, x:x + w]

    # Erstellen einer Maske für die Konturen
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    #mask = cv.UMat(np.zeros(roi.shape[:2], dtype=np.uint8))
    # Anpassen der Konturen relativ zur ROI
    adjusted_contours = []
    for cnt in contours:
        cnt_adjusted = cnt - np.array([x, y])
        # Überprüfen, ob der angepasste Konturpunkt innerhalb der ROI liegt
        #if np.all((cnt_adjusted >= 0) & (cnt_adjusted < [w, h])):
        cnt_adjusted = np.clip(cnt_adjusted, 0, [w - 1, h - 1]).astype(int)
        adjusted_contours.append(cnt_adjusted)

    # Zeichnen der angepassten Konturen auf die Maske
    if adjusted_contours:
        cv.drawContours(mask, adjusted_contours, -1, 255, -1)
    cv.imshow("frame", mask)
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv_roi], [0, 1], mask, [50, 60], [0, 180, 0, 256])
    cv.normalize(hist, hist, 0, 1, cv.NORM_MINMAX)
    return hist


def compare_histograms(hist1, hist2):
    return cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)


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
        #hull = cv.convexHull(merged)
        #merged_contours.append(hull)

        '''epsilon = 0.001 * cv.arcLength(merged, True)
        approx = cv.approxPolyDP(merged, epsilon, True)
        merged_contours.append(approx)'''

        merged_contours.append(merged)

        used[i] = True

    return merged_contours
