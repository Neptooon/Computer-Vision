import cv2 as cv
import numpy as np

lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 15, 0.03))
feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=5, blockSize=7)

#quality level: Beeinflusst wie viele Ecken erkannt werden

cap = cv.VideoCapture('../../assets/videos/Cross-Tafel+2J-3D-Mix-Hell-LD.mov')


# Hintergrundsubtraktion
bgs = cv.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=250)
bgs.setBackgroundRatio(0.7)
bgs.setShadowValue(255)
bgs.setShadowThreshold(0.3)

tracks = []
detect_interval = 10
min_tracks = 10
active_boxes = []
tracking_failed = True

track_len = 5
prev_gray = None
filtered_boxes = []
hog = cv.HOGDescriptor()
box_tracks = []
updated_box_tracks = []
last_box_tracks = []
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())


def inside(r, q): # Prüft auf vollständige Box in einer anderen Box
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


while True:
    ret, frame = cap.read()
    if not ret:
        break

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    fgmask = bgs.apply(frame)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)
    fgmask = cv.medianBlur(fgmask, 3)

    vis = frame.copy()
    cv.imshow('frame', fgmask)

    contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:

        # Konturen filtern
        filtered_contours = [contour for contour in contours if cv.contourArea(contour) >= 10000]
        for contour in filtered_contours:
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.drawContours(vis, filtered_contours, -1, (0, 255, 0), 3)

    # HOG-Detektion
    if int(cap.get(cv.CAP_PROP_POS_FRAMES)) % detect_interval == 0:
        box_tracks.clear()
        frame_down_sample = cv.resize(frame, ((frame.shape[1] // 25) * 10, (frame.shape[0] // 25) * 10))
        boxes, weights = hog.detectMultiScale(frame_down_sample, winStride=(8,8), padding=(8,8), scale=1.06)

        boxes = np.divide(boxes * 25, 10).astype(int) # Koordinaten der Boxen wieder upsamplen
        #print("BOX", boxes, "-----------------------", "WEIGHT", weights)

        filtered_boxes.clear()

        for (box, weight) in zip(boxes, weights):
            #print(box, weight)
            if weight > 0.5:
                filtered_boxes.append(box)

        for ri, r in enumerate(boxes):
            for qi, q in enumerate(boxes):
                if ri != qi and inside(r, q):

                    #print("Wahr", ri, r, "---------------------------", qi, q)
                    break
            else:
                filtered_boxes.append(r)

        for (x, y, w, h) in filtered_boxes:  # Boxen drin die Überlappen
            if np.count_nonzero(fgmask[y:y+h, x:x+w]) > 10000 and w * h > 100000:  # Fläche der Bounding Box in Pixel für die Detektion

                cv.rectangle(vis, (x + int(0.15*w), y + int(0.05*h)), (x + w-int(0.15*w), y + h-int(0.05*h)), (255, 0, 0), 1)
                roi = frame_gray[y:y + h, x:x + w]  # ROI um Feature-Suche einzugrenzen

                # Maske für das aktuelle ROI erstellen
                mask_roi = np.ones_like(roi) * 255

                # Vorhandene Punkte auf der Maske im ROI-Bereich blockieren
                for tx, ty in [np.int32(tr[-1]) for tr in tracks]:
                    if x <= tx <= x + w and y <= ty <= y + h:
                        cv.circle(mask_roi, (int(tx - x), int(ty - y)), 3, 0, -1)

                p = cv.goodFeaturesToTrack(roi, mask=mask_roi, **feature_params)

                if p is not None:
                    box_tracks.append(
                        {
                            "box": (x, y, w, h),
                            "features": [(px + x, py + y) for px, py in np.float32(p).reshape(-1, 2)],
                            "mean_shift": [0, 0]
                        }
                    )

    if len(box_tracks) > 0:
        for track in box_tracks:
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)

            box = track["box"]
            features = np.float32(track["features"]).reshape(-1, 1, 2)

            p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, features, None, **lk_params)

            # Filtere valide Punkte
            valid_points = p1[st == 1].reshape(-1, 2)
            previous_points = features[st == 1].reshape(-1, 2)

            mask_filter = [
                not np.all(fgmask[int(p[1]):int(p[1]) + 5, int(p[0]):int(p[0]) + 5] == 0)
                for p in valid_points
            ]

            valid_points = valid_points[mask_filter]
            previous_points = previous_points[mask_filter]

            if len(valid_points) > 0:
                # Berechne die durchschnittliche Verschiebung der Punkte
                movement = valid_points - previous_points
                mean_shift = np.mean(movement, axis=0)

                # Verschiebe die Box basierend auf der durchschnittlichen Verschiebung
                x, y, w, h = box
                new_x = int(x + mean_shift[0])
                new_y = int(y + mean_shift[1])

                if np.count_nonzero(fgmask[new_y:new_y + h, new_x:new_x + w]) > 10000 and w * h > 100000:
                    updated_box_tracks.append({
                        "box": (new_x, new_y, w, h),
                        "features": valid_points.tolist(),
                        "mean_shift": mean_shift
                    })

            for p in valid_points:
                cv.circle(vis, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)

            #cv.polylines(vis, [np.int32(tr) for tr in features], False, (0, 255, 0))

    if len(updated_box_tracks) == 0:
        if len(last_box_tracks) > 0:
            for track in last_box_tracks:
                x, y, w, h = track["box"]
                x = int(x + track["mean_shift"][0])
                track["box"] = (x, y, w, h)
                cv.rectangle(vis, (x + int(0.15 * w), y + int(0.05 * h)),
                             (x + w - int(0.15 * w), y + h - int(0.05 * h)), (255, 0, 0), 1)
    else:
        box_tracks.clear()
        last_box_tracks.clear()
        for track in updated_box_tracks:
            box_tracks.append(track)
            last_box_tracks.append(track)
        updated_box_tracks.clear()

        for track in box_tracks:
            x, y, w, h = track["box"]
            if np.count_nonzero(fgmask[y:y + h, x:x + w]) > 10000 and w * h > 100000:
                cv.rectangle(vis, (x + int(0.15 * w), y + int(0.05 * h)), (x + w - int(0.15 * w), y + h - int(0.05 * h)), (255, 0, 0), 1)

    prev_gray = frame_gray

    cv.imshow('HOG', vis)

    key = cv.waitKey(22)
    if key & 0xFF == 27:
        break

    if key == ord('p'):
        cv.waitKey(-1)

cap.release()
cv.destroyAllWindows()