
import cv2 as cv
import numpy as np

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)

cap = cv.VideoCapture('../../assets/videos/Cross-Tafel+2J-3D-Mix-Hell-LD.mov')


# Hintergrundsubtraktion
bgs = cv.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=250)
bgs.setBackgroundRatio(0.7)
bgs.setShadowValue(255)
bgs.setShadowThreshold(0.3)

tracks = []
detect_interval = 5
track_len = 5
prev_gray = None

hog = cv.HOGDescriptor()

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

    # HOG-Detektion
    #if int(cap.get(cv.CAP_PROP_POS_FRAMES)) % detect_interval == 0:

    frame_down_sample = cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    boxes, weights = hog.detectMultiScale(frame_down_sample, winStride=(8,8), padding=(4,4), scale=1.05)
    print(boxes)
    boxes *= 2  # Koordinaten der Boxen wieder upsamplen
    n_box = []

    for ri, r in enumerate(boxes):
        for qi, q in enumerate(boxes):
            if ri != qi and inside(r, q):

                print("Wahr", ri, r, "---------------------------", qi, q)
                break
        else:
            n_box.append(r)

    for (x, y, w, h) in boxes:  # Boxen drin die Überlappen
        #if w * h > 10000:  # Fläche der Bounding Box in Pixel für die Detektion
            cv.rectangle(vis, (x + int(0.15*w), y + int(0.05*h)), (x + w-int(0.15*w), y + h-int(0.05*h)), (255, 0, 0), 1)
            roi = frame_gray[y:y + h, x:x + w]  # ROI um Feature-Suche einzugrenzen

            # Maske für das aktuelle ROI erstellen
            mask_roi = np.ones_like(roi) * 255

            # Vorhandene Punkte auf der Maske im ROI-Bereich blockieren
            for tx, ty in [np.int32(tr[-1]) for tr in tracks]:
                if x <= tx <= x + w and y <= ty <= y + h:
                    cv.circle(mask_roi, (int(tx - x), int(ty - y)), 5, 0, -1)

            p = cv.goodFeaturesToTrack(roi, mask=mask_roi, **feature_params)

            if p is not None:
                for px, py in np.float32(p).reshape(-1, 2):
                    tracks.append([(px + x, py + y)])

    if len(tracks) > 0:
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
        p0r, st, err = cv.calcOpticalFlowPyrLK(frame_gray, prev_gray, p1, None, **lk_params)

        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > track_len:
                del tr[0]
            new_tracks.append(tr)
            cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

        tracks = new_tracks
        cv.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))

    prev_gray = frame_gray

    cv.imshow('HOG', vis)

    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()