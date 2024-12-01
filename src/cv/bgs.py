import cv2 as cv
import numpy as np

# max_point_loss = 10  # Mindestanzahl von Punkten für das Tracking
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=5000, qualityLevel=0.3, minDistance=7, blockSize=7)

cap = cv.VideoCapture('../../assets/videos/Cross-Tafel+2J-3D-Mix-Hell-LD.mov')

# Hintergrundsubtraktion
bgs = cv.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=250)
bgs.setBackgroundRatio(0.7)
bgs.setShadowValue(255)
bgs.setShadowThreshold(0.3)

tracks = []  # Punkte die verfolgt werden sollen
detect_interval = 5  # Alle x = 5 Frames detektieren wir neue Feature Punkte
track_len = 10
prev_gray = None
lost_track_counter = 0  # Noch nicht implementiert soll eig. für die Situation sein wenn der Tracke die Person verliert
max_lost_frames = 10  # Same here ""

# TODO: HINWEIS der Tracker mit Lucas-Kanade von hier: https://github.com/npinto/opencv/blob/master/samples/python2/lk_track.py

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
    cv.imshow('frame', fgmask)
    contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    vis = frame.copy()

    if contours:
        # Größte Kontur finden
        """largest_contour = max(contours, key=cv.contourArea)

        if cv.contourArea(largest_contour) > 500:
            x, y, w, h = cv.boundingRect(largest_contour)  # Box um die Kontur ziehen
            cv.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)"""

        # Konturen filtern
        filtered_contours = [contour for contour in contours if cv.contourArea(contour) >= 20000]
        for contour in filtered_contours:
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if len(tracks) > 0:
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)  # Aktuellen Punkte der Tracks
        p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None,
                                              **lk_params)  # Bewegung der Punkte berechnen für: (Frame davor) und (Frame aktuell)
        p0r, st, err = cv.calcOpticalFlowPyrLK(frame_gray, prev_gray, p1, None,
                                               **lk_params)  # Umgekehrte optischer Fluss von aktuellem Frame zu Frame davor

        # Konsistenzprüfung
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)  # Punkte bei denen die Distanz zu groß ist, werden entfernt
        good = d < 1
        new_tracks = []  # Neue Punkte zum tracken kommen hier rein
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > track_len:  # Entfernt alte Punkte, wenn die Track-Länge überschritten wird
                del tr[0]
            new_tracks.append(tr)
            cv.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

        tracks = new_tracks
        cv.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))

    if int(cap.get(cv.CAP_PROP_POS_FRAMES)) % detect_interval == 0:  # Alle 5 Frames wird detektiert

        # Maske auf vorhandene Punkte anwenden tr[-1] letzter Punkt des aktuellen Tracks
        # Punkte die bereits fürs Tracking existieren werden auf der weißen Maske mit radius 5 Schwarz gefärbt um redundante Punkte zu vermeiden und nicht doppelt zu tracken
        # Neue Feature-Punkte detektieren
        for contour in contours:
            if cv.contourArea(contour) > 500:  # Mindestfläche
                x, y, w, h = cv.boundingRect(contour)
                roi = frame_gray[y:y + h, x:x + w]  # ROI um Feature-Suche einzugrenzen

                # Maske für das aktuelle ROI erstellen
                mask_roi = np.ones_like(roi) * 255  # Weiß (255) für die ganze ROI-Maske

                # Vorhandene Punkte auf der Maske im ROI-Bereich blockieren
                for tx, ty in [np.int32(tr[-1]) for tr in tracks]:  # Punkte aus den Tracks
                    if x <= tx <= x + w and y <= ty <= y + h:  # Nur Punkte im aktuellen ROI
                        cv.circle(mask_roi, (int(tx - x), int(ty - y)), 5, 0, -1)  # Blockieren mit schwarz

                p = cv.goodFeaturesToTrack(roi, mask=mask_roi, **feature_params)  # Feature Punkte

                if p is not None:
                    for px, py in np.float32(p).reshape(-1, 2):
                        tracks.append([(px + x, py + y)])

    prev_gray = frame_gray
    cv.imshow('Tracking', vis)

    # Beenden mit ESC
    if cv.waitKey(80) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()