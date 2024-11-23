import cv2 as cv
import numpy as np


# https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
def bgs():

    # Setzt Namenslabel für einen Frame sowie den aktuellen Frame
    def labelFrame(frame, name):
        # Textparameter
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (255, 0, 0)
        size = 2

        frame_width = frame.shape[1]
        x = frame_width - 60
        y = 30

        cv.putText(frame, name, (10, 30), font, font_scale, color, size, cv.LINE_AA)
        cv.rectangle(frame, (x - 5, y - 15), (x + 50, y + 5), (255, 255, 255), -1)
        cv.putText(frame, str(int(cap.get(cv.CAP_PROP_POS_FRAMES))), (x, y), font, 0.5, (0, 0, 0)) #frames oben rechts


    # Zielbreite für das Entbild
    target_width = 1180

    # Video laden
    cap = cv.VideoCapture('../../assets/videos/LL-Default-Swap-Hell.mov')

    # BGS-methoden mit Parametrisierung

    MOG2 = cv.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=450)
    MOG2.setBackgroundRatio(0.8)
    MOG2.setShadowValue(255)
    MOG2.setShadowThreshold(0.4)

    # Kernel für Morph. Operatoren
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    tracker_initialized = False
    prev_frame = None
    prev_points=None
    bounding_box = None
    good_new_points = None
    good_old_points = None
    roi = None
    x = None
    y = None
    w = None
    h = None

    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        frame = cv.GaussianBlur(frame, (5, 5), 0)  # Rauschunterdrückung

        # BGS anwenden
        fgmask = MOG2.apply(frame)

        # Schatten von den Vordergrundmasken über Threshold entfernen
        #fgmask = cv.threshold(fgmask, 254, 255, cv.THRESH_BINARY)[1]
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)  # Rauschen entfernen
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)  # Löcher füllen

        # Median-Filter zur weiteren Rauschreduzierung
        fgmask = cv.medianBlur(fgmask, 3)

        # Maske in BGR-Bild umwandeln, weil sonst die Darstellung von Farb- und Graubild von den Channeln her nicht passt
        fgmask_bgr = cv.cvtColor(fgmask, cv.COLOR_GRAY2BGR)

        # Frames beschriften
        original_frame = frame.copy()

        if not tracker_initialized:
            konturen, hierarchie = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if len(konturen) != 0:
                largest_kontur = max(konturen, key=cv.contourArea)
                if cv.contourArea(largest_kontur) > 500:
                    cv.drawContours(fgmask_bgr, largest_kontur, -1, (0, 0, 255), 3)
                    cv.drawContours(original_frame, largest_kontur, -1, (0, 0, 255), 3)
                    x, y, w, h = cv.boundingRect(largest_kontur)
                    bounding_box = [x, y, w, h]
                    roi = original_frame[y:y+h, x:x+w]

                    # Harris Corner Detection im Bereich der Kontur anwenden
                    gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                    #gray_roi = np.float32(gray_roi)
                    #prev_points = cv.cornerHarris(gray_roi, 3, 3, 0.04)

                    prev_points = cv.goodFeaturesToTrack(gray_roi, useHarrisDetector=True,maxCorners=100, qualityLevel=0.3, minDistance=7)

                    # Ecken markieren
                    for point in prev_points:
                        dx, dy = point.ravel()
                        cv.circle(roi, (int(dx), int(dy)), 3, (0, 255, 0), -1)

                    # Ecken markieren
                    #roi[prev_points > 0.01 * prev_points.max()] = [0, 255, 0]

                    # Das markierte ROI zurück ins Originalbild einfügen
                    prev_frame = cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY)
                    original_frame[y:y+h, x:x+w] = roi
                    if cap.get(cv.CAP_PROP_POS_FRAMES) % 10 == 0:
                        tracker_initialized = True
        else:
            # Optischer Fluss
            current_frame = cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY)
            #current_frame = np.float32(current_frame)
            new_points, status, error = cv.calcOpticalFlowPyrLK(prev_frame, current_frame, prev_points, None)

            if new_points is not None:
                good_new_points = new_points[status == 1]
                good_old_points = prev_points[status == 1]

            # Zeichne die Bewegungen im Frame
            for i, (new, old) in enumerate(zip(good_new_points, good_old_points)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(original_frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                original_frame = cv.circle(original_frame, (int(a), int(b)), 5, (0, 0, 255), -1)

            # Aktualisiere Variablen
            prev_points = good_new_points.reshape(-1, 1, 2)
            prev_gray = current_frame
            if cap.get(cv.CAP_PROP_POS_FRAMES) % 10 == 0:
                tracker_initialized = False



        labelFrame(original_frame, 'Original')
        labelFrame(fgmask_bgr, 'MOG2')


        # Frames zusammenfügen (Original, MOG, CNT, KNN)
        combined_frame = np.hstack((original_frame, fgmask_bgr))

        # Bild skalieren auf die oben angegebene Größe
        height, width = combined_frame.shape[:2]
        scaling_factor = target_width / width
        resized_frame = cv.resize(combined_frame, (int(width * scaling_factor), int(height * scaling_factor)))

        # Ergebnis der BGS
        cv.imshow('', resized_frame)

        keyboard = cv.waitKey(145)
        if keyboard == ord('q') or keyboard == 27:  # 'q' oder Esc
            break

    cap.release()
    cv.destroyAllWindows()


bgs()
