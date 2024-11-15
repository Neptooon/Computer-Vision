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
    cap = cv.VideoCapture('../../assets/videos/LL-Default-Pulli-Jogging-Hell-RL.mov')

    # BGS-methoden mit Parametrisierung

    MOG2 = cv.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=450)
    MOG2.setBackgroundRatio(0.8)
    MOG2.setShadowValue(255)
    MOG2.setShadowThreshold(0.4)


    CNT = cv.bgsegm.createBackgroundSubtractorCNT()
    KNN = cv.createBackgroundSubtractorKNN(detectShadows=True, dist2Threshold=1000)

    # Kernel für Morph. Operatoren
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        frame = cv.GaussianBlur(frame, (5, 5), 0)  # Rauschunterdrückung

        # BGS anwenden
        fgmask = MOG2.apply(frame)
        fgmask2 = CNT.apply(frame)
        fgmask3 = KNN.apply(frame)

        # Schatten von den Vordergrundmasken über Threshold entfernen
        #fgmask = cv.threshold(fgmask, 254, 255, cv.THRESH_BINARY)[1]
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)  # Rauschen entfernen
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)  # Löcher füllen

        # Median-Filter zur weiteren Rauschreduzierung
        fgmask = cv.medianBlur(fgmask, 3)

        # ... Gleicher Code = Gleiche Doku s.o.
        fgmask2 = cv.threshold(fgmask2, 254, 255, cv.THRESH_BINARY)[1]
        fgmask2 = cv.morphologyEx(fgmask2, cv.MORPH_OPEN, kernel)
        fgmask2 = cv.morphologyEx(fgmask2, cv.MORPH_CLOSE, kernel)
        fgmask2 = cv.medianBlur(fgmask2, 3)

        # ...
        fgmask3 = cv.morphologyEx(fgmask3, cv.MORPH_OPEN, kernel)
        fgmask3 = cv.morphologyEx(fgmask3, cv.MORPH_CLOSE, kernel)
        fgmask3 = cv.medianBlur(fgmask3, 3)

        # Masken in BGR-Bild umwandeln, weil sonst die Darstellung von Farb- und Graubild von den Channeln her nicht passt
        fgmask_bgr = cv.cvtColor(fgmask, cv.COLOR_GRAY2BGR)
        fgmask2_bgr = cv.cvtColor(fgmask2, cv.COLOR_GRAY2BGR)
        fgmask3_bgr = cv.cvtColor(fgmask3, cv.COLOR_GRAY2BGR)

        # Frames beschriften
        original_frame = frame.copy()

        labelFrame(original_frame, 'Original')
        labelFrame(fgmask_bgr, 'MOG2')
        labelFrame(fgmask2_bgr, 'CNT')
        labelFrame(fgmask3_bgr, 'KNN')

        # Frames zusammenfügen (Original, MOG, CNT, KNN)
        top_row = np.hstack((original_frame, fgmask_bgr))
        bottom_row = np.hstack((fgmask3_bgr, fgmask2_bgr))
        combined_frame = np.vstack((top_row, bottom_row))

        # Bild skalieren auf die oben angegebene Größe
        height, width = combined_frame.shape[:2]
        scaling_factor = target_width / width
        resized_frame = cv.resize(combined_frame, (int(width * scaling_factor), int(height * scaling_factor)))

        # Ergebnis der BGS
        cv.imshow('', resized_frame)

        keyboard = cv.waitKey(1)
        if keyboard == ord('q') or keyboard == 27:  # 'q' oder Esc
            break

    cap.release()
    cv.destroyAllWindows()


bgs()
