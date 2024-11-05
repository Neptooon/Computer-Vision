import cv2 as cv
import numpy as np

# https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
def bgs():

    def labelFrame(frame, name):
        # Textparameter
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (255, 0, 0)
        thickness = 2

        frame_width = frame.shape[1]
        box_x = frame_width - 60
        box_y = 30

        cv.putText(frame, name, (10, 30), font, font_scale, color, thickness, cv.LINE_AA)
        cv.rectangle(frame, (box_x - 5, box_y - 15), (box_x + 50, box_y + 5), (255, 255, 255), -1)
        cv.putText(frame, str(int(cap.get(cv.CAP_PROP_POS_FRAMES))), (box_x, box_y), font, 0.5, (0, 0, 0))


    # Zielbreite für das kombinierte Bild
    target_width = 1180

    # Video laden
    cap = cv.VideoCapture('../../assets/videos/LL-Tafel+2J+Bin-Pulli-Kapuze-Hell.mov')

    # BG-methoden
    MOG2 = cv.createBackgroundSubtractorMOG2(detectShadows=True)
    CNT = cv.bgsegm.createBackgroundSubtractorCNT()
    KNN = cv.createBackgroundSubtractorKNN()

    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        # BGS anwenden
        fgmask = MOG2.apply(frame)
        fgmask2 = CNT.apply(frame)
        fgmask3 = KNN.apply(frame)

        # Masken in BGR umwandeln, weil sonst die Darstellung von Farb- und Graubild von den Channeln her nicht passt
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

        # Bild skalieren auf Zielbreite
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
