import cv2 as cv
from src.cv_modules.BGS import BGS
from src.cv_modules.detector import Detector
from src.cv_modules.helpers import draw_boxes, draw_features, filter_contours
from src.cv_modules.tracker import Tracker
from src.metrics.MOTA import MOTA_Metrik


class MultipleObjectTrackingPipeline:
    """
    Diese Klasse implementiert die Pipeline für die Verfolgung mehrerer Objekte.
    Sie kombiniert die Module zur Hintergrundsubtraktion, Objektdetektion und -verfolgung.
    Hinweis!!!: Diese Pipeline wird speziell verwendet für das Spiel
    """
    def __init__(self, cap):
        """
        Initialisiert die Pipeline mit den erforderlichen Modulen und Parametern.

        Args:
            cap (cv.VideoCapture): Ein OpenCV-Video-Capture-Objekt.
        """

        self.cap = cap
        self.bgs = BGS()  # BGS
        self.detector = Detector()  # Objektdetektor
        self.tracker = Tracker()  # Tracker

        self.prev_gray = None  # Speichert das vorherige Graustufen-Bild

        self.frame_counter = 0  # Zähler für die Frames
        self.collision = []  # Liste für Kollisionsdaten

    def run(self, frame):
        """
        Führt die Pipeline für einen einzelnen Frame aus. Dies umfasst den einmaligen Durchlauf des MOT-Algorithmus.

        Args:
            frame (numpy.ndarray): Der aktuelle Frame des Videos.

        Returns:
            numpy.ndarray: Der visualisierte Frame mit gezeichneten Objekten und Merkmalen.
        """
        vis = frame.copy()

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Wende BGS an
        fgmask = self.bgs.bgs_apply(frame)

        # Extrahiere und filtere Konturen aus der Vordergrundmaske
        contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        filtered_contours = filter_contours(contours)

        # Alle drei Frames Detektion und Aktualisierung der Verfolgung durchführen
        if self.frame_counter % 3 == 0:
            boxes = self.detector.detect(frame)  # Detektion von Objekten
            self.tracker.associate_tracks(boxes, frame_gray, vis, filtered_contours)  # Aktualisierung der Tracks

        # Aktualisiere die Tracks basierend auf optischem Fluss und anderen Merkmalen
        if self.prev_gray is not None:
            self.tracker.update_tracks(self.prev_gray, frame_gray, fgmask, filtered_contours)

        self.frame_counter += 1
        self.prev_gray = frame_gray
        return cv.flip(vis, 1)  # Frame horizontal spiegeln, um den Pygame-Flip auszugleichen




class MultipleObjectTrackingPipelineMetrik:
    """
    Erweiterte Pipeline zur Verfolgung mehrerer Objekte mit MOTA-Metrik. Diese Pipeline ist unabhängig vom Spiel
    und kann für die allgemeine Videoverarbeitung und -analyse verwendet werden.
    """
    def __init__(self, video_path):
        """
        Initialisiert die Pipeline mit den erforderlichen Modulen und Parametern.

        Args:
            video_path (str): Pfad zum Eingabevideo.
        """
        self.cap = cv.VideoCapture(video_path)
        self.bgs = BGS()  # BGS
        self.detector = Detector()  # Objektdetektor
        self.tracker = Tracker()  # Tracker
        self.prev_gray = None  # Speichert das vorherige Graustufen-Bild
        self.width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)  # Breite des Videos
        self.height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # Höhe des Videos
        self.mota = MOTA_Metrik(video_path)  # MOTA-Metrik

        self.frame_counter = 0  # Zähler für die Frames
        self.collision = []  # Liste für Kollisionsdaten

    def run_metrik(self):
        """
        Führt die Pipeline aus und berechnet zusätzlich die Metriken.
        """

        # Hinweis: Die auskommentierten Abschnitte sind für die Auswertung der Metrik verwendet worden
        # Die Auswertung der MOTA-Metrik finden sie in der Datei: MOTA_results.json
        #gt_file = "../../assets/videos/gt/gt_MOT-Livedemo2.txt"  # Ground Truth Datei
        #pd_file = "../../src/cv_modules/pd_MOT-InAndOut-Crossing.txt"  # Predicted Datei
        #frames = self.mota.extract_frames(gt_file)  # Extrahiere die relevanten Frames

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            vis = frame.copy()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            fgmask = self.bgs.bgs_apply(frame) # Wende BGS an

            # Extrahiere und filtere Konturen
            contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            filtered_contours = filter_contours(contours)
            # Alle drei Frames Detektion und Aktualisierung der Verfolgung durchführen
            if self.frame_counter % 3 == 0:
                boxes = self.detector.detect(frame)
                self.tracker.associate_tracks(boxes, frame_gray, vis, filtered_contours)

            # Aktualisiere die Tracks basierend auf optischem Fluss und anderen Merkmalen
            if self.prev_gray is not None:
                self.tracker.update_tracks(self.prev_gray, frame_gray, fgmask, filtered_contours)

            # Zeichne Merkmale und Boxen auf dem visualisierten Frame
            draw_features(vis, self.tracker.tracks)
            draw_boxes(vis, self.tracker.tracks)

            self.frame_counter += 1
            self.prev_gray = frame_gray

            # Wurde für die MOTA-Metrik verwendet, um die Informationen der Tracks nur aus den relevanten Frames zu extrahieren
            """if self.frame_counter in frames: 
                self.mota.write_track_info(pd_file,self.tracker.tracks, self.frame_counter)"""

            # Zeichne Konturen auf dem Frame
            cv.drawContours(vis, filtered_contours, -1, (0, 255, 0), 2)

            cv.imshow('HOG', vis) # Zeige das Ergebnis an
            key = cv.waitKey(1)

            if key & 0xFF == 27: # Beenden
                break

            if key == ord('p'): # Pause
                cv.waitKey(-1)

        # Hinweis: Gehört auch zur MOTA-Metrik dort muss dass ground-truth-file mit dem passenden prediction-file als parameter übergeben werden um die MOTA-Metrik auszuwerten
        #self.mota.motMetricsEnhancedCalculator(gt_file, pd_file, frames, '../../src/metrics/MOTA_results.json', self.video)

        # Release Ressourcen nach Abschluss
        self.cap.release()
        cv.destroyAllWindows()


# Hier ausführen, um das gegebene Video im Capture abzuspielen ohne die Spielintegration
if __name__ == "__main__":
    pipeline = MultipleObjectTrackingPipelineMetrik('../../assets/videos/MOT-Crossing-Deluxe.mov')
    pipeline.run_metrik()

