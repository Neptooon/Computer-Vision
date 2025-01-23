import cv2 as cv
import numpy as np

class Detector:
    """
    Klasse zur Objektdetektion von Personen.
    Diese Klasse verwendet den vortrainierten HOG-Personendetektor von OpenCV.
    """

    def __init__(self):
        """
        Initialisiert die Detector-Klasse und konfiguriert den HOG-Deskriptor mit einem vortrainierten Personendetektor.
        """
        self.hog = cv.HOGDescriptor()  # HOG-Deskriptor initialisieren
        self.hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())  # Vortrainierten Personendetektor laden

    def detect(self, frame):
        """
        Führt die Objektdetektion in einem gegebenen Frame durch.

        Args:
            frame (numpy.ndarray): Das Eingabebild, in dem Objekte erkannt werden sollen.

        Returns:
            numpy.ndarray: Ein Array von Bounding-Boxen für die erkannten Objekte.
        """
        # Reduzierung der Bildgröße für schnellere Berechnung
        frame_down_sample = cv.resize(frame, ((frame.shape[1] // 40) * 10, (frame.shape[0] // 40) * 10))

        # HOG-basierte Detektion von Objekten im Bild
        boxes, weights = self.hog.detectMultiScale(
            frame_down_sample,
            winStride=(2, 2),  # Schrittgröße für das Fenster
            padding=(4, 4),  # Padding um das Detektionsfenster
            scale=1.07,  # Skalierungsfaktor zwischen den Bildpyramidenebenen
            useMeanshiftGrouping=True  # Gruppieren von überlappenden Boxen
        )

        # Skalieren der Koordinaten zurück auf die Originalbildgröße
        boxes = np.divide(boxes * 40, 10).astype(int)

        return boxes