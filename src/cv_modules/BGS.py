import cv2 as cv


class BGS:
    """
    Klasse zur Durchführung der Hintergrundsubtraktion (Background Subtraction).
    Die Klasse verwendet einen MOG2-Hintergrundsubtraktor von OpenCV, um bewegte Objekte in einem Video zu erkennen.
    Zusätzlich werden morphologische Operationen angewandt, um die Maske zu bereinigen.
    """

    def __init__(self):
        """
        Konstruktor der BGS-Klasse. Initialisiert den Hintergrundsubtraktor und die dazugehörigen Parameter.
        """
        # Initialisierung des Hintergrundsubtraktors mit Schattenerkennung
        self.backgroundSubtraction = cv.createBackgroundSubtractorMOG2(detectShadows=True,
                                                                       varThreshold=75)  # Schwellwert = 75
        self.backgroundSubtraction.setBackgroundRatio(0.7)  # Verhältnis der Hintergrundmodellierung
        self.backgroundSubtraction.setShadowValue(255)  # Wert für erkannte Schatten
        self.backgroundSubtraction.setShadowThreshold(0.2)  # Schwelle für die Schattenerkennbarkeit

        # Strukturierungselement zur Anwendung von Morphologieoperationen
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    def bgs_apply(self, frame):
        """
        Wendet die Hintergrundsubtraktion auf einen einzelnen Frame an.

        Die Methode erzeugt eine Vordergrundmaske (fgmask) und bereinigt diese mittels
        morphologischer Öffnungs- und Schließungsoperationen.

        Args:
            frame (numpy.ndarray): Der Eingabeframe, auf den die Hintergrundsubtraktion angewendet wird.

        Returns:
            numpy.ndarray: Die bereinigte Vordergrundmaske.
        """
        # Wendet den Hintergrundsubtraktor auf den Frame an
        fgmask = self.backgroundSubtraction.apply(frame)

        # Führt eine Öffnung durch, um kleine weiße Flecken in der Maske zu entfernen
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, self.kernel)

        # Führt eine Schließung durch, um kleine schwarze Löcher in der Maske zu schließen
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, self.kernel)

        return fgmask  # Gibt die bereinigte Vordergrundmaske zurück