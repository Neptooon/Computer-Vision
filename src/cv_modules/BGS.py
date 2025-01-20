import cv2 as cv

class BGS: # BGS
    def __init__(self):
        self.backgroundSubtraction = cv.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=75)  # 70
        self.backgroundSubtraction.setBackgroundRatio(0.7)
        self.backgroundSubtraction.setShadowValue(255)
        self.backgroundSubtraction.setShadowThreshold(0.2)  # 0.3
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    def bgs_apply(self, frame):
        fgmask = self.backgroundSubtraction.apply(frame)

        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, self.kernel)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, self.kernel)
        return fgmask
