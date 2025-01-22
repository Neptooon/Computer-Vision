import cv2 as cv
from src.metrics.IoU import IoUMetrik
from src.cv_modules.BGS import BGS
from src.cv_modules.detector import Detector
from src.cv_modules.helpers import merge_contours, draw_boxes, draw_features, compute_iou
from src.cv_modules.tracker import Tracker
from src.metrics.MOT import MOTA_Metrik


class MultipleObjectTrackingPipeline:
    def __init__(self, cap):
        self.cap = cap
        self.bgs = BGS()
        self.detector = Detector()
        self.tracker = Tracker()

        self.prev_gray = None

        self.frame_counter = 0
        self.collision = []

    def filter_contours(self,contours):  # Auslagern in Helpers
        filtered = []
        for contour in contours:
            area = cv.contourArea(contour)
            if area < 1000:
                continue

            x, y, w, h = cv.boundingRect(contour)
            aspect_ratio = h / w

            if 1.2 < aspect_ratio < 5.0: # 1.2
                filtered.append(contour)
        return filtered

    def run(self, frame):

        vis = frame.copy()

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        fgmask = self.bgs.bgs_apply(frame)

        contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        filtered_contours = self.filter_contours(contours)

        if self.frame_counter % 3 == 0:
            boxes = self.detector.detect(frame, fgmask)
            self.tracker.update2(boxes, frame_gray, vis, filtered_contours, fgmask)

        if self.prev_gray is not None:
            self.tracker.update_tracks(self.prev_gray, frame_gray, fgmask, filtered_contours, vis, self.collision,
                                       self.frame_counter)

        self.frame_counter += 1
        self.prev_gray = frame_gray
        return cv.flip(vis, 1)




class MultipleObjectTrackingPipelineMetrik:
    def __init__(self, video_path):
        self.cap = cv.VideoCapture(video_path)
        self.bgs = BGS()
        self.detector = Detector()
        self.tracker = Tracker()
        self.iou_metrik = IoUMetrik(video_path)
        self.prev_gray = None
        self.width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.mota = MOTA_Metrik(video_path)

        self.frame_counter = 0
        self.collision = []

    def filter_contours(self,contours):
        filtered = []
        for contour in contours:
            area = cv.contourArea(contour)
            if area < 1000:
                continue

            x, y, w, h = cv.boundingRect(contour)
            aspect_ratio = h / w

            if 1.2 < aspect_ratio < 5.0:

                filtered.append(contour)
        return filtered

    def run_metrik(self):
        gt_file = "../../assets/videos/gt/gt_MOT-Livedemo2.txt"
        pd_file = "../../src/cv_modules/pd_MOT-InAndOut-Crossing.txt"
        frames = self.mota.extract_frames(gt_file)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            vis = frame.copy()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            fgmask = self.bgs.bgs_apply(frame)

            contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            filtered_contours = self.filter_contours(contours)

            if self.frame_counter % 3 == 0:
                boxes = self.detector.detect(frame, fgmask)
                self.tracker.update2(boxes, frame_gray, vis, filtered_contours, fgmask)

            if self.prev_gray is not None:
                self.tracker.update_tracks(self.prev_gray, frame_gray, fgmask, filtered_contours, vis, self.collision, self.frame_counter)
                print(f"Frame Count: {self.frame_counter}")

            draw_features(vis, self.tracker.tracks)
            draw_boxes(vis, self.tracker.tracks)

            self.frame_counter += 1
            self.prev_gray = frame_gray

            """if self.frame_counter in frames:
                self.mota.write_track_info("pd_MOT-Livedemo2.txt",self.tracker.tracks, self.frame_counter)"""


            cv.drawContours(vis, filtered_contours, -1, (0, 255, 0), 2)

            cv.imshow('HOG', vis)
            #cv.imshow('BG', fgmask)
            key = cv.waitKey(1)

            if key & 0xFF == 27:
                break

            if key == ord('p'):
                cv.waitKey(-1)

        #self.mot.motMetricsEnhancedCalculator(gt_file, pd_file, frames)
        self.cap.release()
        cv.destroyAllWindows()



if __name__ == "__main__":
    pipeline = MultipleObjectTrackingPipelineMetrik('../../assets/videos/MOT-Livedemo1.mov')
    pipeline.run_metrik()

