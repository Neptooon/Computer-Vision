import cv2 as cv
from src.metrics.IoU import IoUMetrik
from src.cv_modules.BGS import BGS
from src.cv_modules.detector import Detector
from src.cv_modules.helpers import merge_contours, draw_boxes, draw_features
from src.cv_modules.tracker import Tracker


class SingleObjectTrackingPipeline:
    def __init__(self, video_path):
        self.cap = cv.VideoCapture(video_path)
        self.bgs = BGS()
        self.detector = Detector()
        self.tracker = Tracker()
        self.iou_metrik = IoUMetrik(video_path)
        self.prev_gray = None
        self.width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)


        # Todo -- Nur für Metrik zwecke angelegt --
        self.frame_counter = 0
        self.detect_counter = 0
        self.tracking_counter = 0
        self.empty = 0


    def run(self):
        lock = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            vis = frame.copy()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            fgmask = self.bgs.bgs_apply(frame)

            contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            filtered_contours = [contour for contour in contours if cv.contourArea(contour) >= 1000]

            if self.frame_counter % 5 == 0 and lock < 1:
                boxes = self.detector.detect(frame, fgmask)
                if len(self.tracker.tracks) > 0:
                    lock += 1
                self.tracker.update2(boxes, frame_gray, vis, filtered_contours, fgmask)
            self.tracker.update_tracks(self.prev_gray, frame_gray, fgmask, filtered_contours, vis)

            draw_features(vis, self.tracker.tracks)
            draw_boxes(vis, self.tracker.tracks)
            #if len(self.tracker.tracks) > 0:
                #vis = cv.drawKeypoints(vis, self.tracker.tracks[0].sift_keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            #self.iou_metrik.get_iou_info(self.tracker.tracks, self.frame_counter)
            self.frame_counter += 1
            self.prev_gray = frame_gray
            #filtered_contours = merge_contours(filtered_contours)
            cv.drawContours(vis, filtered_contours, -1, (0, 255, 0), 2)

            cv.imshow('HOG', vis)
            #cv.imshow('BG', fgmask)
            key = cv.waitKey(1)

            if key & 0xFF == 27:
                break

            if key == ord('p'):
                cv.waitKey(-1)

        self.cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    pipeline = SingleObjectTrackingPipeline('../../assets/videos/ML3-DS-Dunkel-Tafel-LiveDemo.mov')
    pipeline.run()

