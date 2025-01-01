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

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            vis = frame.copy()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            fgmask = self.bgs.bgs_apply(frame)
            points = []
            contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            filtered_contours = [contour for contour in contours if cv.contourArea(contour) >= 1000]


            #if len(self.tracker.tracks) < 1:
            if self.frame_counter % 5 == 0:
                boxes = self.detector.detect(frame, fgmask)
                self.tracker.init_tracks(boxes, frame_gray, vis, self.detector, filtered_contours, fgmask)

                # -------------------------------------- TODO Nur für Metrik
                '''if len(self.tracker.tracks) >= 1:
                    self.detect_counter += 1
                else:
                    self.empty += 1'''
                # --------------------------------------
                #self.tracker.check_virt(boxes, points, vis, self.height, self.width)

            #else:
            self.tracker.virt_frame_counter = 0
            self.tracking_counter += 1
            self.tracker.update_tracks(self.prev_gray, frame_gray, fgmask, filtered_contours,
                                                self.frame_counter, self.detector, vis)
            draw_features(vis, self.tracker.tracks)
            draw_boxes(vis, self.tracker.tracks)
            #self.iou_metrik.get_iou_info(self.tracker.tracks, self.frame_counter)
            self.frame_counter += 1
            self.prev_gray = frame_gray
            filtered_contours = merge_contours(filtered_contours)
            cv.drawContours(vis, filtered_contours, -1, (0, 255, 0), 2)

            cv.imshow('HOG', vis)
            #cv.imshow('BG', fgmask)
            key = cv.waitKey(10)

            if key & 0xFF == 27:
                break

            if key == ord('p'):
                cv.waitKey(-1)

        self.cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    pipeline = SingleObjectTrackingPipeline('../../assets/videos/ML3-DS-Dunkel-Tafel-LiveDemo.mov')
    pipeline.run()

'''
In die Pipeline Einbinden, falls bestimmte Funktionen nur auf die BoundingBox beschränkt werden sollen.

# Erweiterungsparameter für die ROI
padding = 20  # Erweiterung um 20 Pixel in alle Richtungen
# Aktuelle Bounding-Box
x, y, w, h = self.tracker.box_tracks[0]["box"]
# Erweiterte ROI berechnen
roi_x_start = max(0, x - padding)
roi_y_start = max(0, y - padding)
roi_x_end = min(fgmask.shape[1], x + w + padding)
roi_y_end = min(fgmask.shape[0], y + h + padding)
# Erweiterte ROI auf Vordergrundmaske anwenden
roi = fgmask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
'''

"""max_K_height = cv.boundingRect(max(merge_contours(filtered_contours), key=cv.contourArea))[3]
                    max_K_width = cv.boundingRect(max(merge_contours(filtered_contours), key=cv.contourArea))[2]

                    if max_K_height + max_K_width <= 450:
                        filtered_contours.clear()

                    if filtered_contours:
                        max_K_height = cv.boundingRect(max(merge_contours(filtered_contours), key=cv.contourArea))[3]
                        max_K_width = cv.boundingRect(max(merge_contours(filtered_contours), key=cv.contourArea))[2]

                        if max_K_height + max_K_width <= 450:
                            # if max_K_height <= 75 or max_K_width <= 150:
                            if max_K_height + max_K_width <= 350 and (max_K_height * 2.5 < max_K_width or max_K_height > max_K_width * 2.5):
                                filtered_contours.clear()
                            self.virtual_movement_feet(track, i, filtered_contours, box_center, vis, valid_points,
                                                       frame_counter, frame_gray, prev_mean_shift, mean_shift)
                            continue"""

"""@staticmethod
    def _filter_valid_points(p1, st, features, fgmask, box):
        x, y, w, h = box
        valid_points = p1[st == 1].reshape(-1, 2)
        previous_points = features[st == 1].reshape(-1, 2)

        box_filter = [
            (x - 55 <= p[0] <= x + w + 55) and (y - 55 <= p[1] <= y + h + 55) for p in valid_points
        ]

        mask_filter = [  # +25
            not np.all(fgmask[int(p[1]):int(p[1]) + 5, int(p[0]):int(p[0]) + 5] == 0) for p in valid_points
        ]

        combined_filter = np.array(box_filter) & np.array(mask_filter)

        return valid_points[combined_filter], previous_points[combined_filter]"""
