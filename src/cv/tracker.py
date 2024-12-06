import cv2 as cv
import numpy as np



class BGS:
    def __init__(self):
        self.backgroundSubtraction = cv.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=50)
        self.backgroundSubtraction.setBackgroundRatio(0.7)
        self.backgroundSubtraction.setShadowValue(255)
        self.backgroundSubtraction.setShadowThreshold(0.3)
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    def bgs_apply(self, frame):

        fgmask = self.backgroundSubtraction.apply(frame)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, self.kernel)
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, self.kernel)
        return cv.medianBlur(fgmask, 3)


class Detector:
    def __init__(self):
        self.hog = cv.HOGDescriptor()
        self.hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame, fgmask):

        frame_down_sample = cv.resize(frame, ((frame.shape[1] // 25) * 10, (frame.shape[0] // 25) * 10))
        boxes, weights = self.hog.detectMultiScale(frame_down_sample, winStride=(8, 8), padding=(8, 8), scale=1.06)
        boxes = np.divide(boxes * 25, 10).astype(int)  # Up-sampling der Koordinaten

        return self.filter_boxes(boxes, fgmask)

    @staticmethod
    def _inside(r, q):
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q
        return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

    def filter_boxes(self, boxes, fgmask, min_area=1000):

        filtered_boxes = []
        for ri, r in enumerate(boxes):
            for qi, q in enumerate(boxes):
                if ri != qi and self._inside(r, q):
                    break
            else:
                filtered_boxes.append(r)
                print("FILTER")
        return [box for box in filtered_boxes if np.count_nonzero(fgmask[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]) > min_area]



class Tracker:
    def __init__(self):
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 15, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7) # 21 bzw. 7
        self.box_tracks = []
        self.last_box_tracks = []
        self.updated_box_tracks = []

    def init_tracks(self, boxes, frame_gray):

        for (x, y, w, h) in boxes:
            roi = frame_gray[y:y + h, x:x + w]  # ROI um Feature-Suche einzugrenzen
            #roi = frame_gray[(y+h)//2 - 250:(y+h)//2 + 250, (x+w)//2 - 150:(x+w)//2 + 150]  # ROI um Feature-Suche einzugrenzen
            features = cv.goodFeaturesToTrack(roi, **self.feature_params)
            if features is not None:
                self.box_tracks.append(
                    {
                        "box": (x, y, w, h),
                        "features": [(px + x, py + y) for px, py in np.float32(features).reshape(-1, 2)],
                        "mean_shift": [0, 0],
                        "center": [(x + w) // 2, (y + h) // 2]
                    }
                )

    def update_tracks(self, prev_gray, frame_gray, fgmask):

        points = []
        for track in self.box_tracks:
            box = track["box"]
            features = np.float32(track["features"]).reshape(-1, 1, 2)
            np.append(features, track["center"])


            # p1 = [    [[x1, y2]] ,    [[x2, y2]]      ]
            p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, features, None, **self.lk_params)
            p0, st0, err0 = cv.calcOpticalFlowPyrLK(frame_gray, prev_gray, p1, None, **self.lk_params)
            valid_points, previous_points = self._filter_valid_points(p1, st, p0, track, fgmask)

            if len(valid_points) > 0:
                # Berechne die durchschnittliche Verschiebung der Punkte
                points.append(valid_points)
                movement = valid_points - previous_points
                mean_shift = np.mean(movement, axis=0)

                x, y, w, h = box
                dx = mean_shift[0]
                dy = mean_shift[1]

                new_x = int(x + dx)  # +- 2% um Ausreißer zu glätten
                new_y = int(y + dy)

                if np.count_nonzero(fgmask[new_y:new_y + h, new_x:new_x + w]) > 1000:
                    self.updated_box_tracks.append({
                        "box": (new_x, new_y, w, h),
                        "features": valid_points.tolist(),
                        "mean_shift": mean_shift,
                        "center": [(x + w) // 2, (y + h) // 2]
                    })

        return points


    @staticmethod
    def _filter_valid_points(p1, st, features, box, fgmask):
        #p1[box["center"][1] - 5:box["center"][1] + 5, box["center"][0] - 5:box["center"][0] + 5]
        valid_points = p1[st == 1].reshape(-1, 2)
        previous_points = features[st == 1].reshape(-1, 2)
        mask_filter = [
            not np.all(fgmask[int(p[1]):int(p[1]) + 5, int(p[0]):int(p[0]) + 5] == 0)
            #and (int(p[1]) >= int(box["center"][1]) - 5) and (int(p[1]) <= int(box["center"][1]) + 5)
            #and (int(p[0]) >= int(box["center"][0]) - 5) and (int(p[0]) <= int(box["center"][0]) + 5)
            for p in valid_points
        ]
        return valid_points[mask_filter], previous_points[mask_filter]

    def virtual_movement(self):
        print("VIRT")
        for track in self.last_box_tracks:
            x, y, w, h = track["box"]
            x = np.int32(x + track["mean_shift"][0])
            #y = np.int32(y + track["mean_shift"][1]) # Y-nicht
            track["box"] = (x, y, w, h)

    def init_new_tracks(self):

        self.box_tracks = self.updated_box_tracks
        self.last_box_tracks = self.updated_box_tracks
        self.updated_box_tracks = []

    def draw_boxes(self, vis, box_tracks):
        for track in box_tracks:
            x, y, w, h = track["box"]
            cv.rectangle(vis, (x + int(0.15*w), y + int(0.05*h)), (x + w + int(0.15*w), y + h - int(0.025*h)), (255, 0, 0), 5)

    def draw_features(self, vis, features):
        if features is not None:
            for feature_list in features:
                for i, point in enumerate(feature_list):
                    if i == len(feature_list) - 1:
                        cv.circle(vis, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
                    else:
                        cv.circle(vis, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)




class SingleObjectTrackingPipeline:
    def __init__(self, video_path):
        self.cap = cv.VideoCapture(video_path)
        self.bgs = BGS()
        self.detector = Detector()
        self.tracker = Tracker()
        self.prev_gray = None

    def run(self):

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            vis = frame.copy()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            fgmask = self.bgs.bgs_apply(frame)

            print("yes", self.tracker.box_tracks)
            if len(self.tracker.box_tracks) < 1:
                print("IM IN")
                self.tracker.box_tracks.clear()
                boxes = self.detector.detect(frame, fgmask)
                print("ZEICHNE", len(boxes))
                self.tracker.init_tracks(boxes, frame_gray)
                self.tracker.draw_boxes(vis, self.tracker.box_tracks)

            else:
                points = self.tracker.update_tracks(self.prev_gray, frame_gray, fgmask)
                self.tracker.draw_features(vis, points)

            if len(self.tracker.updated_box_tracks) == 0 and len(self.tracker.box_tracks) == 0:
                if len(self.tracker.last_box_tracks) > 0:
                    self.tracker.virtual_movement()
                    self.tracker.box_tracks.clear()
                    self.tracker.draw_boxes(vis, self.tracker.last_box_tracks)
            elif len(self.tracker.updated_box_tracks) == 0 and len(self.tracker.box_tracks) >=1:
                points = self.tracker.update_tracks(self.prev_gray, frame_gray, fgmask)
                self.tracker.draw_features(vis, points)
                self.tracker.init_new_tracks()
                self.tracker.draw_boxes(vis, self.tracker.box_tracks)
            else:
                self.tracker.init_new_tracks()
                self.tracker.draw_boxes(vis, self.tracker.box_tracks)

            self.prev_gray = frame_gray

            cv.imshow('HOG', vis)
            cv.imshow('BG', fgmask)
            #cv.imshow('BGS', fgmask)
            key = cv.waitKey(20)

            if key & 0xFF == 27:
                break

            if key == ord('p'):
                cv.waitKey(-1)

        self.cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    pipeline = SingleObjectTrackingPipeline(0)
    pipeline.run()