import csv
import cv2
import motmetrics as mm

class MOT_Evaluation:

    def __init__(self, video):
        self.video = cv2.VideoCapture(video)
        self.acc = mm.MOTAccumulator(auto_id=True)
        self.frame_count = 0

    def extract_frames(self, gt_file):

        frames = set()

        # Ground-Truth-Datei lesen
        with open(gt_file, 'r') as gt_file:
            reader = csv.reader(gt_file)
            for row in reader:
                frame = int(row[0])  # Erste Spalte enthält die Frame-Nummer
                frames.add(frame)

        return sorted(frames)

    def write_track_info(self, output_file, tracks, frame):

        with open(output_file, 'a') as outfile:
            for track in tracks:
                # Hole die Informationen aus dem Track
                track_id = track.id
                x, y, w, h = track.box
                # MOT16-Format mit Leerzeichen und Komma
                line = f"{frame}, {track_id}, {x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}, 1, -1, -1, -1\n"
                outfile.write(line)

    def motMetricsEnhancedCalculator(self, gtSource, tSource, frames):
        # import required packages
        import numpy as np

        # load ground truth
        gt = np.loadtxt(gtSource, delimiter=',')

        # load tracking output
        t = np.loadtxt(tSource, delimiter=',')

        # Create an accumulator that will be updated during each frame
        acc = mm.MOTAccumulator(auto_id=True)

        # Max frame number maybe different for gt and t files
        for frame in frames:
            # detection and frame numbers begin at 1

            # select id, x, y, width, height for current frame
            # required format for distance calculation is X, Y, Width, Height \
            # We already have this format
            gt_dets = gt[gt[:, 0] == frame, 1:6]  # select all detections in gt
            t_dets = t[t[:, 0] == frame, 1:6]  # select all detections in t

            C = mm.distances.iou_matrix(gt_dets[:, 1:], t_dets[:, 1:],
                                        max_iou=0.5)  # format: gt, t

            # Call update once for per frame.
            # format: gt object ids, t object ids, distance
            acc.update(gt_dets[:, 0].astype('int').tolist(),
                       t_dets[:, 0].astype('int').tolist(), C)

        mh = mm.metrics.create()

        summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', 'recall', 'precision', 'num_objects',
                                           'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_false_positives',
                                           'num_misses', 'num_switches', 'num_fragmentations', 'mota', 'motp'],
                             name='acc')

        strsummary = mm.io.render_summary(
            summary,
            # formatters={'mota' : '{:.2%}'.format},
            namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll',
                     'precision': 'Prcn', 'num_objects': 'GT',
                     'mostly_tracked': 'MT', 'partially_tracked': 'PT',
                     'mostly_lost': 'ML', 'num_false_positives': 'FP',
                     'num_misses': 'FN', 'num_switches': 'IDsw',
                     'num_fragmentations': 'FM', 'mota': 'MOTA', 'motp': 'MOTP',
                     }
        )
        print(strsummary)
