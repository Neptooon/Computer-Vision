import csv
import json
import os
import cv2
import motmetrics as mm
import numpy as np
# Quelle der Orientierung: https://github.com/cheind/py-motmetrics/tree/develop

class MOTA_Metrik:

    def __init__(self, video):
        self.video = cv2.VideoCapture(video)
        self.frame_count = 0

    def extract_frames(self, gt_file):

        frames = set()

        # Ground-Truth-Datei lesen
        with open(gt_file, 'r') as gt_file:
            reader = csv.reader(gt_file)
            for row in reader:
                frame = int(row[0])
                frames.add(frame)

        return sorted(frames)

    def write_track_info(self, output_file, tracks, frame):

        with open(output_file, 'a') as outfile:
            for track in tracks:
                track_id = track.id
                x, y, w, h = track.box
                # MOT16-Format mit Leerzeichen und Komma
                line = f"{frame}, {track_id}, {x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}, 1, -1, -1, -1\n"
                outfile.write(line)

    def motMetricsEnhancedCalculator(self, gtSource, pdSource, frames, json_file_path, video_name):

        # load ground truth
        gt = np.loadtxt(gtSource, delimiter=',')

        # load tracking output
        pd = np.loadtxt(pdSource, delimiter=',')

        acc = mm.MOTAccumulator(auto_id=True)

        for frame in frames:

            gt_dets = gt[gt[:, 0] == frame, 1:6]  # all detections in gt
            t_dets = pd[pd[:, 0] == frame, 1:6]  # all detections in pd

            C = mm.distances.iou_matrix(gt_dets[:, 1:], t_dets[:, 1:],
                                        max_iou=0.5)  # format: gt, pd

            # acc update
            acc.update(gt_dets[:, 0].astype('int').tolist(),
                       t_dets[:, 0].astype('int').tolist(), C)

        mh = mm.metrics.create()

        summary = mh.compute(acc, metrics=[
            'num_frames',
            'num_matches',
            'num_switches',
            'num_false_positives',
            'num_misses',
            'num_detections',
            'num_objects',
            'num_predictions',
            'num_unique_objects',
            'mostly_tracked',
            'partially_tracked',
            'mostly_lost',
            'num_fragmentations',
            'motp',
            'mota',
            'precision',
            'recall'
        ], name='acc')


        results = {
            'video_name': video_name,
            'num_frames': int(summary['num_frames'].iloc[0]),
            'num_matches': int(summary['num_matches'].iloc[0]),
            'num_switches': int(summary['num_switches'].iloc[0]),
            'num_false_positives': int(summary['num_false_positives'].iloc[0]),
            'num_misses': int(summary['num_misses'].iloc[0]),
            'num_detections': int(summary['num_detections'].iloc[0]),
            'num_objects': int(summary['num_objects'].iloc[0]),
            'num_predictions': int(summary['num_predictions'].iloc[0]),
            'num_unique_objects': int(summary['num_unique_objects'].iloc[0]),
            'mostly_tracked': int(summary['mostly_tracked'].iloc[0]),
            'partially_tracked': int(summary['partially_tracked'].iloc[0]),
            'mostly_lost': int(summary['mostly_lost'].iloc[0]),
            'num_fragmentations': int(summary['num_fragmentations'].iloc[0]),
            'motp': float(summary['motp'].iloc[0]),
            'mota': float(summary['mota'].iloc[0]),
            'precision': float(summary['precision'].iloc[0]),
            'recall': float(summary['recall'].iloc[0])
        }

        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                try:
                    data = json.load(json_file)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        data.append(results)

        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
