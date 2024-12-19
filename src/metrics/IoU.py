
class IoUMetrik:
    def __init__(self, video_path):
        self.tracking_data = []
        self.path = video_path
        self.video_info = {
            "video_file": self.path,
            "frames": []
            }

    def get_iou_info(self, box_tracks, frame):
        # Video-Daten initialisieren
        frame_info = {
            "frame_id": frame,
            "detections": []
        }
        person_found = False  # Flag, ob Personen gefunden wurden

        for track in box_tracks:  # Bounding Boxes in diesem Frame

            if track["box"]:
                person_found = True
                x, y, w, h = track["box"]
                frame_info["detections"].append({
                    "bbox": (int(x), int(y), int(w), int(h))
                })

        # Falls keine Personen gefunden wurden, Frame vermerken
        if not person_found:
            frame_info["detections"].append({
                "bbox": [None, None, None, None],
            })

        # Frame-Daten zum Video hinzufügen
        self.video_info["frames"].append(frame_info)

    def save_data(self):
        self.tracking_data.append(self.video_info)

