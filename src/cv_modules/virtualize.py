import numpy as np
import cv2 as cv

class Virtualizer:
    def __init__(self):
        self.virt_frame_counter = 0

    def virtual_movement(self):
        print("VIRT")
        if len(self.last_partial_virt_box_tracks) > 0:
            for track in self.last_partial_virt_box_tracks:
                x, y, w, h = track["box"]
                x = np.int32(x + np.mean([shift[0] for shift in track["mean_shift"]]) * 2)
                track["box"] = (x, y, w, h)
                track["center"] = [(x + (w // 2)), (y + (h // 2))]
            self.last_box_tracks = self.last_partial_virt_box_tracks.copy()
            self.last_partial_virt_box_tracks = []
        else:
            for track in self.last_box_tracks:
                x, y, w, h = track["box"]
                x = np.int32(x + np.mean([shift[0] for shift in track["mean_shift"]]) * 2)
                # y = np.int32(y + track["mean_shift"][1]) # Y-nicht
                track["box"] = (x, y, w, h)
                track["center"] = [(x + (w // 2)), (y + (h // 2))]

    def virtual_movement_partial(self, track, index, filtered_contours, box_center, vis,
                                 valid_points, frame_counter, frame_gray, prev_mean_shift, mean_shift):
        self.lock = 0
        x, y, w, h = track["box"]
        dx, dy, dw, dh = self.last_box_tracks[index]["box"]
        alpha = 0.15

        track["box"] = (x, dy, w, dh)
        track["center"] = [(x + (w // 2)), (dy + (dh // 2))]

        valid_y_coords = np.float32(self.last_box_tracks[index]["features"]).reshape(-1, 1, 2).reshape(-1, 2)[:, 1]
        valid_x_coords = np.float32(self.last_box_tracks[index]["features"]).reshape(-1, 1, 2).reshape(-1, 2)[:, 0]
        min_y = int(np.min(valid_y_coords)) - 20
        max_y = int(np.max(valid_y_coords)) + 20
        min_x = int(np.min(valid_x_coords)) - 20
        max_x = int(np.max(valid_x_coords)) + 20

        # Berechne dynamische Höhe basierend auf Feature-Punkten
        dynamic_height = max(max_y - min_y, 50)
        dynamic_width = max(max_x - min_x, 50)  # Breite bleibt stabil oder wird angepasst

        if filtered_contours:
            cv.drawContours(vis, filtered_contours, -1,
                            (0, 255, 0), 2)
            contour_centers = []
            for contour in filtered_contours:

                M = cv.moments(contour)
                cx = int(M["m10"] / (M["m00"] + 1e-5))  # x-Koordinate
                cy = self.last_box_tracks[index]["center"][1]  # y-Koordinate
                contour_centers.append((cx, cy))

            if contour_centers:

                distances = [np.linalg.norm(np.array(center) - box_center) for center in contour_centers]

                weights = 1 / (np.array(distances) + 1e-5)
                weights /= np.sum(weights)

                weighted_center = np.sum(np.array(contour_centers) * weights[:, None], axis=0)

                even_center = (1 - alpha) * box_center + alpha * weighted_center
                new_x = int(even_center[0] - dynamic_width // 2)
                new_y = int(even_center[1] - dynamic_height // 2)

                self.updated_box_tracks.append({
                    "box": (new_x, new_y, dynamic_width, dynamic_height),
                    "features": valid_points.tolist() if frame_counter % 5 != 0 else
                    self.reinitialize_features(
                        frame_gray, (new_x, new_y, w, h), filtered_contours),
                    "mean_shift": (
                        prev_mean_shift := [[mean_shift[0], mean_shift[1]]] if prev_mean_shift is None or len(
                            prev_mean_shift) > 25
                        else prev_mean_shift + [[mean_shift[0], mean_shift[1]]]),
                    "center": [new_x + dynamic_width // 2, new_y + dynamic_height // 2],
                    "id": track["id"],
                    "hist": track["hist"]
                })

                self.last_partial_virt_box_tracks = self.updated_box_tracks.copy()

    def check_virt(self, boxes, points, vis, height, width):
        if len(boxes) <= 0 < len(self.last_box_tracks) and self.virt_frame_counter <= 50:
            margin = 50  # Abstand vom Kamerarand in Pixeln
            valid_tracks = []
            for track in self.last_box_tracks:
                x, y, w, h = track["box"]
                # Bedingung: Box nicht in der Nähe des Randes
                if x > margin and y > margin and (x + w) < (width - margin) and (y + h) < (
                        height - margin):
                    valid_tracks.append(track)  # Box ist gültig

            if valid_tracks:  # Nur wenn es gültige Tracks gibt
                self.last_box_tracks = valid_tracks
                self.virtual_movement()
                #draw_features(vis, points, self.last_box_tracks)
                #draw_boxes(vis, self.last_box_tracks)
                self.virt_frame_counter += 1