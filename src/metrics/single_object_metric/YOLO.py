from ultralytics import YOLO
import os
import json


# Load an official or custom model
model = YOLO("yolo11n.pt")  # Load an official Detect model
#model = YOLO("yolo11n-seg.pt")  # Load an official Segment model
#model = YOLO("yolo11n-pose.pt")  # Load an official Pose model
#model = YOLO("path/to/best.pt")  # Load a custom trained model

results = model.track("../../assets/videos/LL-Parkour-CloseUp-Pulli-Hell.mov", show=True)
def yolo_results(video_folder):
    tracking_data = []
    output_file = "YOLO_results.json"

    for video_file in os.listdir(video_folder):
        if video_file.endswith(".mov"):  # Unterstützte Videoformate
            video_path = os.path.join(video_folder, video_file)
            print(f"Processing video: {video_path}")

            # Video-Daten initialisieren
            video_info = {
                "video_file": video_file,
                "frames": []
            }

            # Tracking durchführen
            results = model.track(video_path, show=True)  # ByteTrack verwenden

            for frame_id, frame in enumerate(results):  # Über jedes Frame iterieren
                frame_info = {
                    "frame_id": frame_id,
                    "detections": []
                }
                person_found = False  # Flag, ob Personen gefunden wurden

                for box in frame.boxes:  # Bounding Boxes in diesem Frame
                    if box.xyxy is not None and box.cls is not None:  # Überprüfen, ob die Box valide ist
                        class_id = int(box.cls.numpy())  # Klassen-ID der Detektion (z. B. 0 für Personen)

                        if class_id == 0:  # Nur Personen speichern
                            person_found = True
                            xyxy = box.xyxy.numpy()  # Koordinaten der Bounding Box
                            confidence = float(box.conf.numpy())  # Konfidenzwert
                            track_id = int(box.id) if box.id is not None else -1  # Tracking-ID, falls vorhanden

                            # Bounding Box-Daten hinzufügen
                            frame_info["detections"].append({
                                "bbox": (int(xyxy[0][0]), int(xyxy[0][1]), int(xyxy[0][2] - xyxy[0][0]), int(xyxy[0][3] - xyxy[0][1])) #(x, y, w, h)
                            })

                # Falls keine Personen gefunden wurden, Frame vermerken
                if not person_found:
                    frame_info["detections"].append({
                        "bbox": [None, None, None, None],
                    })

                # Frame-Daten zum Video hinzufügen
                video_info["frames"].append(frame_info)

            # Video-Daten zur Gesamtstruktur hinzufügen
            tracking_data.append([video_info])
            print(f"Finished processing: {video_file}")

    # Ergebnisse in die JSON-Datei schreiben
    with open(output_file, "w") as f:
        json.dump(tracking_data, f, indent=4)
    print(f"Tracking results saved to {output_file}")
    return tracking_data, output_file



# Perform tracking with the model
#results = model.track("../../assets/videos/ML3-LL-Dunkel-Default-LiveDemo.mov", show=True)  # Tracking with default tracker
#results = model.track("../../assets/videos/ML3-LL-Hell-Tafel-Sprint-Turnaround.mov", show=True, tracker="bytetrack.yaml")  # with ByteTrack