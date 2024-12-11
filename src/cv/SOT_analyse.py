import os
import json
from tracker import SingleObjectTrackingPipeline

#Metrik für: Robustheit und Konsistenz des Trackens und der Detektion
#Robustheit:
# Komplizierte Gegebenheiten:
# - Lichtverhältnisse Hell / Dunkel
# - Unordnung im Hintergrund → BGS mehr unausgewogen
# - Occlusion: Verdeckungen des Zielobjekts
# Anpassungsfähigkeit / Adaptability:
# - Veränderungen in der Umgebung:
# - Komplexe Bewegungen wie plötzliches Umdrehen
# - Detector bzw. Tracker muss die sorgfältig erhobenen Charakteristiken des Zielobjekts tracken können
# Real-Time-Performance:
# - Verarbeitungsgeschwindigkeit muss in Echtzeit erfolgen



video_folder = '../../assets/videos/single_detect_videos'  # Ordner mit Videos
single_results = {} # Ergebnisse für Videos die tendenziell mit nur einmaliger Detektion und Tracking möglich sind

for video_file in os.listdir(video_folder):
    if video_file.endswith('.mov'):
        video_path = os.path.join(video_folder, video_file)
        print(f"Aktuelles Video: {video_file}")

        # Pipeline
        pipeline = SingleObjectTrackingPipeline(video_path)
        pipeline.run()

        # Ergebnisse sichern
        single_results[video_file] = {
            "Gesamt Anzahl Detektionen": pipeline.detect_counter,
            "Gesamt Anzahl Tracks": pipeline.tracking_counter,
            "Gesamtanzahl der stichhaltigen Frames": pipeline.detect_counter + pipeline.tracking_counter,
            "Anzahl Detektionen zu stichhaltigen Frames in %": pipeline.detect_counter / (pipeline.detect_counter + pipeline.tracking_counter) if (pipeline.detect_counter + pipeline.tracking_counter) > 0 else 0,
            "Anzahl Tracks zu stichhaltigen Frames %": pipeline.tracking_counter / (pipeline.detect_counter + pipeline.tracking_counter) if (pipeline.detect_counter + pipeline.tracking_counter) > 0 else 0,
            "Anzahl nicht stichhaltiger Frames": pipeline.frame_counter - (pipeline.detect_counter + pipeline.tracking_counter),
            "1-Detektion": 1 if pipeline.detect_counter == 1 else 0

        }


total_videos = len(single_results)
single_detect_success = sum([1 for result in single_results.values() if result["1-Detektion"] == 1])
avg_detection_ratio = round(sum([result["Anzahl Detektionen zu stichhaltigen Frames in %"] for result in single_results.values()]) / total_videos, 2)
avg_tracking_ratio = round(sum([result["Anzahl Tracks zu stichhaltigen Frames in %"] for result in single_results.values()]) /
                           total_videos, 2)
# Ergebnisse in eine JSON-Datei schreiben
with open('SOT_eval.json', 'w') as f:
    json.dump(single_results, f, indent=4)


zusammenfassung = {
    "Gesamtanzahl Videos": total_videos,
    "Best Case Detektion in Beziehung zur gesamt-Anz. der Videos": f"{single_detect_success}/{total_videos} = {round(single_detect_success / total_videos, 2)}",
    "Overall Average: Anzahl Detektionen zur stichhaltigen Frames": avg_detection_ratio,
    "Overall Average: Anzahl Tracks zu stichhaltigen Frames": avg_tracking_ratio,
    "Fehldetektion und Tracking": f"{total_videos - single_detect_success}/{total_videos} = {1 - avg_tracking_ratio - avg_detection_ratio}"
}

with open('SOT_result.json', 'w') as res:
    json.dump(zusammenfassung, res, indent=4)