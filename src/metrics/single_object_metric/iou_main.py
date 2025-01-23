import os
import json

from src.cv_modules.main import MultipleObjectTrackingPipeline


video_folder = '../../assets/videos/single_detect_videos'  # Ordner mit Videos


def iou_results(video_folder):
    output_file = "IoU_results.json"
    results_iou = []
    for video_file in os.listdir(video_folder): # Alle Videos durchgehen
        if video_file.endswith('.mov'):
            video_path = os.path.join(video_folder, video_file)
            print(f"Aktuelles Video: {video_file}")

            # Pipeline
            pipeline = MultipleObjectTrackingPipeline(video_path)  # SOT ausführen
            pipeline.run()
            pipeline.iou_metrik.save_data()  # Daten für Metrik erheben
            results_iou.append(pipeline.iou_metrik.tracking_data) # Speichern

    with open(output_file, "w") as file:
        json.dump(results_iou, file, indent=4)
    return results_iou, output_file


def compute_iou(box_a, ground_truth_box_b):
    # Boxen (x, y, w, h) in (x1, y1, x2, y2)

    if any(val is None for val in box_a) or any(val is None for val in ground_truth_box_b):
        return None
    x1_a, y1_a, w_a, h_a = box_a
    x2_a, y2_a = x1_a + w_a, y1_a + h_a

    x1_b, y1_b, w_b, h_b = ground_truth_box_b
    x2_b, y2_b = x1_b + w_b, y1_b + h_b

    # Schnittmenge
    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    # Vereinigungsmenge
    area_a = w_a * h_a
    area_b = w_b * h_b
    union = area_a + area_b - intersection

    # IoU berechnen
    return intersection / union if union > 0 else 0.0


def main():
    # Datenauswertung
    #iou_results(video_folder)
    #yolo_results(video_folder)


    with open('IoU_results.json') as f:
        results_iou = json.load(f)
    with open('YOLO_results.json') as f:
        results_yolo = json.load(f)

    video_ious = []

    for video_iou, video_yolo in zip(results_iou, results_yolo):
        video_name = video_iou[0]["video_file"]
        frame_ious = []

        for frame_iou, frame_yolo in zip(video_iou[0]["frames"], video_yolo[0]["frames"]):
            pred_boxes = frame_iou["detections"][0]["bbox"]
            gt_boxes = frame_yolo["detections"][0]["bbox"]

            iou = compute_iou(pred_boxes, gt_boxes)

            if iou is not None:
                frame_ious.append(iou)

        # Durchschnittliche IoU für ein Video
        if frame_ious:
            video_average_iou = sum(frame_ious) / len(frame_ious)
            video_ious.append({"video_name": video_name, "average_iou": video_average_iou})

        else:
            video_ious.append({"video_name": video_name, "average_iou": 0})

    #Gesamtdurchschnitt
    total_ious = [video["average_iou"] for video in video_ious if video["average_iou"] > 0]
    overall_average_iou = round(sum(total_ious) / len(total_ious), 2)if total_ious else 0

    zusammenfassung = {
        "Gesamtanzahl Videos": len(video_ious),
        "Overall Average IoU über alle Videos": overall_average_iou
    }

    all_ious = [{"Video Name": video["video_name"], "Average IoU": video["average_iou"]} for video in video_ious]

    with open('iou_final_results.json', "w") as f:
        json.dump(zusammenfassung, f, indent=4)

    with open('all_video_ious.json', "w") as f:
        json.dump(all_ious, f, indent=4)


main()
