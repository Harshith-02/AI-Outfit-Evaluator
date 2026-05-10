from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_humans(image_path):

    results = model(image_path)

    person_count = 0
    confidences = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            confidence = float(box.conf[0])

            # class 0 = person in COCO dataset
            if cls == 0:
                person_count += 1
                confidences.append(round(confidence, 2))

    return {
        "person_detected": person_count > 0,
        "person_count": person_count,
        "confidences": confidences
    }