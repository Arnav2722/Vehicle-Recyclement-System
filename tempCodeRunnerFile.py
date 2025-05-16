import cv2 as cv
import numpy as np

class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}  # Store the center positions of detected objects
        self.id_count = 0  # Keep track of object IDs

    def update(self, detections):
        objects_bbs_ids = []
        new_center_points = {}

        for detection in detections:
            x, y, w, h = detection
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            # Find if this object was already detected
            same_object_detected = False
            for object_id, (px, py) in self.center_points.items():
                distance = np.hypot(cx - px, cy - py)
                if distance < 20:  # Adjusted threshold for better tracking
                    same_object_detected = True
                    new_center_points[object_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, object_id])
                    break

            if not same_object_detected:
                self.id_count += 1
                new_center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])

        self.center_points = new_center_points.copy()
        return objects_bbs_ids


def findObjects(outputs, img, confThreshold, nmsThreshold, tracker):
    hT, wT, _ = img.shape
    bbox, classIds, confs = [], [], []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append(bbox[i])

    boxes_ids = tracker.update(detections)
    for x, y, w, h, obj_id in boxes_ids:
        cv.putText(img, f'ID {obj_id}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return len(boxes_ids)  # Return count of objects detected


def main(video_path="Resources/Drone.mp4"):
    cap = cv.VideoCapture(video_path)
    whT, confThreshold, nmsThreshold = 320, 0.6, 0.2
    tracker = EuclideanDistTracker()
    total_objects = 0

    # Load class names
    classesFile = "Resources/coco.names"
    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip().split('\n')

    # Load YOLO model
    net = cv.dnn.readNetFromDarknet("Resources/custom-yolov4-tiny-detector.cfg", "Resources/custom-yolov4-tiny-detector_best.weights")
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or unable to fetch frame.")
            break

        frame = cv.resize(frame, (0, 0), fx=0.6, fy=0.6)
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outputNames = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)

        total_objects = findObjects(outputs, frame, confThreshold, nmsThreshold, tracker)
        cv.putText(frame, f'Count: {total_objects}', (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('Vehicle Counter', frame)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
