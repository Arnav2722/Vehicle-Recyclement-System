import cv2 as cv
import numpy as np
import os
import sqlite3
from datetime import datetime


class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}  # Store the center positions of detected objects
        self.id_count = 0  # Keep track of object IDs
        self.object_classes = {}  # Store object classes by ID

    def update(self, detections, class_ids=None):
        objects_bbs_ids = []
        new_center_points = {}
        new_object_classes = {}

        for i, detection in enumerate(detections):
            x, y, w, h = detection
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            # Get class if available
            class_id = class_ids[i] if class_ids else None

            same_object_detected = False
            for object_id, (px, py) in self.center_points.items():
                distance = np.hypot(cx - px, cy - py)
                if distance < 20:  # Threshold for considering the same object
                    same_object_detected = True
                    new_center_points[object_id] = (cx, cy)
                    # Keep the original class or update if we have a new one
                    if class_id is not None:
                        new_object_classes[object_id] = class_id
                    elif object_id in self.object_classes:
                        new_object_classes[object_id] = self.object_classes[object_id]
                    objects_bbs_ids.append([x, y, w, h, object_id])
                    break

            if not same_object_detected:
                self.id_count += 1
                new_center_points[self.id_count] = (cx, cy)
                if class_id is not None:
                    new_object_classes[self.id_count] = class_id
                objects_bbs_ids.append([x, y, w, h, self.id_count])

        self.center_points = new_center_points.copy()
        self.object_classes = new_object_classes.copy()
        return objects_bbs_ids


class SQLDatabase:
    def __init__(self, db_path="vehicle_detection.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # Create sessions table
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS detection_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TIMESTAMP,
            video_path TEXT,
            description TEXT
        )
        """
        )

        # Create vehicles table
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS detected_vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            vehicle_id INTEGER,
            class_name TEXT,
            detection_time TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES detection_sessions(session_id)
        )
        """
        )

        # Create summary table
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS session_summary (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            total_vehicles INTEGER,
            car_count INTEGER,
            truck_count INTEGER,
            bus_count INTEGER,
            motorcycle_count INTEGER,
            bicycle_count INTEGER,
            other_count INTEGER,
            FOREIGN KEY (session_id) REFERENCES detection_sessions(session_id)
        )
        """
        )

        self.conn.commit()

    def start_new_session(self, video_path, description="Vehicle detection session"):
        self.cursor.execute(
            "INSERT INTO detection_sessions (start_time, video_path, description) VALUES (?, ?, ?)",
            (datetime.now(), video_path, description),
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def add_vehicle(self, session_id, vehicle_id, class_name):
        self.cursor.execute(
            "INSERT INTO detected_vehicles (session_id, vehicle_id, class_name, detection_time) VALUES (?, ?, ?, ?)",
            (session_id, vehicle_id, class_name, datetime.now()),
        )
        self.conn.commit()

    def update_summary(self, session_id, vehicle_counts):
        # Check if summary exists
        self.cursor.execute(
            "SELECT * FROM session_summary WHERE session_id = ?", (session_id,)
        )
        if self.cursor.fetchone() is None:
            # Create new summary
            self.cursor.execute(
                """INSERT INTO session_summary 
                (session_id, total_vehicles, car_count, truck_count, bus_count, motorcycle_count, bicycle_count, other_count) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    vehicle_counts["total"],
                    vehicle_counts.get("car", 0),
                    vehicle_counts.get("truck", 0),
                    vehicle_counts.get("bus", 0),
                    vehicle_counts.get("motorcycle", 0),
                    vehicle_counts.get("bicycle", 0),
                    vehicle_counts.get("other", 0),
                ),
            )
        else:
            # Update existing summary
            self.cursor.execute(
                """UPDATE session_summary 
                SET total_vehicles = ?, car_count = ?, truck_count = ?, bus_count = ?,
                motorcycle_count = ?, bicycle_count = ?, other_count = ?
                WHERE session_id = ?""",
                (
                    vehicle_counts["total"],
                    vehicle_counts.get("car", 0),
                    vehicle_counts.get("truck", 0),
                    vehicle_counts.get("bus", 0),
                    vehicle_counts.get("motorcycle", 0),
                    vehicle_counts.get("bicycle", 0),
                    vehicle_counts.get("other", 0),
                    session_id,
                ),
            )
        self.conn.commit()

    def get_session_summary(self, session_id=None):
        if session_id:
            self.cursor.execute(
                "SELECT * FROM session_summary WHERE session_id = ?", (session_id,)
            )
            return self.cursor.fetchone()
        else:
            self.cursor.execute(
                "SELECT * FROM session_summary ORDER BY session_id DESC"
            )
            return self.cursor.fetchall()

    def get_vehicle_class_counts(self, session_id):
        self.cursor.execute(
            "SELECT class_name, COUNT(*) FROM detected_vehicles WHERE session_id = ? GROUP BY class_name",
            (session_id,),
        )
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()


def findObjects(outputs, img, confThreshold, nmsThreshold, tracker, classNames):
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

    detections = []
    detected_class_ids = []
    if bbox:
        indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append(bbox[i])
                detected_class_ids.append(classIds[i])

    boxes_ids = tracker.update(detections, detected_class_ids)
    vehicle_count_by_class = {}

    for i, (x, y, w, h, obj_id) in enumerate(boxes_ids):
        class_id = tracker.object_classes.get(obj_id, 0)
        class_name = classNames[class_id] if class_id < len(classNames) else "unknown"

        # Update count for this class
        if class_name not in vehicle_count_by_class:
            vehicle_count_by_class[class_name] = 0
        vehicle_count_by_class[class_name] += 1

        # Display ID number only on the vehicle
        cv.putText(
            img,
            f"ID {obj_id}",
            (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return boxes_ids, vehicle_count_by_class


def main(video_path=None, db_path="vehicle_detection.db"):
    base_path = os.path.dirname(
        os.path.abspath(__file__)
    )  # Use relative path for better portability

    if video_path is None:
        video_path = os.path.join(base_path, "Resources", "Drone.mp4")

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    # Initialize SQL database
    db = SQLDatabase(db_path)
    session_id = db.start_new_session(video_path)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_path}'")
        db.close()
        return

    whT, confThreshold, nmsThreshold = 320, 0.6, 0.2
    tracker = EuclideanDistTracker()

    classesFile = os.path.join(base_path, "Resources", "coco.names")
    if not os.path.exists(classesFile):
        print(f"Error: Class names file '{classesFile}' not found.")
        db.close()
        return

    with open(classesFile, "rt") as f:
        classNames = f.read().rstrip().split("\n")

    cfg_file = os.path.join(base_path, "Resources", "custom-yolov4-tiny-detector.cfg")
    weights_file = os.path.join(
        base_path, "Resources", "custom-yolov4-tiny-detector_best.weights"
    )

    if not os.path.exists(cfg_file) or not os.path.exists(weights_file):
        print("Error: Model files not found. Please check paths.")
        db.close()
        return

    net = cv.dnn.readNetFromDarknet(cfg_file, weights_file)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    try:
        outputNames = [
            layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()
        ]
    except AttributeError:
        outputNames = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Vehicle tracking variables
    processed_vehicle_ids = set()
    frame_count = 0

    # Create a header section on the top of the video frame
    def create_header(frame, vehicle_count_by_class, total_objects):
        # Create a dark overlay for the header
        header_height = 80
        overlay = frame.copy()
        cv.rectangle(overlay, (0, 0), (frame.shape[1], header_height), (40, 40, 40), -1)
        alpha = 0.7  # Transparency factor
        cv.addWeighted(
            overlay[0:header_height, :],
            alpha,
            frame[0:header_height, :],
            1 - alpha,
            0,
            frame[0:header_height, :],
        )

        # Draw dividing lines
        cv.line(
            frame,
            (0, header_height),
            (frame.shape[1], header_height),
            (255, 255, 255),
            2,
        )  # Bottom line
        divider_x = frame.shape[1] // 2
        cv.line(
            frame, (divider_x, 0), (divider_x, header_height), (255, 255, 255), 2
        )  # Middle divider

        # Display headers
        cv.putText(
            frame, "NAME", (20, 30), cv.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2
        )
        cv.putText(
            frame,
            "NO. OF CARS",
            (divider_x + 20, 30),
            cv.FONT_HERSHEY_TRIPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        # Display vehicle counts
        class_text = ", ".join(
            [class_name for class_name in vehicle_count_by_class.keys()]
        )
        if len(class_text) > 30:  # Truncate if too long
            class_text = class_text[:30] + "..."
        cv.putText(
            frame,
            class_text,
            (20, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv.putText(
            frame,
            str(total_objects),
            (divider_x + 20, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or unable to fetch frame.")
            break

        frame = cv.resize(frame, (640, 480))  # Ensuring standard frame size
        blob = cv.dnn.blobFromImage(
            frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False
        )
        net.setInput(blob)
        outputs = net.forward(outputNames)

        boxes_ids, vehicle_count_by_class = findObjects(
            outputs, frame, confThreshold, nmsThreshold, tracker, classNames
        )

        # Add newly detected vehicles to the database
        for x, y, w, h, obj_id in boxes_ids:
            if obj_id not in processed_vehicle_ids:
                class_id = tracker.object_classes.get(obj_id, 0)
                class_name = (
                    classNames[class_id] if class_id < len(classNames) else "unknown"
                )
                db.add_vehicle(session_id, obj_id, class_name)
                processed_vehicle_ids.add(obj_id)

        # Update session summary every 30 frames
        frame_count += 1
        if frame_count % 30 == 0:
            total_count = len(processed_vehicle_ids)
            vehicle_counts = {"total": total_count}

            # Get counts by class from database
            class_counts = db.get_vehicle_class_counts(session_id)
            for class_name, count in class_counts:
                vehicle_counts[class_name.lower()] = count

            db.update_summary(session_id, vehicle_counts)

        # Create header with NAME and NO. OF CARS
        total_objects = len(boxes_ids)
        create_header(frame, vehicle_count_by_class, total_objects)

        # Add "DRONE SURVEILLANCE COUNT" text at the top
        cv.putText(
            frame,
            "DRONE SURVEILLANCE COUNT",
            (frame.shape[1] // 4, 20),
            cv.FONT_HERSHEY_TRIPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv.imshow("Vehicle Counter", frame)

        if cv.waitKey(10) & 0xFF == 27:  # Press 'Esc' to exit
            break

    # Final database update
    total_count = len(processed_vehicle_ids)
    vehicle_counts = {"total": total_count}

    # Get counts by class from database
    class_counts = db.get_vehicle_class_counts(session_id)
    for class_name, count in class_counts:
        vehicle_counts[class_name.lower()] = count

    db.update_summary(session_id, vehicle_counts)

    # Print final summary
    print("\n----- VEHICLE DETECTION SUMMARY -----")
    print(f"Total vehicles detected: {total_count}")
    for class_name, count in class_counts:
        print(f"{class_name}: {count}")
    print("------------------------------------\n")

    # Close resources
    cap.release()
    cv.destroyAllWindows()
    db.close()


def print_database_summary(db_path="vehicle_detection.db"):
    """Utility function to query and print database summary"""
    db = SQLDatabase(db_path)

    print("\n----- DATABASE SUMMARY -----")

    # Get all sessions
    db.cursor.execute(
        "SELECT session_id, start_time, video_path FROM detection_sessions ORDER BY session_id"
    )
    sessions = db.cursor.fetchall()

    for session in sessions:
        session_id, start_time, video_path = session
        print(f"\nSession {session_id} - {start_time} - {os.path.basename(video_path)}")

        # Get session summary
        summary = db.get_session_summary(session_id)
        if summary:
            (
                summary_id,
                session_id,
                total,
                cars,
                trucks,
                buses,
                motorcycles,
                bicycles,
                others,
            ) = summary
            print(f"Total vehicles: {total}")
            print(f"Cars: {cars}, Trucks: {trucks}, Buses: {buses}")
            print(f"Motorcycles: {motorcycles}, Bicycles: {bicycles}, Others: {others}")

        # Get unique vehicles by class
        class_counts = db.get_vehicle_class_counts(session_id)
        print("Vehicle classes detected:")
        for class_name, count in class_counts:
            print(f"  {class_name}: {count}")

    db.close()


if __name__ == "__main__":
    main()
