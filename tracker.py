import math

class EuclideanDistTracker:
    def __init__(self):
        """
        Initializes the tracker.

        Attributes:
            center_points (dict): Stores the center positions of detected objects.
                                  Key: Object ID
                                  Value: (x, y) coordinates of the object's center
            id_count (int): Counter for assigning unique IDs to new objects.
        """
        self.center_points = {}
        self.id_count = 0

    def update(self, detections):
        """
        Updates the tracker with new detections.

        Args:
            detections (list): A list of bounding boxes (x, y, w, h) for detected objects.

        Returns:
            list: A list of bounding boxes with IDs: [x, y, w, h, id]
        """
        objects_bbs_ids = []

        for rect in detections:
            x, y, w, h = rect
            cx = (x + x + w) // 2  # Calculate center x
            cy = (y + y + h) // 2  # Calculate center y

            matched = False
            for id, pt in self.center_points.items():
                distance = math.hypot(cx - pt[0], cy - pt[1])

                # If distance is lesser than threshold, count as same object
                if distance < 25:  # Adjust threshold as needed
                    matched = True
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    break

            # If no match found, create a new track
            if not matched:
                self.id_count += 1
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])

        # Remove stale tracks (objects not detected in current frame)
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            new_center_points[object_id] = self.center_points[object_id]
        self.center_points = new_center_points.copy()

        return objects_bbs_ids