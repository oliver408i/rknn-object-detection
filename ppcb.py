import cv2
import numpy as np

class Yolov8:
    @staticmethod
    def filter_detections(outputs, threshold=0.1):
        """
        Filters the detections based on a confidence threshold and returns them
        as (xmin, ymin, xmax, ymax, confidence).

        Parameters:
            outputs (np.ndarray): The raw output from the model with shape [1, 5, 8400].
            threshold (float): The confidence threshold to filter detections.

        Returns:
            List of tuples: Each tuple contains (xmin, ymin, xmax, ymax, confidence).
        """
        # Reshape and transpose the output
        reshaped_output = outputs[0].reshape(5, 8400).transpose()

        # Prepare a list to store valid detections
        detections = []

        # Iterate over the reshaped output to filter by confidence
        for row in reshaped_output:
            x_center, y_center, width, height, confidence = row
            
            # Check if the confidence is above the threshold
            if confidence >= threshold:
                # Convert (x_center, y_center, width, height) to (xmin, ymin, xmax, ymax)
                xmin = x_center - width / 2
                ymin = y_center - height / 2
                xmax = x_center + width / 2
                ymax = y_center + height / 2
                
                # Append the valid detection to the list
                detections.append((xmin, ymin, xmax, ymax, confidence))

        # Apply Non-Maximum Suppression
        filtered_detections = Yolov8.non_max_suppression(detections)

        return filtered_detections

    @staticmethod
    def non_max_suppression(detections, iou_threshold=0.5):
        if len(detections) == 0:
            return []

        boxes = np.array([[d[0], d[1], d[2], d[3]] for d in detections])
        scores = np.array([float(d[4]) for d in detections])

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.0, nms_threshold=iou_threshold)

        if isinstance(indices, tuple):
            indices = indices[0]

        filtered_detections = [detections[i] for i in indices.flatten()]

        return filtered_detections

class CommonOps:
    @staticmethod
    def draw_boxes(img, detections):
        
        # Resize the image to 640x640 pixels
        #img_resized = cv2.resize(img, (640, 640))

        for detection in detections:
            xmin = int(detection[0])
            ymin = int(detection[1])
            xmax = int(detection[2])
            ymax = int(detection[3])
            confidence = float(detection[4])

            # Create a semi-transparent overlay
            overlay = img.copy()
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (0, 255, 0), -1)

            # Apply the overlay with transparency
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            # Draw the bounding box outline
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

            # Optional: Add confidence score to the box with a smaller font
            label = f'{confidence:.2f}'
            cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


        # Convert back from RGB to BGR for displaying with OpenCV
        #img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        return img

class Yolov5:
    @staticmethod
    def convert_bbox_format(detection):
        x_center = int(float(detection[0]))
        y_center = int(float(detection[1]))
        width = int(float(detection[2]))
        height = int(float(detection[3]))

        xmin = int(x_center - (width / 2))
        ymin = int(y_center - (height / 2))
        xmax = int(x_center + (width / 2))
        ymax = int(y_center + (height / 2))

        return xmin, ymin, xmax, ymax

    @staticmethod
    def filter_detections(outputs, threshold=0.5):
        data = outputs[0].reshape(-1).tolist()
        detections = [data[i:i+6] for i in range(0, len(data), 6)]

        # Convert from (Xcenter, Ycenter, Width, Height) to (Xmin, Ymin, Xmax, Ymax)
        converted_detections = []
        for detection in detections:
            confidence_score = float(detection[4])
            if confidence_score >= threshold:
                xmin, ymin, xmax, ymax = Yolov5.convert_bbox_format(detection[:4])
                converted_detections.append([xmin, ymin, xmax, ymax, detection[4]])

        # Apply Non-Maximum Suppression
        filtered_detections = Yolov5.non_max_suppression(converted_detections)

        return filtered_detections

    @staticmethod
    def non_max_suppression(detections, iou_threshold=0.5):
        if len(detections) == 0:
            return []

        boxes = np.array([[d[0], d[1], d[2], d[3]] for d in detections])
        scores = np.array([float(d[4]) for d in detections])

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.0, nms_threshold=iou_threshold)

        if isinstance(indices, tuple):
            indices = indices[0]

        filtered_detections = [detections[i] for i in indices.flatten()]

        return filtered_detections