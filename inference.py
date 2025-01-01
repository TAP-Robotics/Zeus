import cv2 as cv
import time
import numpy as np
import logging

from cv2.typing import MatLike
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

logger = logging.getLogger(__name__)

#WARN: if something goes wrong in loading CLASSES this might be the problem
CLASSES: str = yaml_load(check_yaml("./coco8.yaml"))["names"]

class FrameInference:
    """
    A class that handles object detection in a frame.
    """
    def __init__(self) -> None:
        self.colors = np.random.uniform(0,225,size=(len(CLASSES), 3)) # Create random colors for each element of the file 

        self.resizing_factor = 256

        self.model: cv.dnn.Net = cv.dnn.readNetFromONNX("./models/yolo11n.onnx")

        cv.setNumThreads(4)

    def draw_boxes(self, img:MatLike,class_id: int, conf: float, x: int, y:int, x_plus_w:int, x_plus_h:int):
        label = f"{CLASSES[class_id]} {conf:.2f}"
        color = self.colors[class_id]
        cv.rectangle(img, (x, y), (x_plus_w, x_plus_h), color, 1)
        cv.putText(img, label, (x - 10, y - 10), cv.FONT_HERSHEY_PLAIN, 1, color, 2)

    def forward(self, frame: MatLike):
        original_image: np.ndarray = frame
        [height, width, _] = original_image.shape


        start_time = time.time()  # Start the timer

        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        scale = length / self.resizing_factor

        blob = cv.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(self.resizing_factor,self.resizing_factor), swapRB=True)
        self.model.setInput(blob)

        outputs = self.model.forward()

        inference_time = time.time() - start_time
        logger.info(f"Inference Time: {inference_time:.3f} seconds")

        outputs = np.array([cv.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv.dnn.NMSBoxes(boxes, scores, 0.25, 0.3, 0.5)

        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)
            self.draw_boxes(
                original_image,
                class_ids[index],
                scores[index],
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale),
            )

        # Display the image with bounding boxes

        return original_image
