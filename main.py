import cv2 as cv
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml
import numpy as np

CLASSES = yaml_load(check_yaml("./coco8.yaml"))["names"]
colors = np.random.uniform(0,225,size=(len(CLASSES), 3)) # Create random colors for each element of the file 

model: cv.dnn.Net = cv.dnn.readNetFromONNX("./yolov8n.onnx")

def draw_boxes(img, class_id, conf, x, y, x_plus_w, x_plus_h):
    label = f"{CLASSES[class_id]} {conf:.2f}"
    color = colors[class_id]
    cv.rectangle(img, (x, y), (x_plus_w, x_plus_h), color, 1)
    cv.putText(img, label, (x - 10, y - 10), cv.FONT_HERSHEY_PLAIN, 1, color, 2)

def infer():

    original_image: np.ndarray = cv.imread("./photo.png")
    [height, width, _] = original_image.shape

    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    scale = length / 640

    blob = cv.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(640,640), swapRB=True)
    model.setInput(blob)

    outputs = model.forward()

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
    result_boxes = cv.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []

    # Iterate through NMS results to draw bounding boxes and labels
    for i in range(len(result_boxes)):
        print("Class IDs:", class_ids)
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_boxes(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )

    # Display the image with bounding boxes
    cv.imshow("image", original_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return detections

if __name__ == "__main__":
    print("TAP Vision System")

    # get per frame camera
    cap = cv.VideoCapture(2)
    if not cap.isOpened():
        print("No camera feed is found.")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read frame")
            break
        cv.imshow("frame",frame)

    cap.release()
    cv.destroyAllWindows()
