import cv2
import supervision as sv
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
import subprocess

image = 'test.jpg'

MODEL_ID = 'underwaterworld-bpi5h/1'


config = InferenceConfiguration(confidence_threshold = 0.5, iou_threshold = 0.1)


client = InferenceHTTPClient(
    api_url = "http://localhost:9001",
    api_key = "MESgIVvUduefNb8muGmq",
    )

client.configure(config)

client.select_model(MODEL_ID)

class_ids = {}

predictions = client.infer(image)

print(predictions)

class_ids = {}
image_width = predictions['image']['width']
image_height = predictions['image']['height']
for p in predictions["predictions"]:
    x = p['x']
    y = p['y']
    class_id = p["class_id"]
    text = p["class"]
    if x > image_width / 2:
        print(f"{text} on your right")
    if x < image_width / 2:
        print(f"{text} on your left")
    if class_id not in class_ids:
        class_ids[class_id] = p["class"]

detections = sv.Detections.from_inference(predictions)

image = cv2.imread(image)

box_annotator = sv.RoundBoxAnnotator()
detections.data["class_name"] = [
    f"{class_ids[class_id]} {confidence:0.2f}"
    for class_id, confidence in zip(detections.class_id,detections.confidence)
]

annotated_frame = box_annotator.annotate(
    scene=image.copy(), detections=detections
    
)

sv.plot_image(image=annotated_frame, size=(16, 16))
