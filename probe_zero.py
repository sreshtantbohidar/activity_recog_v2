"""Reproduce ObjectDetector.detect on an image the model finds nothing in."""
import glob
import traceback

import cv2
from ultralytics import YOLO

target = sorted(glob.glob("/app/data/uploads/*activity_*"))[-1]
print("probing", target)
model = YOLO("/app/object_detect_model_256_class.pt")
result = model(target, conf=0.25, imgsz=640, verbose=False)[0]
print("boxes:", len(result.boxes))

try:
    rendered = result.plot()
    ok, encoded = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("plot/imencode ok:", ok, rendered.shape)
except Exception:
    traceback.print_exc()

try:
    detections = []
    for box, cls, confidence in zip(result.boxes.xyxy.cpu().tolist(),
                                    result.boxes.cls.cpu().tolist(),
                                    result.boxes.conf.cpu().tolist()):
        detections.append({"class": result.names[int(cls)], "confidence": round(float(confidence), 4)})
    print("loop ok, detections:", detections)
except Exception:
    traceback.print_exc()
