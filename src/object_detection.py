import cv2
import os
from inference_sdk import InferenceHTTPClient
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T


class RoboflowObjectDetector:
    def __init__(self, api_url, api_key, model_id):
        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self.model_id = model_id

    def detect_objects(self, image):
        result = self.client.infer(image, model_id=self.model_id)
        detections = []
        for item in result['predictions']:
            bbox = item['bbox']
            label = item['class']
            score = item['confidence']
            detections.append((bbox, label, score))
        return detections

    def draw_detections(self, image, detections):
        for bbox, label, score in detections:
            x1, y1, w, h = bbox
            cv2.rectangle(image, (int(x1), int(y1)), (int(x1+w), int(y1+h)), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image


def preprocess_frame(frame):
    processed_frame = cv2.resize(frame, (320, 180))
    return processed_frame


def get_transform():
    return T.Compose([T.ToTensor()])


def get_dataloader(data_dir, batch_size=4):
    dataset = CocoDetection(root=os.path.join(data_dir, 'images'),
                            annFile=os.path.join(data_dir, '_annotations.coco.json'),
                            transform=get_transform())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
