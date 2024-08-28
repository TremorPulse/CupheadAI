import logging
import numpy as np
from object_detection import RoboflowObjectDetector
from reinforcement_learning import load_rl_model
from environment import CupheadEnv
import cv2


class CupheadAI:
    def __init__(self):
        self.rl_model = load_rl_model()
        self.object_detector = RoboflowObjectDetector(
            api_url="https://detect.roboflow.com",
            api_key="Z2Hxb8cSL6eWbRHhpnNM",
            model_id="rootpack/2"
        )
        self.env = CupheadEnv()

    def play(self):
        obs, _ = self.env.reset()
        if obs is None or not isinstance(obs, np.ndarray):
            logging.error("Invalid observation returned from reset")
            return

        done = False
        while not done:
            detections = self.object_detector.detect_objects(obs)
            action = self.get_action(obs, detections)
            obs, reward, done, _ = self.env.step(action)

            ai_view = self.object_detector.draw_detections(obs.copy(), detections)
            cv2.imshow("AI View", ai_view)
            cv2.waitKey(1)

    def get_action(self, obs, detections):
        detection_input = self.process_detections(detections)
        combined_input = np.concatenate([obs.flatten(), detection_input])
        action, _ = self.rl_model.predict(combined_input)
        return action

    def process_detections(self, detections):
        detection_vector = np.zeros(len(self.object_detector.classes) * 5)
        for box, label, score in detections:
            class_index = self.object_detector.classes.index(label)
            x1, y1, w, h = box
            detection_vector[class_index * 5:(class_index + 1) * 5] = [x1, y1, w, h, score]
        return detection_vector
