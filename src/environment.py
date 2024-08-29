import time
import cv2
import numpy as np
import mss
import gymnasium as gym
from pynput.keyboard import Controller
from pynput.keyboard import Key

class CupheadEnv(gym.Env):
    def __init__(self):
        super(CupheadEnv, self).__init__()
        self.keyboard = Controller()
        self.sct = mss.mss()
        self.monitor = {"top": 0, "left": 0, "width": 1280, "height": 720}
        self.previous_health = 3  # 3 health points indicated by cards on bottom left

        self.action_space = gym.spaces.Discrete(7)  # 7 possible key actions
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(180, 320, 3), dtype=np.uint8)

    def get_screen(self):
        screenshot = np.array(self.sct.grab(self.monitor))
        if screenshot.shape[2] == 4:  # remove alpha channel if present
            screenshot = screenshot[:, :, :3]

        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGBA2RGB)
        return self.preprocess_frame(screenshot)

    def preprocess_frame(self, frame):
        processed_frame = cv2.resize(frame, (320, 180))
        if processed_frame is None or processed_frame.size == 0:
            raise ValueError("Processed frame is invalid. Check the resize operation.")
        print(f"Processed frame: shape={processed_frame.shape}")
        return processed_frame

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

        self.previous_health = 3  # reset health to the starting value
        obs = self.get_screen()
        return obs, {}

    def take_action(self, action):
        key_map = {
            0: Key.right,
            1: Key.left,
            2: Key.up,
            3: Key.down,
            4: Key.space,  # jumping or shooting
            5: 'z',        # shooting or another action
            6: 'x'
        }

        key = key_map.get(action)
        if key:
            self.keyboard.press(key)
            time.sleep(0.1)
            self.keyboard.release(key)

    def get_reward(self):
        current_health = self.get_health_from_screen()
        reward = (current_health - self.previous_health) * 0.5
        self.previous_health = current_health
        return reward

    def step(self, action):
        self.take_action(action)
        obs = self.get_screen()
        reward = self.get_reward()
        done = self.is_done()
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, {}

    def render(self, mode='human'):
        cv2.imshow("Cuphead", self.get_screen())
        cv2.waitKey(1)

    def is_done(self):
        return self.get_health_from_screen() <= 0

    def get_health_from_screen(self):
        frame = self.get_screen()
        health_region = frame[-40:, :40]  # attempt to capture bottom left corner for health

        if health_region.size == 0:
            raise ValueError("Health region is empty. Check the region coordinates.")

        health_cards = self.detect_health_cards(health_region)
        return len(health_cards)

    def detect_health_cards(self, health_region):
        gray = cv2.cvtColor(health_region, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
