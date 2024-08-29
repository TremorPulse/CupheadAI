import cv2

def preprocess_frame(frame):
    processed_frame = cv2.resize(frame, (320, 180))
    return processed_frame


def action_to_key(action):
    keys = ['right', 'space', 'z', 'x']
    return keys[action]
