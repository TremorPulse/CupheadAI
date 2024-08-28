import torch
import torch.nn as nn
import tensorflow as tf
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CupheadCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super(CupheadCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def train_rl_model(env):
    # Ensure GPU utilization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy_kwargs = dict(
        features_extractor_class=CupheadCNN,
        features_extractor_kwargs=dict(features_dim=64),
    )

    # Adjust buffer size by lowering n_steps
    model = PPO("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                n_steps=1024,  # Reduce from 2048 to 1024 to lower memory usage
                batch_size=64, n_epochs=10,
                learning_rate=3e-4, clip_range=0.2, device=device)

    # Training the model
    model.learn(total_timesteps=50000)
    model.save('models/rl_model.zip')
    return model


def load_rl_model():
    model_path = 'models/rl_model.zip'
    try:
        custom_objects = {
            "policy_kwargs": {
                "features_extractor_class": CupheadCNN,
                "features_extractor_kwargs": {"features_dim": 64},
            },
            "clip_range": lambda _: 0.2,
            "lr_schedule": lambda _: 3e-4
        }
        return PPO.load(model_path, custom_objects=custom_objects, strict=False)
    except FileNotFoundError:
        print(f"Model not found at '{model_path}'.")
        raise
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        raise


# Ensure TensorFlow uses GPU if available (if you use TensorFlow)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
