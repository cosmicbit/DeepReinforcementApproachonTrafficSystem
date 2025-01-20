from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from sumoenv import SumoEnv

# Define your custom environment
env = SumoEnv()

# Load the trained model
model = PPO.load("models/ppo_model_.zip")

# Evaluate the model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()