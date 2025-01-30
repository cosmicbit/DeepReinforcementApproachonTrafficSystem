from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sumoenv import SumoEnv
import time

# Define your custom environment
env = DummyVecEnv([lambda : SumoEnv()])
env = VecNormalize(env, norm_reward=True)

# Initialize the PPO model
agent = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cuda",
    learning_rate=3e-4,  # Example, experiment with this
    n_steps=2048,  # Adjust to match environment requirements
    batch_size=64   # Adjust based on complexity
)
print(agent.device)
print("Starting training...")

try:
    agent.learn(total_timesteps=4500)  # Adjust as needed for your setup
except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    agent.save("models/ppo_model_")
    print("Model Saved..")
finally:
    env.close()
    # Save models
    agent.save(f"models/ppo_model_")
    print("Model Saved..")
