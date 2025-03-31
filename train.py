
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
#from stable_baselines3.common.utils import linear_schedule
from sumo_env_one import SumoEnv
import time
import os
from gymnasium.wrappers import RecordEpisodeStatistics


# Define your custom environment
env = SumoEnv()
# Record episode statistics
env = RecordEpisodeStatistics(env)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)  # Normalize rewards and observations

# Checkpoint callback to save the model every 100K timesteps
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path="./models/", name_prefix="ppo_sumo_4")
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)
# Initialize the PPO model with optimized hyperparameters
agent = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cuda",
    tensorboard_log="./tensorboard_logs/",
    learning_rate=0.0003,  # Adaptive learning rate
    n_steps=2048,  # More frequent updates
    batch_size=128,  # Adjusted for stable learning
    ent_coef=0.01,  # Encourages exploration
    gamma=0.995,  # Helps long-term rewards
    clip_range=0.2, 
    clip_range_vf=0.2,  # Helps value function updates
)

print(f"Using device: {agent.device}")
print("Starting training...")

try:
    agent.learn(total_timesteps=1_000_000, callback=checkpoint_callback)
    env.close()
    agent.save(f'models/ppo_model_4{int(time.time())}')
    print("Final Model Saved.")
    stats_path = os.path.join(log_dir, f"vec_normalize_4.pkl")
    env.save(stats_path)
except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    agent.save(f'models/ppo_model_4{int(time.time())}')
    print("Model Saved.")
    stats_path = os.path.join(log_dir, "vec_normalize_4.pkl")
    env.save(stats_path)


'''
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from sumo_env_one import SumoEnv
import time
import os

# Number of parallel environments
NUM_ENVS = 4  # Adjust based on CPU cores

# Function to create environments
def make_env():
    return SumoEnv()

def train():
    # Create parallel environments
    env = SubprocVecEnv([lambda: make_env() for _ in range(NUM_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)  # Normalize rewards and observations

    # Checkpoint callback to save the model every 100K timesteps
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path="./models/", name_prefix="ppo_sumo")

    # Initialize PPO model with optimized hyperparameters
    agent = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        tensorboard_log="./tensorboard_logs/",
        learning_rate=0.0005,
        n_steps=1024 * NUM_ENVS,  
        batch_size=512,  
        ent_coef=0.01,  
        gamma=0.995,  
        clip_range=0.2,
        clip_range_vf=0.2,  
    )

    print(f"Using device: {agent.device}")
    print("Starting training...")

    try:
        agent.learn(total_timesteps=1_000_000, callback=checkpoint_callback)
        
        # Save final model and normalization stats
        model_path = f'models/ppo_model_final.zip'
        norm_path = f"models/vec_normalize_final.pkl"
        
        agent.save(model_path)
        env.save(norm_path)
        
        print(f"Final Model Saved at {model_path}")
        print(f"Normalization stats saved at {norm_path}")

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        agent.save("models/ppo_model_interrupted.zip")
        env.save("models/vec_normalize_interrupted.pkl")
        print("Model and normalization stats saved.")

if __name__ == '__main__':
    train()

'''

