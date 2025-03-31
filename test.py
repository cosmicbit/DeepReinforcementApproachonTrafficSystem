from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sumo_env_one import SumoEnv

'''
ppo_sumo_400000_steps.zip is the trained model file.
it is the only good model file I have.

ppo_model_41743298875 much better


'''
# Load trained model
agent = PPO.load("models/ppo_model_41743298875.zip")

# Create environment
env = DummyVecEnv([lambda: SumoEnv()])

# Load VecNormalize if available, otherwise skip
try:
    env = VecNormalize.load("logs/vec_normalize_4.pkl", env)
    env.training = False  # Disable updates to running mean/std
    env.norm_reward = False
    print("Loaded VecNormalize settings.")
except FileNotFoundError:
    print("VecNormalize file not found. Proceeding without it.")


'''
# Run test
obs = env.reset()
for _ in range(1000):
    action, _states = agent.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
'''

obs = env.reset()
done = False

while not done:
    action, _states = agent.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
print("Evaluation completed.")
