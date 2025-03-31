from sumoenv import SumoEnv

#num_episodes = 10000
max_steps = 500
outfile = open(f'log2.txt', 'a')

env = SumoEnv()
state = env.reset()

for step in range(max_steps):
    print("step ",step)
    state, reward, done=env.step()
