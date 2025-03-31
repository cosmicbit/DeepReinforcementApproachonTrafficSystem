import traci
import os
import subprocess
import numpy as np
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import matplotlib.pyplot as plt

process_id = os.getpid()
SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"  # Use raw string to handle backslashes
SUMO_BINARY = os.path.join(SUMO_HOME, "bin", "sumo")  # Use os.path.join for cross-platform compatibility
SUMO_CONFIG = r"sumo_network\simulation.sumocfg"  # Use raw string for Windows path
sumo_tools_path = os.path.join(SUMO_HOME, 'tools')  # Path to the SUMO tools directory
net_file = r'sumo_network\one_inter2.net.xml'  # Use raw string for Windows path
routes_file = r'sumo_network\randomRoutes.rou.xml'
trips_file = r'sumo_network\randomTrips.trips.xml'
config_file = r'sumo_network\simulation.sumocfg'

class SumoEnv(gym.Env):
    def __init__(self):
        super(SumoEnv, self).__init__()
        self.net_file = net_file
        self.route_file = routes_file
        self.sumoCmd=[SUMO_BINARY, "-c", SUMO_CONFIG]
        self.rewards_per_step = []
        self.waiting_times_per_step = []
        self.total_waiting_time = 0.0
        self.max_duration = 0

        #ESWN//RSL
        self.valid_phases = [
            ["GrrrGG","rrGGGr","GGGrrr"],
            ["GrrrGG","rrGGGr","GGGrrr"],
            # E_W green      # S green      # E green      # W green       # N green     # protected_left_E_W     # N_S green   # protected_left_N_S
            ["GGrrrrGGrrrr","rrrGGGrrrrrr","GGGrrrrrrrrr","rrrrrrGGGrrr","rrrrrrrrrGGG",  "rrGGrrrrGGrr",       "rrrGGrrrrGGr" , "GrrrrGGrrrrG"],
            ["GGGrrr","GrrrGG","rrGGGr"],
            ["rrGGGr","GGGrrr","GrrrGG"]
        ]

        self.phase_duration_combos = { 
            'b': [ (phase_idx, dur) for phase_idx in range(3) for dur in [36,24,12] ],
            'd': [ (phase_idx, dur) for phase_idx in range(3) for dur in [36,24,12] ],
            'e': [ (phase_idx, dur) for phase_idx in range(8) for dur in [36,24,12] ],
            'f': [ (phase_idx, dur) for phase_idx in range(3) for dur in [36,24,12] ],
            'h': [ (phase_idx, dur) for phase_idx in range(3) for dur in [36,24,12] ]
        }

        self.numIntersections=1
        self.intersections = {
            
            'e':
                {
                    'N': ['be_0','RSL'],
                    'W': ['de_0','RSL'],
                    'S': ['he_0','RSL'],
                    'E': ['fe_0','RSL']
                }
        }

        self.maxLanes = 4

        self.action_space = MultiDiscrete([24]) 
        self.observation_space = Box(low=0, high=np.inf, shape=( self.numIntersections * self.maxLanes , ), dtype=np.float32)
        self.stepCount = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Ensure SUMO is properly closed before restarting
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass  # Connection was not open
        except AttributeError:
            pass

        print("***** New Episode Started *****")
        #self.generate_routes(10000)

        traci.start(self.sumoCmd)

        observation = self._get_state()
        info = {}

        return observation, info  # Gymnasium requires (obs, info)

    def step(self, actions):
        for idx, action in enumerate(actions):
            self._apply_action(idx, action)

        try:
            for _ in range(self.max_duration):
                traci.simulationStep()
        except traci.exceptions.FatalTraCIError as e:
            print(f"Error during simulation step: {e}")
            self.close()
            raise

        self.max_duration = 0
        observation = self._get_state()
        reward = self.calculate_reward2()
        self.rewards_per_step.append(reward)  # Store reward

        # Track average waiting time
        self.waiting_times_per_step.append(self.total_waiting_time/4)

        terminated = self._is_done()
        truncated = False  # No external truncation condition in this case
        info = {}

        self.stepCount += 1
        #print(f'Step={self.stepCount}, Reward={reward}, Done={terminated}')

        if terminated:
            self.close()

        return observation, reward, terminated, truncated, info  # Gymnasium format

    def _apply_action(self, intersection_id, action):

        # Get the phase duration combo for the current intersection
        model_action = self.phase_duration_combos['e'][action]
        model_phase, model_duration = model_action
        print(f'model_phase={model_phase}, model_duration={model_duration}')
        self.max_duration = max(model_duration, self.max_duration)

        #model_phase = self.valid_phases[intersection_id][action]
        #print(self.intersections.keys())
        #print(f'Intersection ID={self.sample_inter[intersection_id]}, Model Phase={self.valid_phases[intersection_id][model_phase]}')

        #print(f'tls_programs = {traci.trafficlight.getAllProgramLogics(self.sample_inter[intersection_id])}')
        print(self.valid_phases[2][model_phase])
        traci.trafficlight.setRedYellowGreenState('e', self.valid_phases[2][model_phase])


    def _get_state(self):  
        state=[]
        #for intersection_id in self.intersections.keys():

        current_phase = traci.trafficlight.getRedYellowGreenState('e')
        encoded_phase = self.encode_traffic_code( 'e' , current_phase)

        state.extend(encoded_phase)
        return np.array(state)

    def encode_traffic_code(self, intersection_id, phase):

        phase = phase.lower()
        intersection_config = []
        phase_index = 0  

        for direction in ['N', 'E', 'S', 'W']:

            if self.intersections[intersection_id][direction] == -1:
                # Append placeholders for missing lanes
                intersection_config.extend([0.0, 0.0])  # Assuming normalized values are 0.0
                continue

            lane_info = self.intersections[intersection_id][direction]
            #print(f'Intersection ID={intersection_id}, Direction={direction}, Lane Info={lane_info}')
            lane_id, possible_movements = lane_info

            # Get queue length and normalize
            queue_length = traci.lane.getLastStepVehicleNumber(lane_id)
            #max_queue_length = 20  # Adjust based on expected max
            #normalized_queue_length = min(queue_length / max_queue_length, 1.0)

            # Get the traffic light states for this lane
            num_signals = len(possible_movements)
            lane_phase = phase[phase_index:phase_index + num_signals]
            phase_index += num_signals  # Update for next iteration

            current_traffic_state = ""
            for idx, light in enumerate(lane_phase):
                if light in ('g', 'G'):
                    current_traffic_state += possible_movements[idx]

            # Map the current traffic state to the movement code
            #traffic_code = self.movement_mapping.get(current_traffic_state, 0)
            #max_traffic_code = len(self.movement_combinations) - 1
            #normalized_traffic_code = traffic_code / max_traffic_code

            # Append normalized queue_length and traffic_code to intersection_config
            intersection_config.extend([queue_length])

        return intersection_config        


    def calculate_shared_reward(self, state):
        # Weight factors (tune these to adjust priorities)
        print(f'Stateeeeee={state}')
        alpha = 1.0 # Weight for waiting time
        beta = 1.0 # Weight for congestion (density)


        # Option A: Calculate total waiting time directly from simulation data
        total_waiting_time = 0.0
        #for intersection_id in self.intersections.keys():
            # Get the lanes controlled by the traffic light at this intersection.
        controlled_lanes = traci.trafficlight.getControlledLanes('e')
        for lane in controlled_lanes:
            total_waiting_time += traci.lane.getWaitingTime(lane)

        # Option B: Calculate congestion using the observation vector.
        # Our observation vector is defined so that for each lane we have two values: [normalized_density, normalized_traffic_code].
        # We only use the density (the first value) to estimate congestion.
        total_density = 0.0
        # Iterate over the state vector in steps of 2 (density is at index 0, 2, 4, …)
        for i in range(0, len(state), 2):
            total_density += state[i]

        # The composite reward is a weighted negative sum. Lower waiting time and lower density give a higher (less negative) reward.
        composite_reward = - (alpha * total_waiting_time + beta * total_density)

        return composite_reward 

    def _is_done(self): # Implementation to check if the simulation is done
        # Define when the episode is done
        return traci.simulation.getMinExpectedNumber() <= 0

    def plot_rewards(self):
        plt.figure(figsize=(12, 5))  # Set figure size

        # First subplot: Reward per step
        plt.subplot(1, 2, 1)  # Define the first subplot
        plt.plot(self.rewards_per_step, label="Reward per step")
        plt.xlabel("Time Steps")
        plt.ylabel("Reward")
        plt.title("Reward Progression Over an Episode")
        plt.legend()

        # Second subplot: Average Waiting Time per step
        plt.subplot(1, 2, 2)  # Define the second subplot
        plt.plot(self.waiting_times_per_step, label="Avg Waiting Time", color="red")
        plt.xlabel("Time Steps")
        plt.ylabel("Avg Waiting Time (s)")
        plt.title("Average Waiting Time Over Episode")
        plt.legend()

        plt.tight_layout()  # Adjust layout to prevent overlapping
        plt.show()

    def close(self):
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass
        print("***** Episode Ended *****")
        self.plot_rewards()
        average_reward = np.mean(self.rewards_per_step)
        print(f"Average Reward: {average_reward}")

        print(f"Average Waiting Time: {np.mean(self.waiting_times_per_step)}")


    '''
    def generate_routes(self, end_time, begin_time=0, trip_probability=10):
        random_trips_script = os.path.join(sumo_tools_path, 'randomTrips.py')

        # Step 1: Generate realistic random trips
        generate_trips_command = [
            'python', random_trips_script,
            '-n', net_file,
            '-o', trips_file,
            '--seed', str(np.random.randint(0, 10000)),
            '-b', str(begin_time),
            '-e', str(end_time),
            '--period', '5',  # Balanced vehicle departure timing
            '--fringe-factor', '5',  # Favor edges, reducing congestion
            '--validate'  # Ensure trip validity
        ]

        # Execute the command to generate random trips
        subprocess.run(generate_trips_command, check=True)

        # Step 2: Convert trips to realistic routes
        duarouter_command = [
            'duarouter',
            '-n', net_file,
            '-t', trips_file,
            '-o', routes_file
        ]

        # Execute the command to convert trips to routes
        subprocess.run(duarouter_command, check=True) 
    '''
    def generate_routes(self, end_time, begin_time=0, trip_probability=10):
        random_trips_script = os.path.join(sumo_tools_path, 'randomTrips.py')

        # Adjust period for varying congestion levels (Lower = higher congestion)
        period = np.random.choice([3, 5, 8], p=[0.4, 0.4, 0.2])  

        # Modify fringe factor (Lower values concentrate traffic in central areas)
        fringe_factor = np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])

        generate_trips_command = [
            'python', random_trips_script,
            '-n', net_file,
            '-o', trips_file,
            '--seed', str(np.random.randint(0, 10000)),
            '-b', str(begin_time),
            '-e', str(end_time),
            '--period', str(period),  
            '--fringe-factor', str(fringe_factor),  
            '--validate'
        ]

        subprocess.run(generate_trips_command, check=True)

        duarouter_command = [
            'duarouter',
            '-n', net_file,
            '-t', trips_file,
            '-o', routes_file
        ]

        subprocess.run(duarouter_command, check=True)


    def calculate_reward(self):
        """
        Production-grade composite reward that prioritizes reducing congestion and waiting times.

        Collapse
        It computes:
        Reward = – [ α * (NewTotalCongestion – OldTotalCongestion) 
                        + β * (NewWaitingTime – OldWaitingTime) 
                        + γ * FairnessPenalty ]

        Where:
        - NewTotalCongestion and NewWaitingTime are computed in this step using SUMO (via traci).
        - FairnessPenalty is computed as the difference between the maximum and average waiting time across lanes.
        """

        # 1) Compute current totals from SUMO
        new_total_congestion = 0.0    # e.g., sum of number of vehicles waiting in lanes
        new_total_waiting_time = 0.0  # e.g., sum of waiting times from each lane
        waiting_times = []          # List of waiting times for detecting unevenness (fairness)

        # Loop over each intersection defined in your environment
        for intersection_id in self.intersections.keys():
            # Get controlled lanes for the intersection.
            controlled_lanes = traci.trafficlight.getControlledLanes(intersection_id)
            for lane in controlled_lanes:
                vehicles = traci.lane.getLastStepVehicleNumber(lane)
                new_total_congestion += vehicles

                wt = traci.lane.getWaitingTime(lane)
                new_total_waiting_time += wt
                waiting_times.append(wt)

        # 2) Compute Fairness Penalty
        if waiting_times:
            max_waiting = max(waiting_times)
            avg_waiting = np.mean(waiting_times)
            fairness_penalty = max_waiting - avg_waiting
        else:
            fairness_penalty = 0.0

        # 3) If "old" values are not set, initialize them with current values
        if not hasattr(self, "last_total_congestion"):
            self.last_total_congestion = new_total_congestion
        if not hasattr(self, "last_total_waiting_time"):
            self.last_total_waiting_time = new_total_waiting_time

        # 4) Compute the deltas (improvements if negative)
        congestion_delta = new_total_congestion - self.last_total_congestion
        waiting_delta = new_total_waiting_time - self.last_total_waiting_time

        # 5) Compute reward as a weighted negative sum:
        # A decrease in congestion (i.e., a negative delta) increases the reward.
        reward = - (self.alpha * congestion_delta  + self.beta * waiting_delta  + self.gamma * fairness_penalty)

        # 6) Update the stored baseline values for the next step.
        self.last_total_congestion = new_total_congestion
        self.last_total_waiting_time = new_total_waiting_time

        return reward


    # main reward function
    def calculate_reward2(self):
        """
        Calculates lane-wise rewards and ensures high-congestion lanes get priority.
        """
        
        total_reward = 0.0
        lane_rewards = {}  # Store rewards per lane
        self.total_waiting_time = 0.0

        # Parameters (Tunable)
        waiting_threshold = getattr(self, "waiting_threshold", 10.0)
        lambda_bonus = getattr(self, "lambda_bonus", 0.5)
        congestion_threshold = 10  # If queue exceeds this, increase priority
        high_congestion_weight = 2  # Boost penalty for highly congested lanes

        for intersection_id in self.intersections.keys():
            controlled_lanes = traci.trafficlight.getControlledLanes(intersection_id)
            for lane in controlled_lanes:
                q = traci.lane.getLastStepVehicleNumber(lane)
                w = traci.lane.getWaitingTime(lane)
                self.total_waiting_time += w  # Accumulate waiting time for all lanes

                print(f"Lane={lane}, Queue Length={q}, Waiting Time={w}")
                # Adjust priority for highly congested lanes
                congestion_penalty = (q + w) * (high_congestion_weight if q > congestion_threshold else 1)

                # Bonus for excessive waiting time
                bonus = lambda_bonus * max(0, w - waiting_threshold)

                lane_reward = -congestion_penalty + bonus  # Negative penalty means we want to minimize it
                lane_rewards[lane] = lane_reward
                total_reward += lane_reward  # Sum instead of averaging

        return total_reward  # Return sum instead of average


    


