import traci
import os
import subprocess
import numpy as np
from gym import Env
from gym.spaces import MultiDiscrete, Box

# Set the path to your SUMO installation
SUMO_HOME = "/home/ryoiki/sumo"  # SUMO directory path
SUMO_BINARY = SUMO_HOME + "/bin/sumo-gui"  # SUMO-GUI for visual simulation, use "sumo" for command-line version
SUMO_CONFIG = "Simulation Files/simulation.sumocfg"  # Your SUMO simulation configuration file
sumo_tools_path = os.path.join(SUMO_HOME, 'tools')  # Path to the SUMO tools directory.
net_file = 'Simulation Files/simple_network.net.xml' # Path to the SUMO network file.
routes_file = 'Simulation Files/randomRoutes.rou.xml' # Path to the SUMO routes file.
trips_file = 'Simulation Files/randomTrips.trips.xml' # Path to the SUMO routes file.


class SumoEnv(Env):
    def __init__(self):
        super(SumoEnv, self).__init__()
        self.net_file = net_file
        self.route_file = routes_file
        self.sumoCmd=[SUMO_BINARY, "-c", SUMO_CONFIG]
        self.numIntersections=12
        self.intersections = ['b', 'c', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'n', 'o']
        self.action_space = MultiDiscrete([16]*12)  # Observations: Number of vehicles, queue lengths, etc., for each intersection
        self.observation_space = Box(low=0, high=np.inf, shape=(36,), dtype=np.float32)
        self.stepCount=0

    def reset(self):
        # Generate random trips, convert to routes, and run the simulation
        self.generate_routes(10000)
        traci.start(self.sumoCmd)
        return self._get_state()

    def step(self, actions):
        #print("************ step ",self.stepCount,"************")
        #print("actions=",actions)
        for idx, action in enumerate(actions):
            self._apply_action(idx, action)
        traci.simulationStep()
        state = self._get_state()
        rewards = self._calculate_shared_reward(state)
        done = self._is_done()
        #print("state=", state)
        self.stepCount = self.stepCount + 1
        return state, rewards, done, {}

    def _get_state(self):  # Implementation to gather the current state from SUMO
        state=[]
        for i in self.intersections:
            queue_length = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in traci.trafficlight.getControlledLanes(i))
            waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(i))
            current_phase = traci.trafficlight.getRedYellowGreenState(i)
            encoded_phase = self.encode_phase(current_phase)
            state.extend([queue_length, waiting_time, encoded_phase])

        return np.array(state)

    def _apply_action(self, intersection_id, action):
        if action==0:   # 0 for No action
            return
        if intersection_id in [3, 4, 7, 8]:
            decoded_action=self.decode_action_4way(action)
        else:
            decoded_action=self.decode_action_3way(action)
        traci.trafficlight.setRedYellowGreenState(self.intersections[intersection_id], decoded_action)

    def decode_action_3way(self, action):
        return format(action, '04b').replace('1', 'ggg').replace('0', 'rrr')

    def decode_action_4way(self, action):
        return format(action, '04b').replace('1', 'gggg').replace('0', 'rrrr')

    def encode_phase(self, phase):
        phase = phase.lower()
        if len(phase) in [9,12] :
            encoded_phase=phase.replace('ggg','1').replace('rrr','0')
        elif len(phase)==16:
            encoded_phase = phase.replace('gggg', '1').replace('rrrr', '0')
        return int(encoded_phase,2)

    def _calculate_shared_reward(self, state):
        sum=0
        for i in range(len(state)):
            if (i+1)%3 != 0:
                sum = sum + state[i]
        return -sum  # Shared reward for all agents

    def _is_done(self): # Implementation to check if the simulation is done
        # Define when the episode is done
        return traci.simulation.getMinExpectedNumber() <= 0

    def close(self):
        traci.close()

    def generate_routes(self, end_time, begin_time=0, trip_probability=2):
        random_trips_script = os.path.join(sumo_tools_path, 'randomTrips.py')

        # Step 1: Generate random trips
        generate_trips_command = [
            'python', random_trips_script,
            '-n', net_file,
            '-o', trips_file,
            '--seed', str(np.random.randint(0, 10000)),
            '-b', str(begin_time),
            '-e', str(end_time),
            '-p', str(trip_probability)
        ]

        # Execute the command to generate random trips
        subprocess.run(generate_trips_command, check=True)

        # Step 2: Convert trips to routes
        duarouter_command = [
            'duarouter',
            '-n', net_file,
            '-t', trips_file,
            '-o', routes_file
        ]

        # Execute the command to convert trips to routes
        subprocess.run(duarouter_command, check=True)