import traci
import os
import subprocess
import numpy as np
from gym import Env
from gym.spaces import MultiDiscrete, Box
from generator import TrafficGenerator
# Set the path to your SUMO installation
SUMO_HOME = "/usr/bin/sumo"  # SUMO directory path
SUMO_BINARY = "sumo-gui"  # SUMO-GUI for visual simulation, use "sumo" for command-line version
SUMO_CONFIG = "Simulation Files/simulation.sumocfg"  # Your SUMO simulation configuration file
net_file = 'Simulation Files/grid4by4.net.xml' # Path to the SUMO network file.


class SumoEnv(Env):
    def __init__(self):
        super(SumoEnv, self).__init__()
        self.numIntersections = None
        self.intersections = None
        self.net_file = net_file
        #self.route_file = routes_file
        self.sumoCmd=[SUMO_BINARY, "-c", SUMO_CONFIG]
        
        self.action_space = MultiDiscrete([16]*12)  # Observations: Number of vehicles, queue lengths, etc., for each intersection
        self.observation_space = Box(low=0, high=np.inf, shape=(24,), dtype=np.float32)
        self.stepCount=0

    def reset(self):
        TrafficGen=TrafficGenerator(0,0)
        # Generate random trips, convert to routes, and run the simulation
        TrafficGen.generate_routes(net_file, 100)
        traci.start(self.sumoCmd)
        self.intersections =traci.trafficlight.getIDList()
        self.numIntersections = len(self.intersections)
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
            num_of_vehicles = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in traci.trafficlight.getControlledLanes(i))
            current_phase = traci.trafficlight.getRedYellowGreenState(i)
            encoded_phase = self.encode_phase(current_phase)
            state.extend([num_of_vehicles, encoded_phase])

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

