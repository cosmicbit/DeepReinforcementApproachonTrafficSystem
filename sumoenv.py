import traci
import gym
from gym import spaces
from traci import route
import os
import subprocess
import numpy as np


# Set the path to your SUMO installation
SUMO_HOME = "/home/ryoiki/sumo"  # SUMO directory path
SUMO_BINARY = SUMO_HOME + "/bin/sumo-gui"  # SUMO-GUI for visual simulation, use "sumo" for command-line version
SUMO_CONFIG = "Simulation Files/simulation.sumocfg"  # Your SUMO simulation configuration file
sumo_tools_path = os.path.join(SUMO_HOME, 'tools')  # Path to the SUMO tools directory.
net_file = 'Simulation Files/simple_network.net.xml' # Path to the SUMO network file.
routes_file = 'Simulation Files/randomRoutes.rou.xml' # Path to the SUMO routes file.
trips_file = 'Simulation Files/randomTrips.trips.xml' # Path to the SUMO routes file.


class SumoEnv(gym.Env):
    def __init__(self):
        super(SumoEnv, self).__init__()
        self.net_file = net_file
        self.route_file = routes_file
        self.sumoCmd = [SUMO_BINARY, "-c", SUMO_CONFIG]

        # Define the number of traffic lights and their possible states
        #self.n = len(traci.trafficlight.getIDList()) # Number of traffic lights
        #self.m = 3 # Example number of possible states for each traffic light (e.g., green, yellow, red)

        # Define observation and action spaces
        #self.state_size = self.n * 3 # Example state size calculation
        #self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)
        #self.action_space = spaces.MultiDiscrete([self.m for _ in range(self.n)]) # Example

    def reset(self,max_steps):
        # Generate random trips, convert to routes, and run the simulation
        self.generate_routes(max_steps)
        traci.start(self.sumoCmd)
        #initial_state = self._get_state()
        #return initial_state

    def step(self):
        traci.simulationStep()
        #self._perform_action(action)
        #next_state = self._get_state()
        #reward = self._compute_reward()
        #done = self._is_done()
        #info = {}
        #return next_state, reward, done

    def render(self, mode='human'):
        pass

    def close(self):
        traci.close()

    def seed(self, seed=None):
        self.simulation.set_seed(seed)

    def _get_state(self):  # Implementation to gather the current state from SUMO
        state = []
        traffic_lights = traci.trafficlight.getIDList()
        for tl in traffic_lights:
            state.append(traci.trafficlight.getRedYellowGreenState(tl))

        for tl in traffic_lights:
            lane_ids = traci.trafficlight.getControlledLanes(tl)
            vehicle_count = 0
            queue_length = 0

            for lane_id in lane_ids:
                vehicle_count += traci.lane.getLastStepVehicleNumber(lane_id)
                queue_length += traci.lane.getLastStepHaltingNumber(lane_id)

            state.extend([vehicle_count, queue_length])
        return np.array(state, dtype=np.float32)


    def _perform_action(self,action):
        traffic_lights = traci.trafficlight.getIDList()
        for i, tl in enumerate(traffic_lights):
            traci.trafficlight.setRedYellowGreenState(tl, action[i])


    def _compute_reward(self):  # Implementation to compute the reward
        total_waiting_time = sum(traci.lane.getWaitingTime(lane_id) for lane_id in traci.lane.getIDList())
        return -total_waiting_time

    def _is_done(self): # Implementation to check if the simulation is done
        pass

    def generate_routes(self, end_time, begin_time=0, trip_probability=2):
        """
        Generates random trips, converts them to routes, and runs the SUMO simulation.

        Parameters:
        - begin_time: Begin time for generating trips.
        - end_time: End time for generating trips.
        - trip_probability: Probability for generating a trip at each time step.

        Returns:
        - None
        """
        random_trips_script = os.path.join(sumo_tools_path, 'randomTrips.py')

        # Step 1: Generate random trips
        generate_trips_command = [
            'python', random_trips_script,
            '-n', net_file,
            '-o', trips_file,
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

"""
def add_vehicle(step):
    
    route = random.choice(["route0"])  # Assuming routes "route0" and "route1" are defined
    vehicle_id = "vehicle" + str(step)  # Unique ID for each vehicle
    traci.vehicle.add(vehicle_id, route, "car")  # Add vehicle with route and type

    # Set random speed for the vehicle
    speed = random.uniform(10, 30)  # Random speed between 10 and 30 m/s
    traci.vehicle.setSpeed(vehicle_id, speed)

def control_traffic_lights(step):
    
    if step % 10 == 0:  # Every 10th step, change the light
        # Switch traffic light at intersection 1 (you need to have traffic lights defined in your .tll.xml)
        for i in ["b", "c", "e",  "h", "i",  "l", "n", "o"]:
            traci.trafficlight.setRedYellowGreenState(i, "GGGGGGGGG")
        for i in ["f", "g", "j", "k"]:
            traci.trafficlight.setRedYellowGreenState(i, "GGGGGGGGGGGGGGGG")# Green on some phases, red on others
    #if step % 7 == 0:
        #for i in ["b", "c", "e",  "h", "i",  "l", "n", "o"]:
            #traci.trafficlight.setRedYellowGreenState(i, "rrrrrrrrr")
        #for i in ["f", "g", "j", "k"]:
            #traci.trafficlight.setRedYellowGreenState(i, "rrrrrrrrrrrrrrrr")

"""
