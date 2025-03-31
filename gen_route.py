import traci
import os
import subprocess
import numpy as np
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import shutil
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

class Routes:
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

if __name__ == "__main__":
    routes = Routes()
    routes.generate_routes(3600)  
    print("Routes generated successfully.")