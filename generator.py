import numpy as np
import subprocess


routes_file = 'Simulation Files/randomRoutes.rou.xml' # Path to the SUMO routes file.
trips_file = 'Simulation Files/randomTrips.trips.xml' # Path to the SUMO routes file.

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps

    def generate_routes(self, net_file, end_time, begin_time=0, trip_probability=2):
        random_trips_script = 'randomTrips.py'

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
