import random
import routesgeneration
import traci

# Set the path to your SUMO installation
SUMO_HOME = "/home/ryoiki/sumo"  # Replace with your SUMO directory path
SUMO_BINARY = SUMO_HOME + "/bin/sumo-gui"  # SUMO-GUI for visual simulation, use "sumo" for command-line version
SUMO_CONFIG = "simulation.sumocfg"  # Your SUMO simulation configuration file


def run_simulation(veh_ids,detected_vehicles):
    # Define adjacency relationships
    routesgeneration.generate_random_routes(1000, 100)
    # Start the SUMO simulation
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

    # Simulation loop
    step = 0
    while step < 1000:  # Run for 1000 steps
        traci.simulationStep()

        # Add vehicles at random intervals
        #if random.random() < 0.1:  # 10% chance to add a vehicle at each step
            #veh_ids=add_vehicle(step,veh_ids)

        #print(traci.inductionloop.getLastStepVehicleIDs("loop1"))
        #detected_vehicles += traci.inductionloop.getLastStepVehicleIDs("loop1")

        # Control traffic lights (example: switch the light at intersection 1)
        control_traffic_lights(step)

        step += 1
    print(f"Vehicles added: {veh_ids}")
    print(f"Vehicles detected: {detected_vehicles}")
    # Close the simulation
    traci.close()

def add_vehicle(step,veh_ids):
    """
    Add a vehicle to the simulation at a random route
    """

    route = random.choice(["route0", "route1", "route2", "route3"])  # Assuming routes "route0" and "route1" are defined
    vehicle_id = "vehicle" + str(step)  # Unique ID for each vehicle
    traci.vehicle.add(vehicle_id, route, "car")  # Add vehicle with route and type
    #if route in ["route0", "route2"]:
    #veh_ids += (vehicle_id,)

    # Set random speed for the vehicle
    speed = random.uniform(10, 30)  # Random speed between 10 and 30 m/s
    traci.vehicle.setSpeed(vehicle_id, speed)
    return veh_ids

def control_traffic_lights(step):
    """
    Control traffic lights based on the simulation step (simple logic)
    """
    point3 = random.choice(["b", "c", "e",  "h", "i",  "l", "n", "o"])
    point4 = random.choice(["f", "g", "j", "k"])
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

if __name__ == "__main__":
    veh_ids = ()
    detected_vehicles = ()
    # Run the simulation
    run_simulation(veh_ids,detected_vehicles)

