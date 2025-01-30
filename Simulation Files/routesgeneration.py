import random

# Updated Node adjacency list for the 3x3 grid
adjacency = {
    "a": ["b", "d"],
    "b": ["a", "c", "e"],
    "c": ["b", "f"],
    "d": ["a", "e", "g"],
    "e": ["b", "d", "f", "h"],
    "f": ["c", "e", "i"],
    "g": ["d", "h"],
    "h": ["e", "g", "i"],
    "i": ["f", "h"]
}

# Updated edges mapping for the 3x3 grid
edges = {
    ("a", "b"): "ab", ("b", "a"): "ba",
    ("b", "c"): "bc", ("c", "b"): "cb",
    ("a", "d"): "ad", ("d", "a"): "da",
    ("b", "e"): "be", ("e", "b"): "eb",
    ("c", "f"): "cf", ("f", "c"): "fc",
    ("d", "e"): "de", ("e", "d"): "ed",
    ("e", "f"): "ef", ("f", "e"): "fe",
    ("d", "g"): "dg", ("g", "d"): "gd",
    ("e", "h"): "eh", ("h", "e"): "he",
    ("f", "i"): "fi", ("i", "f"): "if",
    ("g", "h"): "gh", ("h", "g"): "hg",
    ("h", "i"): "hi", ("i", "h"): "ih"
}

# Function to generate routes based on nodes and convert to edges
def generate_random_routes(num_vehicles, num_routes):
    with open("random_routes.rou.xml", "w") as route_file:
        route_file.write('<routes>\n')
        route_file.write('    <vType id="type1" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89"/>\n\n')

        # Generate routes
        for i in range(num_routes):
            node_route = []
            current_node = random.choice(list(adjacency.keys()))
            node_route.append(current_node)
            route_length = random.randint(5, 10)  # Adjusted route length for the smaller grid

            while len(node_route) < route_length:
                next_nodes = adjacency.get(current_node, [])
                valid_next_nodes = [node for node in next_nodes if node not in node_route]
                if not valid_next_nodes:
                    break  # No unvisited adjacent nodes, end this route
                current_node = random.choice(valid_next_nodes)
                node_route.append(current_node)

            # Convert node route to edge route
            try:
                edge_route = [edges[(node_route[j], node_route[j + 1])] for j in range(len(node_route) - 1)]
            except KeyError as e:
                print(f"Edge not found for nodes {e}. Skipping route.")
                continue  # Skip this route if an edge is missing
            edge_route_str = " ".join(edge_route)
            route_file.write(f'    <route id="route{i}" edges="{edge_route_str}"/>\n')

        route_file.write('\n')

        # Generate vehicles
        vehicles = []
        for i in range(num_vehicles):
            route_id = random.randint(0, num_routes - 1)
            depart_time = random.uniform(0, 500)  # Adjust depart time as needed
            vehicles.append((i, route_id, depart_time))
        # Sort vehicles by departure time
        vehicles.sort(key=lambda x: x[2])

        for vehicle in vehicles:
            route_file.write(f'    <vehicle id="veh{vehicle[0]}" type="type1" route="route{vehicle[1]}" depart="{vehicle[2]:.2f}"/>\n')

        route_file.write('</routes>\n')

# Example usage:
if __name__ == "__main__":
    generate_random_routes(num_vehicles=50, num_routes=20)