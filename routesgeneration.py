import random

# Node adjacency list
adjacency = {
    "a": ["b", "e"],
    "b": ["a", "c", "f"],
    "c": ["b", "d", "g"],
    "d": ["c", "h"],
    "e": ["a", "f", "i"],
    "f": ["e", "g", "b", "j"],
    "g": ["f", "h", "c", "k"],
    "h": ["g", "d", "l"],
    "i": ["e", "j", "m"],
    "j": ["i", "k", "f", "n"],
    "k": ["j", "l", "g", "o"],
    "l": ["k", "h", "p"],
    "m": ["i", "n"],
    "n": ["m", "o", "j"],
    "o": ["n", "p", "k"],
    "p": ["o", "l"]
}
edges = {
    ("a", "b"): "ab", ("b", "a"): "ba",
    ("b", "c"): "bc", ("c", "b"): "cb",
    ("c", "d"): "cd", ("d", "c"): "dc",
    ("e", "f"): "ef", ("f", "e"): "fe",
    ("f", "g"): "fg", ("g", "f"): "gf",
    ("g", "h"): "gh", ("h", "g"): "hg",
    ("i", "j"): "ij", ("j", "i"): "ji",
    ("j", "k"): "jk", ("k", "j"): "kj",
    ("k", "l"): "kl", ("l", "k"): "lk",
    ("m", "n"): "mn", ("n", "m"): "nm",
    ("n", "o"): "no", ("o", "n"): "on",
    ("o", "p"): "op", ("p", "o"): "po",
    ("a", "e"): "ae", ("e", "a"): "ea",
    ("b", "f"): "bf", ("f", "b"): "fb",
    ("c", "g"): "cg", ("g", "c"): "gc",
    ("d", "h"): "dh", ("h", "d"): "hd",
    ("e", "i"): "ei", ("i", "e"): "ie",
    ("f", "j"): "fj", ("j", "f"): "jf",
    ("g", "k"): "gk", ("k", "g"): "kg",
    ("h", "l"): "hl", ("l", "h"): "lh",
    ("i", "m"): "im", ("m", "i"): "mi",
    ("j", "n"): "jn", ("n", "j"): "nj",
    ("k", "o"): "ko", ("o", "k"): "ok",
    ("l", "p"): "lp", ("p", "l"): "pl"
}

def nodes_to_edges(route, edge_map):
    edges = []
    for i in range(len(route) - 1):
        edge = edge_map.get((route[i], route[i+1]))
        if edge:
            edges.append(edge)
    return edges


import random


# Function to generate routes based on nodes and convert to edges
def generate_random_routes(num_vehicles, num_routes):
    route_file = open("simple_routes.rou.xml", "w")
    route_file.write('<routes>\n')
    route_file.write('<vType id="type1" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89"/>\n')

    routes = []
    for i in range(num_routes):
        node_route = []
        current_node = random.choice(list(adjacency.keys()))
        node_route.append(current_node)
        while len(node_route) < random.randint(2, 5):
            next_nodes = adjacency.get(current_node, [])
            if not next_nodes:
                break
            current_node = random.choice(next_nodes)
            if current_node not in node_route:  # Prevent revisiting nodes in the same route
                node_route.append(current_node)

        edge_route = [edges[(node_route[j], node_route[j + 1])] for j in range(len(node_route) - 1)]
        edge_route_str = " ".join(edge_route)

        route_file.write(f'    <route id="route{i}" edges="{edge_route_str}"/>\n')

    for i in range(num_vehicles):
        route_id = random.randint(0, num_routes - 1)
        depart_time = random.randint(0, 100)
        route_file.write(f'    <vehicle id="veh{i}" type="type1" route="route{route_id}" depart="{depart_time}"/>\n')

    route_file.write('</routes>\n')
    route_file.close()


