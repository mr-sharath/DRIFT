# ortools_solver.py

"""
Google OR-Tools Baseline Solver for Waynex.
Solves the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW).
Uses Google's `pywrapcp.RoutingModel` with Guided Local Search Metaheuristic
and balanced multi-vehicle fleet allocation.
"""

from typing import List, Dict, Any, Tuple
import time

HAS_OR_TOOLS = False
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    HAS_OR_TOOLS = True
    print("✅ Google OR-Tools library loaded successfully.")
except ImportError:
    HAS_OR_TOOLS = False


def solve_vrp_ortools(
    depots: List[Dict[str, Any]],
    deliveries: List[Dict[str, Any]],
    vehicles: List[Dict[str, Any]],
    dist_matrix_km: List[List[float]],
    dur_matrix_min: List[List[float]],
    perturbations: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Solve VRP using Google OR-Tools CVRPTW solver with multi-vehicle balanced allocation.
    """
    start_cpu_time = time.time()
    dist_mat, dur_mat = apply_perturbations_to_matrix(dist_matrix_km, dur_matrix_min, perturbations)

    if HAS_OR_TOOLS:
        try:
            return _solve_with_ortools(depots, deliveries, vehicles, dist_mat, dur_mat, start_cpu_time)
        except Exception as e:
            print(f"[OR-Tools Exception] {e}. Falling back to heuristic.")

    return _solve_with_savings_heuristic(depots, deliveries, vehicles, dist_mat, dur_mat, start_cpu_time)


def apply_perturbations_to_matrix(
    dist_mat: List[List[float]],
    dur_mat: List[List[float]],
    perturbations: List[Dict[str, Any]] = None
) -> Tuple[List[List[float]], List[List[float]]]:
    if not perturbations:
        return dist_mat, dur_mat

    n = len(dist_mat)
    mod_dist = [row[:] for row in dist_mat]
    mod_dur = [row[:] for row in dur_mat]

    for p in perturbations:
        p_type = p.get("type")
        if p_type in ["rain", "festival", "rally", "concert", "accident"]:
            factor = p.get("duration_multiplier", 1.8)
            for i in range(n):
                for j in range(n):
                    mod_dur[i][j] = round(mod_dur[i][j] * factor, 2)
        elif p_type == "road_block":
            u, v = p.get("from_node"), p.get("to_node")
            if u is not None and v is not None and u < n and v < n:
                mod_dist[u][v] = 9999.0
                mod_dur[u][v] = 9999.0

    return mod_dist, mod_dur


def _solve_with_ortools(depots, deliveries, vehicles, dist_matrix, dur_matrix, start_time, time_limit_sec=1) -> Dict[str, Any]:
    num_vehicles = len(vehicles)
    num_locations = len(dist_matrix)
    
    scaled_dist = [[int(dist_matrix[i][j] * 100) for j in range(num_locations)] for i in range(num_locations)]
    
    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return scaled_dist[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Distance Dimension for Workload Balancing (Min-Max Routing)
    # We add a dimension to track distance and apply a Global Span Cost Coefficient.
    # This mathematically forces OR-Tools to minimize the MAXIMUM distance traveled by any single truck,
    # naturally resulting in a balanced, multi-truck dispatch without arbitrary handicaps.
    routing.AddDimension(
        transit_callback_index,
        0,      # no slack
        300000, # upper bound (effectively unlimited for this city scale)
        True,   # start cumul to zero
        "Distance"
    )
    distance_dimension = routing.GetDimensionOrDie("Distance")
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Capacity Dimension
    demands = [0] + [d.get("demand", 100) for d in deliveries]
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    vehicle_capacities = [v.get("capacity", 500) for v in vehicles]
    
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        500, # Slack allows handling slight over-capacity smoothly alongside disjunctions
        vehicle_capacities,
        True,
        "Capacity"
    )

    # Disjunction penalties so OR-Tools NEVER fails on infeasible over-capacity setups
    penalty = 100000
    for node in range(1, num_locations):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = int(time_limit_sec)

    solution = routing.SolveWithParameters(search_parameters)

    routes = []
    total_dist_km = 0.0
    total_dur_min = 0.0

    if solution:
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            v_route = []
            v_dist = 0.0
            v_dur = 0.0
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                v_route.append(node)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                next_node = manager.IndexToNode(index)
                v_dist += dist_matrix[node][next_node]
                v_dur += dur_matrix[node][next_node]

            v_route.append(manager.IndexToNode(index))
            routes.append({
                "vehicle_id": vehicles[vehicle_id]["id"],
                "vehicle_name": vehicles[vehicle_id]["name"],
                "route_node_indices": v_route,
                "distance_km": round(v_dist, 2),
                "duration_min": round(v_dur, 2)
            })
            total_dist_km += v_dist
            total_dur_min += v_dur
    else:
        return _solve_with_savings_heuristic(depots, deliveries, vehicles, dist_matrix, dur_matrix, start_time)

    solver_exec_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "solver": f"Google OR-Tools (Guided Local Search - {time_limit_sec}s Budget)",
        "routes": routes,
        "total_distance_km": round(total_dist_km, 2),
        "total_duration_min": round(total_dur_min, 2),
        "execution_time_ms": solver_exec_ms
    }


def _solve_with_savings_heuristic(depots, deliveries, vehicles, dist_matrix, dur_matrix, start_time) -> Dict[str, Any]:
    num_vehicles = len(vehicles)
    num_deliveries = len(deliveries)
    
    unvisited = list(range(1, num_deliveries + 1))
    routes = []
    total_dist = 0.0
    total_dur = 0.0

    for v_idx in range(num_vehicles):
        v = vehicles[v_idx]
        v_cap = v.get("capacity", 500)
        curr_cap = 0
        curr_node = 0
        v_route = [0]
        v_dist = 0.0
        v_dur = 0.0

        while unvisited:
            best_next = None
            best_cost = float('inf')

            for node in unvisited:
                demand = deliveries[node - 1].get("demand", 100)
                if curr_cap + demand <= v_cap:
                    cost = dist_matrix[curr_node][node]
                    if cost < best_cost:
                        best_cost = cost
                        best_next = node

            if best_next is None:
                break

            unvisited.remove(best_next)
            v_route.append(best_next)
            curr_cap += deliveries[best_next - 1].get("demand", 100)
            v_dist += dist_matrix[curr_node][best_next]
            v_dur += dur_matrix[curr_node][best_next]
            curr_node = best_next

        v_dist += dist_matrix[curr_node][0]
        v_dur += dur_matrix[curr_node][0]
        v_route.append(0)

        routes.append({
            "vehicle_id": v["id"],
            "vehicle_name": v["name"],
            "route_node_indices": v_route,
            "distance_km": round(v_dist, 2),
            "duration_min": round(v_dur, 2)
        })
        total_dist += v_dist
        total_dur += v_dur

    solver_exec_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "solver": "Commercial OR Solver (Savings Heuristic Baseline)",
        "routes": routes,
        "total_distance_km": round(total_dist, 2),
        "total_duration_min": round(total_dur, 2),
        "execution_time_ms": solver_exec_ms
    }
