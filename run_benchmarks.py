# run_benchmarks.py

import time
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from city_templates import CITY_TEMPLATES, get_city_template
from osrm_client import get_haversine_matrix
from ortools_solver import solve_vrp_ortools
from drl_policy import WaynexActorCriticPolicy, run_waynex_neural_routing

def load_policy(checkpoint_path):
    model = WaynexActorCriticPolicy(in_features=6, hidden_dim=64, embed_dim=64)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"Loaded {checkpoint_path}")
    except Exception as e:
        print(f"Warning loading {checkpoint_path}: {e}")
    model.eval()
    return model

def run_scaling_benchmark():
    print("Running Scaling Benchmark (Nodes 10 to 200)...")
    node_counts = [10, 20, 50, 100, 200]
    
    model_v1 = load_policy("waynex_policy_v1.pt")
    model_v2 = load_policy("waynex_policy_v2.pt")
    
    results = {
        "node_counts": node_counts,
        "ortools_time_ms": [],
        "waynex_v1_time_ms": [],
        "waynex_v2_time_ms": [],
        "ortools_dist_km": [],
        "waynex_v1_dist_km": [],
        "waynex_v2_dist_km": []
    }
    
    for n in node_counts:
        print(f"  Testing N={n} nodes...")
        num_deliveries = n - 1
        base_lat, base_lng = 12.9716, 77.5946
        coords = [{"lat": base_lat, "lng": base_lng}]
        deliveries = []
        for i in range(num_deliveries):
            d_lat = base_lat + random.uniform(-0.1, 0.1)
            d_lng = base_lng + random.uniform(-0.1, 0.1)
            coords.append({"lat": d_lat, "lng": d_lng})
            deliveries.append({
                "id": f"del_{i+1}",
                "name": f"Pin {i+1}",
                "lat": d_lat,
                "lng": d_lng,
                "demand": random.randint(50, 150),
                "time_window": [9, 17]
            })
        
        depots = [{"id": "depot_1", "name": "Main Hub", "lat": base_lat, "lng": base_lng}]
        
        num_vehicles = max(2, n // 15)
        vehicles = []
        for v in range(num_vehicles):
            vehicles.append({
                "id": f"v_{v+1}",
                "name": f"Truck {v+1}",
                "capacity": 800,
                "type": "fuel",
                "speed_kmh": 35
            })
            
        dist_mat, dur_mat = get_haversine_matrix(coords)
        
        # 1. OR-Tools
        t0 = time.time()
        or_res = solve_vrp_ortools(depots, deliveries, vehicles, dist_mat, dur_mat, time_limit_sec=2)
        or_time = (time.time() - t0) * 1000.0
        
        # 2. Waynex V1 (Base)
        t0 = time.time()
        v1_res = run_waynex_neural_routing(model_v1, depots, deliveries, vehicles, dist_mat, dur_mat)
        v1_time = (time.time() - t0) * 1000.0
        
        # 3. Waynex V2 (OPT2)
        t0 = time.time()
        v2_res = run_waynex_neural_routing(model_v2, depots, deliveries, vehicles, dist_mat, dur_mat)
        v2_time = (time.time() - t0) * 1000.0
        
        results["ortools_time_ms"].append(round(or_time, 2))
        results["waynex_v1_time_ms"].append(round(v1_time, 2))
        results["waynex_v2_time_ms"].append(round(v2_time, 2))
        
        results["ortools_dist_km"].append(round(or_res.get("total_distance_km", 0), 2))
        results["waynex_v1_dist_km"].append(round(v1_res.get("total_distance_km", 0), 2))
        results["waynex_v2_dist_km"].append(round(v2_res.get("total_distance_km", 0), 2))

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Benchmark data saved to benchmark_results.json")
    return results

def generate_plots(results):
    node_counts = results["node_counts"]
    
    # Figure 1: Latency Scaling (Log Scale)
    plt.figure(figsize=(7, 4.5))
    plt.plot(node_counts, results["ortools_time_ms"], 'r-o', label="Google OR-Tools (MILP/GLS)", linewidth=2)
    plt.plot(node_counts, results["waynex_v1_time_ms"], 'b-s', label="Waynex Base (v1)", linewidth=2)
    plt.plot(node_counts, results["waynex_v2_time_ms"], 'g-^', label="Waynex OPT2 (v2)", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Number of Graph Nodes (N)", fontsize=11)
    plt.ylabel("Execution Latency (ms) [Log Scale]", fontsize=11)
    plt.title("Execution Latency Scaling Comparison", fontsize=12, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("fig1_latency_scaling.png", dpi=300)
    plt.close()
    
    # Figure 2: Total Distance Traveled
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(node_counts))
    width = 0.25
    plt.bar(x - width, results["ortools_dist_km"], width, label="Google OR-Tools", color="#e74c3c")
    plt.bar(x, results["waynex_v1_dist_km"], width, label="Waynex Base (v1)", color="#3498db")
    plt.bar(x + width, results["waynex_v2_dist_km"], width, label="Waynex OPT2 (v2)", color="#2ecc71")
    plt.xlabel("Number of Graph Nodes (N)", fontsize=11)
    plt.ylabel("Total Fleet Distance Traveled (km)", fontsize=11)
    plt.xticks(x, [str(n) for n in node_counts])
    plt.title("Total Fleet Route Distance Efficiency", fontsize=12, fontweight='bold')
    plt.grid(True, axis='y', ls="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("fig2_route_quality.png", dpi=300)
    plt.close()
    
    print("Plots saved: fig1_latency_scaling.png, fig2_route_quality.png")

if __name__ == "__main__":
    res = run_scaling_benchmark()
    generate_plots(res)
