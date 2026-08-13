# test_waynex.py

"""
Automated Integration Test Suite for Waynex (v0.2).
"""

import unittest
from city_templates import CITY_TEMPLATES, get_city_template
from osrm_client import fetch_osrm_distance_matrix, fetch_route_geometry
from ortools_solver import solve_vrp_ortools
from gnn_encoder import prepare_graph_node_features
from drl_policy import WaynexActorCriticPolicy, run_waynex_neural_routing
from mcp_server import execute_mcp_tool
from agent_orchestrator import generate_openai_dispatch_summary
import torch


class TestWaynexPipeline(unittest.TestCase):

    def test_city_templates(self):
        self.assertIn("bengaluru", CITY_TEMPLATES)
        self.assertIn("san_francisco", CITY_TEMPLATES)
        self.assertIn("tokyo", CITY_TEMPLATES)
        blr = get_city_template("bengaluru")
        self.assertEqual(len(blr["deliveries"]), 10)

    def test_osrm_client(self):
        coords = [
            {"lat": 12.9716, "lng": 77.5946},
            {"lat": 12.9352, "lng": 77.6245}
        ]
        dist, dur = fetch_osrm_distance_matrix(coords)
        self.assertEqual(len(dist), 2)
        self.assertEqual(len(dur), 2)
        self.assertGreater(dist[0][1], 0)

    def test_ortools_solver(self):
        tmpl = get_city_template("bengaluru")
        coords = [{"lat": tmpl["depots"][0]["lat"], "lng": tmpl["depots"][0]["lng"]}] + [{"lat": d["lat"], "lng": d["lng"]} for d in tmpl["deliveries"]]
        dist, dur = fetch_osrm_distance_matrix(coords)
        
        sol = solve_vrp_ortools(tmpl["depots"], tmpl["deliveries"], tmpl["vehicles"], dist, dur)
        self.assertIn("routes", sol)
        self.assertGreater(sol["total_distance_km"], 0)

    def test_waynex_neural_routing(self):
        tmpl = get_city_template("bengaluru")
        coords = [{"lat": tmpl["depots"][0]["lat"], "lng": tmpl["depots"][0]["lng"]}] + [{"lat": d["lat"], "lng": d["lng"]} for d in tmpl["deliveries"]]
        dist, dur = fetch_osrm_distance_matrix(coords)
        
        model = WaynexActorCriticPolicy(6, 64, 64)
        sol = run_waynex_neural_routing(model, tmpl["depots"], tmpl["deliveries"], tmpl["vehicles"], dist, dur)
        self.assertIn("routes", sol)
        self.assertLess(sol["execution_time_ms"], 500)

    def test_mcp_tool_execution(self):
        tmpl = get_city_template("san_francisco")
        args = {
            "depots": tmpl["depots"],
            "deliveries": tmpl["deliveries"],
            "vehicles": tmpl["vehicles"]
        }
        res = execute_mcp_tool("run_dual_benchmark", args)
        self.assertIn("or_tools_baseline", res)
        self.assertIn("waynex_neural_engine", res)


if __name__ == "__main__":
    unittest.main()
