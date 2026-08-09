# osrm_client.py

"""
OSRM Routing & Distance Matrix Client for Waynex.
Fetches real road network distance/duration matrices and geometry polylines.
Includes robust fallback to Haversine geodesic calculation if OSRM server is offline/unreachable.
"""

import math
import json
import urllib.request
import urllib.parse
from typing import List, Dict, Tuple, Any

# Public OSRM routing server URL (can be customized to self-hosted OSRM server)
OSRM_TABLE_URL = "http://router.project-osrm.org/table/v1/driving/"
OSRM_ROUTE_URL = "http://router.project-osrm.org/route/v1/driving/"


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the Great-Circle distance between two points in km."""
    R = 6371.0  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def get_haversine_matrix(coords: List[Dict[str, float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """Fallback generator for distance (km) and duration (mins) matrices using Haversine."""
    n = len(coords)
    dist_matrix = [[0.0] * n for _ in range(n)]
    dur_matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                dist_km = haversine_distance_km(
                    coords[i]["lat"], coords[i]["lng"],
                    coords[j]["lat"], coords[j]["lng"]
                )
                # Estimate street distance ~ 1.3x geodesic line distance in cities
                street_dist = dist_km * 1.3
                # Assume average city speed 30 km/h -> duration in minutes = dist / 30 * 60
                duration_mins = (street_dist / 30.0) * 60.0
                dist_matrix[i][j] = round(street_dist, 2)
                dur_matrix[i][j] = round(duration_mins, 2)

    return dist_matrix, dur_matrix


def fetch_osrm_distance_matrix(coords: List[Dict[str, float]], timeout: int = 4) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Fetch distance matrix (km) and duration matrix (minutes) for a set of coordinates.
    coords: List of dicts [{"lat": 12.9, "lng": 77.6}, ...]
    Returns (distance_matrix_km, duration_matrix_minutes)
    """
    if len(coords) < 2:
        return [[0.0]], [[0.0]]

    try:
        # OSRM expects coordinates as "lng,lat;lng,lat;..."
        coord_str = ";".join([f"{c['lng']:.6f},{c['lat']:.6f}" for c in coords])
        url = f"{OSRM_TABLE_URL}{coord_str}?annotations=distance,duration"
        
        req = urllib.request.Request(url, headers={"User-Agent": "WaynexLogisticsEngine/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            if data.get("code") == "Ok" and "distances" in data and "durations" in data:
                # OSRM distances are in meters; durations are in seconds
                distances = data["distances"]
                durations = data["durations"]
                
                n = len(coords)
                dist_km = [[round(distances[i][j] / 1000.0, 2) for j in range(n)] for i in range(n)]
                dur_min = [[round(durations[i][j] / 60.0, 2) for j in range(n)] for i in range(n)]
                return dist_km, dur_min

    except Exception as e:
        print(f"[OSRM Matrix Warning] OSRM API call failed or timed out ({e}). Using Haversine fallback.")

    return get_haversine_matrix(coords)


def fetch_route_geometry(waypoints: List[Dict[str, float]], timeout: int = 4) -> List[List[float]]:
    """
    Fetch real road polyline geometry [[lat, lng], [lat, lng], ...] connecting a series of waypoints.
    """
    if len(waypoints) < 2:
        return [[w["lat"], w["lng"]] for w in waypoints]

    try:
        coord_str = ";".join([f"{w['lng']:.6f},{w['lat']:.6f}" for w in waypoints])
        url = f"{OSRM_ROUTE_URL}{coord_str}?overview=full&geometries=geojson"
        
        req = urllib.request.Request(url, headers={"User-Agent": "WaynexLogisticsEngine/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            if data.get("code") == "Ok" and data.get("routes"):
                geojson = data["routes"][0]["geometry"]["coordinates"]
                # Convert GeoJSON [lng, lat] to Leaflet [lat, lng]
                leaflet_coords = [[pt[1], pt[0]] for pt in geojson]
                return leaflet_coords

    except Exception as e:
        print(f"[OSRM Route Warning] Polyline fetch failed ({e}). Returning straight waypoints.")

    return [[w["lat"], w["lng"]] for w in waypoints]


if __name__ == "__main__":
    test_coords = [
        {"lat": 12.9716, "lng": 77.5946},
        {"lat": 12.9352, "lng": 77.6245},
        {"lat": 12.9784, "lng": 77.6408}
    ]
    dist, dur = fetch_osrm_distance_matrix(test_coords)
    print("Distance Matrix (km):", dist)
    print("Duration Matrix (min):", dur)
    geom = fetch_route_geometry(test_coords)
    print(f"Fetched {len(geom)} polyline points.")
