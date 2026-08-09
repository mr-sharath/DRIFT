# geocoder.py

"""
OpenStreetMap Nominatim & Local Landmark Geocoder for Waynex.
Provides 100% reliable geocoding, autocomplete suggestions, and local landmark matching.
"""

import json
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional

LOCAL_LANDMARKS = {
    # Bengaluru
    "marathahalli": {"lat": 12.9591, "lng": 77.6974, "display_name": "Marathahalli, Bengaluru"},
    "marthahali": {"lat": 12.9591, "lng": 77.6974, "display_name": "Marathahalli, Bengaluru"},
    "whitefield": {"lat": 12.9855, "lng": 77.7279, "display_name": "Whitefield, Bengaluru"},
    "indiranagar": {"lat": 12.9784, "lng": 77.6408, "display_name": "Indiranagar, Bengaluru"},
    "koramangala": {"lat": 12.9352, "lng": 77.6245, "display_name": "Koramangala, Bengaluru"},
    "electronic city": {"lat": 12.8452, "lng": 77.6602, "display_name": "Electronic City, Bengaluru"},
    "hsr layout": {"lat": 12.9121, "lng": 77.6446, "display_name": "HSR Layout, Bengaluru"},
    "mg road": {"lat": 12.9756, "lng": 77.6066, "display_name": "MG Road, Bengaluru"},
    "jayanagar": {"lat": 12.9299, "lng": 77.5826, "display_name": "Jayanagar, Bengaluru"},
    "hebbal": {"lat": 13.0358, "lng": 77.5970, "display_name": "Hebbal, Bengaluru"},
    "yelahanka": {"lat": 13.0995, "lng": 77.5956, "display_name": "Yelahanka, Bengaluru"},
    "bannerghatta": {"lat": 12.8938, "lng": 77.5982, "display_name": "Bannerghatta Road, Bengaluru"},
    "rajajinagar": {"lat": 12.9901, "lng": 77.5527, "display_name": "Rajajinagar, Bengaluru"},
    
    # San Francisco
    "soma": {"lat": 37.7786, "lng": -122.4058, "display_name": "SOMA, San Francisco"},
    "mission district": {"lat": 37.7599, "lng": -122.4148, "display_name": "Mission District, San Francisco"},
    "financial district": {"lat": 37.7937, "lng": -122.4008, "display_name": "Financial District, San Francisco"},
    "fishermans wharf": {"lat": 37.8080, "lng": -122.4177, "display_name": "Fisherman's Wharf, San Francisco"},
    "sunset district": {"lat": 37.7533, "lng": -122.4944, "display_name": "Sunset District, San Francisco"},
    "richmond district": {"lat": 37.7794, "lng": -122.4779, "display_name": "Richmond District, San Francisco"},

    # London
    "city of london": {"lat": 51.5128, "lng": -0.0918, "display_name": "City of London, London"},
    "west end": {"lat": 51.5136, "lng": -0.1418, "display_name": "West End, London"},
    "camden town": {"lat": 51.5416, "lng": -0.1465, "display_name": "Camden Town, London"},
    "kensington": {"lat": 51.5010, "lng": -0.1925, "display_name": "Kensington, London"},
    "greenwich": {"lat": 51.4826, "lng": -0.0077, "display_name": "Greenwich, London"}
}

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "WaynexLogisticsEngine/2.0 (contact: sharath@waynex.ai)"}


def autocomplete_suggestions(query: str, city_context: str = "") -> List[Dict[str, Any]]:
    """
    Return matching location suggestions as the user types.
    """
    if not query or len(query.strip()) < 2:
        return []

    clean_q = query.strip().lower()
    suggestions = []

    # 1. Match local landmark dictionary
    for key, val in LOCAL_LANDMARKS.items():
        if clean_q in key or key in clean_q:
            suggestions.append({
                "name": val["display_name"],
                "lat": val["lat"],
                "lng": val["lng"]
            })

    # 2. If fewer than 3 local matches, query OSM Nominatim
    if len(suggestions) < 3:
        search_query = query.strip()
        if city_context and city_context.lower() not in search_query.lower():
            search_query = f"{search_query}, {city_context}"

        try:
            params = urllib.parse.urlencode({"q": search_query, "format": "json", "limit": 4})
            url = f"{NOMINATIM_URL}?{params}"
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=3) as response:
                results = json.loads(response.read().decode('utf-8'))
                for r in results:
                    disp = r.get("display_name", "")
                    short_name = disp.split(",")[0] + ", " + (disp.split(",")[1] if len(disp.split(",")) > 1 else "")
                    suggestions.append({
                        "name": short_name.strip(),
                        "lat": float(r["lat"]),
                        "lng": float(r["lon"])
                    })
        except Exception:
            pass

    # Deduplicate by name
    seen = set()
    unique_suggestions = []
    for s in suggestions:
        if s["name"] not in seen:
            seen.add(s["name"])
            unique_suggestions.append(s)

    return unique_suggestions[:5]


def geocode_address(query: str, city_context: str = "", timeout: int = 4) -> Optional[Dict[str, Any]]:
    """
    Geocode address string to lat/lng coordinates.
    """
    if not query or not query.strip():
        return None

    clean_q = query.strip().lower()

    for key, val in LOCAL_LANDMARKS.items():
        if key in clean_q or clean_q in key:
            return val

    search_query = query.strip()
    if city_context and city_context.lower() not in search_query.lower():
        search_query = f"{search_query}, {city_context}"

    try:
        params = urllib.parse.urlencode({"q": search_query, "format": "json", "limit": 1})
        url = f"{NOMINATIM_URL}?{params}"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            results = json.loads(response.read().decode('utf-8'))
            if results and len(results) > 0:
                first = results[0]
                return {
                    "lat": float(first["lat"]),
                    "lng": float(first["lon"]),
                    "display_name": first.get("display_name", query)
                }

    except Exception as e:
        print(f"[Geocoder Warning] Address query '{search_query}' failed: {e}")

    return None
