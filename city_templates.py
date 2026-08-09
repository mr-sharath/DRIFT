# city_templates.py

"""
Pre-configured city templates for Waynex — The next way to route.
Provides real GPS coordinates for depots, delivery stops, fuel/charging hubs, and fleet specs for:
- Bengaluru (India)
- San Francisco (USA)
- London (UK)
"""

CITY_TEMPLATES = {
    "bengaluru": {
        "name": "Bengaluru Tech & Logistics Hub",
        "country": "India",
        "center": {"lat": 12.9716, "lng": 77.5946},
        "zoom": 12,
        "depots": [
            {"id": "depot_blr_1", "name": "Peenya Industrial Distribution Center", "lat": 13.0285, "lng": 77.5197},
            {"id": "depot_blr_2", "name": "Koramangala Logistics Hub", "lat": 12.9352, "lng": 77.6245}
        ],
        "charging_stations": [
            {"id": "charge_blr_1", "name": "Indiranagar Fast Charger", "lat": 12.9784, "lng": 77.6408},
            {"id": "charge_blr_2", "name": "Electronic City EV Hub", "lat": 12.8452, "lng": 77.6602}
        ],
        "vehicles": [
            {"id": "v_blr_1", "name": "Light Truck 01", "type": "fuel", "capacity": 500, "depot_id": "depot_blr_1", "speed_kmh": 35},
            {"id": "v_blr_2", "name": "Heavy Duty Truck 02", "type": "fuel", "capacity": 1200, "depot_id": "depot_blr_1", "speed_kmh": 30},
            {"id": "v_blr_3", "name": "Light Truck 03", "type": "fuel", "capacity": 600, "depot_id": "depot_blr_2", "speed_kmh": 35},
            {"id": "v_blr_4", "name": "Medium Truck 04", "type": "fuel", "capacity": 800, "depot_id": "depot_blr_2", "speed_kmh": 32}
        ],
        "deliveries": [
            {"id": "del_1", "name": "MG Road Retail Store", "lat": 12.9756, "lng": 77.6066, "demand": 120, "time_window": [9, 13], "priority": "high"},
            {"id": "del_2", "name": "Whitefield Tech Park Block A", "lat": 12.9855, "lng": 77.7279, "demand": 250, "time_window": [10, 14], "priority": "medium"},
            {"id": "del_3", "name": "Electronic City Phase 1 Mall", "lat": 12.8485, "lng": 77.6645, "demand": 180, "time_window": [11, 16], "priority": "high"},
            {"id": "del_4", "name": "HSR Layout Sector 1 Mart", "lat": 12.9121, "lng": 77.6446, "demand": 90, "time_window": [9, 12], "priority": "medium"},
            {"id": "del_5", "name": "Jayanagar 4th Block Superstore", "lat": 12.9299, "lng": 77.5826, "demand": 210, "time_window": [10, 15], "priority": "high"},
            {"id": "del_6", "name": "Hebbal Outer Ring Rd Complex", "lat": 13.0358, "lng": 77.5970, "demand": 150, "time_window": [9, 14], "priority": "medium"},
            {"id": "del_7", "name": "Yelahanka New Town Market", "lat": 13.0995, "lng": 77.5956, "demand": 300, "time_window": [12, 17], "priority": "medium"},
            {"id": "del_8", "name": "Marathahalli Innovation Hub", "lat": 12.9591, "lng": 77.6974, "demand": 140, "time_window": [11, 15], "priority": "high"},
            {"id": "del_9", "name": "Bannerghatta Rd Hypermarket", "lat": 12.8938, "lng": 77.5982, "demand": 160, "time_window": [10, 16], "priority": "medium"},
            {"id": "del_10", "name": "Rajajinagar Industrial Depot", "lat": 12.9901, "lng": 77.5527, "demand": 220, "time_window": [9, 13], "priority": "high"}
        ]
    },

    "san_francisco": {
        "name": "San Francisco Bay Area Fleet",
        "country": "USA",
        "center": {"lat": 37.7749, "lng": -122.4194},
        "zoom": 12,
        "depots": [
            {"id": "depot_sf_1", "name": "Port of SF Logistics Depot", "lat": 37.7905, "lng": -122.3892},
            {"id": "depot_sf_2", "name": "Mission District Distribution Center", "lat": 37.7599, "lng": -122.4148}
        ],
        "charging_stations": [],
        "vehicles": [
            {"id": "v_sf_1", "name": "Light Truck 101", "type": "fuel", "capacity": 600, "depot_id": "depot_sf_1", "speed_kmh": 40},
            {"id": "v_sf_2", "name": "Heavy Duty Truck 102", "type": "fuel", "capacity": 1000, "depot_id": "depot_sf_1", "speed_kmh": 35},
            {"id": "v_sf_3", "name": "Light Truck 103", "type": "fuel", "capacity": 550, "depot_id": "depot_sf_2", "speed_kmh": 40},
            {"id": "v_sf_4", "name": "Medium Truck 104", "type": "fuel", "capacity": 900, "depot_id": "depot_sf_2", "speed_kmh": 35}
        ],
        "deliveries": [
            {"id": "del_sf_1", "name": "Financial District Tower B", "lat": 37.7937, "lng": -122.4008, "demand": 150, "time_window": [8, 12], "priority": "high"},
            {"id": "del_sf_2", "name": "Fisherman's Wharf Retail Outlet", "lat": 37.8080, "lng": -122.4177, "demand": 200, "time_window": [9, 13], "priority": "medium"},
            {"id": "del_sf_3", "name": "Sunset District Supermarket", "lat": 37.7533, "lng": -122.4944, "demand": 180, "time_window": [10, 15], "priority": "medium"},
            {"id": "del_sf_4", "name": "Richmond District Medical Supply", "lat": 37.7794, "lng": -122.4779, "demand": 100, "time_window": [8, 11], "priority": "high"},
            {"id": "del_sf_5", "name": "Castro Market Hub", "lat": 37.7609, "lng": -122.4350, "demand": 140, "time_window": [9, 14], "priority": "medium"},
            {"id": "del_sf_6", "name": "Potrero Hill Logistics Office", "lat": 37.7577, "lng": -122.3995, "demand": 220, "time_window": [11, 16], "priority": "high"},
            {"id": "del_sf_7", "name": "Dogpatch Fulfillment Center", "lat": 37.7602, "lng": -122.3879, "demand": 310, "time_window": [10, 16], "priority": "medium"},
            {"id": "del_sf_8", "name": "Marina Green Event Hub", "lat": 37.8052, "lng": -122.4376, "demand": 130, "time_window": [12, 17], "priority": "low"}
        ]
    },

    "tokyo": {
        "name": "Tokyo Metro Logistics Center",
        "country": "Japan",
        "center": {"lat": 35.6895, "lng": 139.6917},
        "zoom": 12,
        "depots": [
            {"id": "depot_tyo_1", "name": "Shinjuku Distribution Hub", "lat": 35.6894, "lng": 139.7005},
            {"id": "depot_tyo_2", "name": "Shinagawa Logistics Park", "lat": 35.6284, "lng": 139.7388}
        ],
        "charging_stations": [],
        "vehicles": [
            {"id": "v_tyo_1", "name": "Light Truck 01", "type": "fuel", "capacity": 550, "depot_id": "depot_tyo_1", "speed_kmh": 35},
            {"id": "v_tyo_2", "name": "Heavy Duty Truck 02", "type": "fuel", "capacity": 1100, "depot_id": "depot_tyo_1", "speed_kmh": 30},
            {"id": "v_tyo_3", "name": "Light Truck 03", "type": "fuel", "capacity": 600, "depot_id": "depot_tyo_2", "speed_kmh": 35},
            {"id": "v_tyo_4", "name": "Medium Truck 04", "type": "fuel", "capacity": 950, "depot_id": "depot_tyo_2", "speed_kmh": 30}
        ],
        "deliveries": [
            {"id": "del_tyo_1", "name": "Shibuya Retail Store", "lat": 35.6580, "lng": 139.7016, "demand": 210, "time_window": [8, 12], "priority": "high"},
            {"id": "del_tyo_2", "name": "Ginza Shopping District", "lat": 35.6712, "lng": 139.7661, "demand": 190, "time_window": [9, 13], "priority": "high"},
            {"id": "del_tyo_3", "name": "Akihabara Tech Hub", "lat": 35.6983, "lng": 139.7731, "demand": 150, "time_window": [10, 15], "priority": "medium"},
            {"id": "del_tyo_4", "name": "Roppongi Hills Center", "lat": 35.6604, "lng": 139.7292, "demand": 170, "time_window": [9, 14], "priority": "medium"},
            {"id": "del_tyo_5", "name": "Odaiba Entertainment Complex", "lat": 35.6248, "lng": 139.7758, "demand": 260, "time_window": [11, 16], "priority": "high"},
            {"id": "del_tyo_6", "name": "Ikebukuro Commercial Area", "lat": 35.7295, "lng": 139.7109, "demand": 130, "time_window": [8, 12], "priority": "medium"},
            {"id": "del_tyo_7", "name": "Asakusa Tourist Zone", "lat": 35.7147, "lng": 139.7966, "demand": 140, "time_window": [10, 14], "priority": "high"}
        ]
    }
}

def get_city_template(city_key: str):
    """Retrieve city template dictionary by key."""
    return CITY_TEMPLATES.get(city_key.lower(), CITY_TEMPLATES["bengaluru"])
