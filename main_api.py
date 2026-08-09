# main_api.py

"""
Waynex API Engine Backend Server.
Uses Python native `http.server` for zero-dependency high-reliability REST & static web serving.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import urllib.parse
import re

def sanitize_input(text):
    """Strip HTML tags and limit input length for security."""
    if not isinstance(text, str):
        return str(text)
    cleaned = re.sub(r'<[^>]+>', '', text)
    return cleaned[:500]  # Limit input length

from city_templates import CITY_TEMPLATES, get_city_template
from geocoder import geocode_address, autocomplete_suggestions
from mcp_server import execute_mcp_tool
from agent_orchestrator import generate_openai_dispatch_summary


class WaynexAPIRequestHandler(BaseHTTPRequestHandler):

    def _set_headers(self, status_code=200, content_type="application/json"):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('X-XSS-Protection', '1; mode=block')
        self.send_header('Referrer-Policy', 'strict-origin-when-cross-origin')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline' https://unpkg.com https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com https://unpkg.com; font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; img-src 'self' data: https://*.basemaps.cartocdn.com https://*.tile.openstreetmap.org; connect-src 'self' https://router.project-osrm.org https://nominatim.openstreetmap.org https://api.openai.com")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path == "/api/templates":
            templates_list = [
                {"key": k, "name": v["name"], "country": v["country"], "deliveries_count": len(v["deliveries"])}
                for k, v in CITY_TEMPLATES.items()
            ]
            self._set_headers(200)
            self.wfile.write(json.dumps({"templates": templates_list}).encode('utf-8'))

        elif path.startswith("/api/templates/"):
            city_key = path.replace("/api/templates/", "").strip()
            template = get_city_template(city_key)
            self._set_headers(200)
            self.wfile.write(json.dumps(template).encode('utf-8'))

        elif path.startswith("/static/"):
            rel_path = path.replace("/static/", "")
            file_path = os.path.join("public", rel_path)
            
            if os.path.exists(file_path):
                content_type = "text/css" if file_path.endswith(".css") else "application/javascript"
                self._set_headers(200, content_type)
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._set_headers(404, "text/plain")
                self.wfile.write(b"File not found")

        else:
            index_path = "public/index.html"
            if os.path.exists(index_path):
                self._set_headers(200, "text/html")
                with open(index_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._set_headers(200, "text/html")
                self.wfile.write(b"<h1>Waynex API Engine v2.0</h1>")

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body_data = self.rfile.read(content_length).decode('utf-8')
        payload = json.loads(body_data) if body_data else {}

        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path == "/api/autocomplete":
            query = sanitize_input(payload.get("query", ""))
            city_context = sanitize_input(payload.get("city_context", ""))
            suggestions = autocomplete_suggestions(query, city_context)
            self._set_headers(200)
            self.wfile.write(json.dumps({"suggestions": suggestions}).encode('utf-8'))

        elif path == "/api/geocode":
            query = sanitize_input(payload.get("query", ""))
            city_context = sanitize_input(payload.get("city_context", ""))
            res = geocode_address(query, city_context)
            if res:
                self._set_headers(200)
                self.wfile.write(json.dumps(res).encode('utf-8'))
            else:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": f"Geocoding failed for '{query}'"}).encode('utf-8'))

        elif path == "/api/benchmark":
            depots = payload.get("depots", [])
            deliveries = payload.get("deliveries", [])
            vehicles = payload.get("vehicles", [])
            perturbations = payload.get("perturbations", [])
            city_key = payload.get("city_key", "Bengaluru")
            custom_api_key = payload.get("openai_api_key", "")
            user_prompt = payload.get("user_prompt", "")

            ortools_time_limit = payload.get("ortools_time_limit", 1)
            tool_args = {
                "depots": depots,
                "deliveries": deliveries,
                "vehicles": vehicles,
                "perturbations": perturbations,
                "ortools_time_limit": ortools_time_limit
            }
            result = execute_mcp_tool("run_dual_benchmark", tool_args)
            
            # OpenAI Dispatch Explainability Summary
            summary = generate_openai_dispatch_summary(city_key, result, perturbations, custom_api_key, user_prompt)
            result["ai_dispatch_copilot_summary"] = summary

            self._set_headers(200)
            self.wfile.write(json.dumps(result).encode('utf-8'))

        elif path == "/api/perturb":
            event_type = payload.get("event_type", "accident")
            location = payload.get("affected_location_name", "City Center")

            tool_args = {
                "event_type": event_type,
                "affected_location_name": location
            }
            res = execute_mcp_tool("inject_perturbation", tool_args)

            self._set_headers(200)
            self.wfile.write(json.dumps(res).encode('utf-8'))
            
        elif path == "/api/chat":
            messages = payload.get("messages", [])
            benchmark_context = payload.get("benchmark_context", {})
            custom_api_key = payload.get("openai_api_key", "")
            
            from agent_orchestrator import generate_openai_chat_response
            reply = generate_openai_chat_response(messages, benchmark_context, custom_api_key)
            
            self._set_headers(200)
            self.wfile.write(json.dumps({"reply": reply}).encode('utf-8'))

        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode('utf-8'))


def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, WaynexAPIRequestHandler)
    print(f"🚀 Waynex API Engine running on http://localhost:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    run_server(port)
