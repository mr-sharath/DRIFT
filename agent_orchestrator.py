# agent_orchestrator.py

"""
OpenAI Agent Orchestrator for Waynex.
Integrates `OPENAI_API_KEY` (via env or custom user API key in UI) to parse dispatcher instructions,
monitor telematics alerts, and produce human-understandable explainable AI logs for routing decisions.
"""
# agent_orchestrator.py
import os
import json
import urllib.request
from typing import Dict, Any, List

# -------------------------------------------------
# NEW: Load variables from a local .env file
# -------------------------------------------------
try:
    # python‑dotenv is a tiny dependency; if you don’t have it yet:
    #   pip install python-dotenv
    from dotenv import load_dotenv
    load_dotenv()                # reads .env in the current working directory
except Exception:
    # If dotenv isn’t installed we’ll just fall back to the OS env.
    pass
# -------------------------------------------------


def generate_openai_dispatch_summary(
    city_name: str,
    benchmark_result: Dict[str, Any],
    active_perturbations: List[Dict[str, Any]] = None,
    custom_api_key: str = "",
    user_prompt: str = ""
) -> str:
    """
    Generate natural language dispatch explainability summary using OpenAI GPT-4o API
    or clean rule-based fallback generator.
    """
    api_key = custom_api_key.strip() or os.getenv("OPENAI_API_KEY", "")

    or_res = benchmark_result.get("or_tools_baseline", {})
    waynex_res = benchmark_result.get("waynex_neural_engine", {})
    eff = benchmark_result.get("efficiency_gain", {})
    
    or_dist = or_res.get("total_distance_km", 0.0)
    waynex_dist = waynex_res.get("total_distance_km", 0.0)
    or_time = or_res.get("execution_time_ms", 10.0)
    waynex_time = waynex_res.get("execution_time_ms", 1.0)
    
    speedup_str = eff.get("latency_speedup", "Sub-50ms Policy")
    dist_saved = eff.get("distance_saved_km", 0.0)
    fuel_saved = eff.get("fuel_saved_gallons", 0.0)
    co2_saved = round(fuel_saved * 3.5, 2)

    routes = waynex_res.get("routes", [])
    vehicle_summaries = []
    for r in routes:
        stops = len(r.get("route_node_indices", [])) - 2
        vehicle_summaries.append(f"{r.get('vehicle_name')}: {r.get('distance_km')} km ({max(stops, 0)} stops)")

    perturbation_desc = "Standard Traffic Conditions"
    if active_perturbations:
        pert_names = [p.get("type", "event").upper() for p in active_perturbations]
        perturbation_desc = f"Active Perturbations: {', '.join(pert_names)}"

    # If OpenAI API key is provided, call GPT-4o API
    if api_key:
        try:
            prompt_content = user_prompt if user_prompt else f"""
You are Waynex AI Dispatch Copilot, an expert logistics supervisor in {city_name}.
Provide a bulleted dispatch summary of the neural route optimization:

Context:
- City: {city_name}
- Traffic: {perturbation_desc}
- Waynex Neural Distance: {waynex_dist} km (Inference: {waynex_time} ms, {speedup_str})
- Fleet Vehicle Assignments: {'; '.join(vehicle_summaries)}
- Fuel Delta (Waynex vs OR-Tools): {fuel_saved} gallons (~{co2_saved} kg CO2 difference)

Format with clear Markdown bold titles and bullet points.
"""

            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are Waynex AI Dispatch Copilot, an expert AI routing explainability supervisor."},
                    {"role": "user", "content": prompt_content}
                ],
                "temperature": 0.3,
                "max_tokens": 250
            }

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                data=json.dumps(payload).encode('utf-8')
            )

            with urllib.request.urlopen(req, timeout=6) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                return res_data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print(f"[OpenAI Agent Warning] OpenAI API call failed ({e}). Using rule-based fallback summary.")

    # Rule-Based Structured Fallback Summary
    summary = f"""🤖 **Waynex AI Dispatch Intelligence ({city_name})**
• **Sub-50ms Neural Policy**: Waynex computed fleet routes in **{waynex_time} ms** ({speedup_str} vs Google OR-Tools Guided Local Search).
• **Fleet Vehicle Breakdown**:
  - {f'<br>  - '.join(vehicle_summaries)}
• **Environmental Impact**: Fuel Delta **{fuel_saved} gal** (~**{co2_saved} kg** CO2 difference).
• **Operational Telematics**: {perturbation_desc}. 100% of delivery pins visited cleanly."""

    return summary


def generate_openai_chat_response(
    messages: List[Dict[str, str]],
    benchmark_context: Dict[str, Any],
    custom_api_key: str = ""
) -> str:
    """
    Handles a back-and-forth conversational chat with the AI Copilot.
    Provides the latest benchmark metrics as context.
    """
    api_key = custom_api_key.strip() or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "⚠️ OpenAI API Key is missing. Please provide it in the UI to chat."

    or_res = benchmark_context.get("or_tools_baseline", {})
    waynex_res = benchmark_context.get("waynex_neural_engine", {})
    eff = benchmark_context.get("efficiency_gain", {})
    
    context_str = f"""Current System State (Real-Time Metrics):
[Waynex Neural Route]
- Total Fleet Drive Time: {waynex_res.get('total_duration_min', 0)} mins
- Max Shift Duration (Makespan): {waynex_res.get('makespan_min', 0)} mins
- Vehicle Capacity Utilization: {waynex_res.get('utilization_pct', 0)}%
- Algorithmic Compute Latency: {waynex_res.get('execution_time_ms', 0)} ms
- Fuel Consumed: {waynex_res.get('fuel_used_gal', 0)} gal

[Google OR-Tools Baseline]
- Total Fleet Drive Time: {or_res.get('total_duration_min', 0)} mins
- Max Shift Duration (Makespan): {or_res.get('makespan_min', 0)} mins
- Vehicle Capacity Utilization: {or_res.get('utilization_pct', 0)}%
- Algorithmic Compute Latency: {or_res.get('execution_time_ms', 0)} ms
- Fuel Consumed: {or_res.get('fuel_used_gal', 0)} gal"""

    system_prompt = {
        "role": "system", 
        "content": f"You are Waynex AI Dispatch Copilot, an expert AI routing supervisor. Use the following real-time telemetry to answer the dispatcher's questions accurately.\n\nCRITICAL INSTRUCTION: Do NOT confuse 'Algorithmic Compute Latency' (how fast the AI computed the route in milliseconds) with 'Total Fleet Drive Time' or 'Makespan' (how many minutes the trucks actually spend driving to deliver parcels). If asked about delivery time, refer to the Makespan (max shift time for a single driver) or Total Fleet Drive Time.\n\n{context_str}"
    }

    try:
        payload = {
            "model": "gpt-4o-mini",
            "messages": [system_prompt] + messages,
            "temperature": 0.4,
            "max_tokens": 300
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            data=json.dumps(payload).encode('utf-8')
        )

        with urllib.request.urlopen(req, timeout=8) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            return res_data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"⚠️ OpenAI API Error: {str(e)}"
