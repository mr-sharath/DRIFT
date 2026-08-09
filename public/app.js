// app.js - Waynex Control Center Interactivity Engine

let map = null;
let currentTemplateData = null;
let mapMarkers = [];
let routePolylines = [];
let vehicleTruckMarkers = [];
let hazardMarkers = [];
let activePerturbations = [];
let userOpenAIApiKey = localStorage.getItem('waynex_openai_api_key') || '';
let chatMessages = [];

let lastBenchmarkData = null;
let currentMapViewMode = 'ortools'; // 'waynex', 'ortools'

const ROUTE_COLORS = ['#0f766e', '#2563eb', '#d97706', '#7c3aed', '#db2777', '#6b7280'];

document.addEventListener('DOMContentLoaded', () => {
    initMap();
    loadCityTemplate('bengaluru');

    if (userOpenAIApiKey) {
        document.getElementById('apiKeyInput').value = userOpenAIApiKey;
    }
});

// View Switching Logic
function switchView(viewId) {
    document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(el => el.classList.remove('active'));

    document.getElementById(viewId).classList.add('active');
    const navLink = document.querySelector(`[data-view="${viewId}"]`);
    if(navLink) navLink.classList.add('active');

    // Close mobile menu if open
    const navLinks = document.getElementById('navLinks');
    if (navLinks) navLinks.classList.remove('open');

    if (viewId === 'view-dispatch' && map) {
        setTimeout(() => map.invalidateSize(), 100);
    }
}

// Mobile Navigation
function toggleMobileMenu() {
    const navLinks = document.getElementById('navLinks');
    if (navLinks) navLinks.classList.toggle('open');
}

// Settings Modal
function toggleSettingsModal() {
    const modal = document.getElementById('settingsModal');
    if (modal) modal.classList.toggle('hidden');
}

function closeSettingsModal(event) {
    if (event.target.classList.contains('settings-modal-overlay')) {
        toggleSettingsModal();
    }
}

function saveApiKey() {
    userOpenAIApiKey = document.getElementById('apiKeyInput').value.trim();
    localStorage.setItem('waynex_openai_api_key', userOpenAIApiKey);
}

// Map Initialization — Light Carto Positron tiles
function initMap() {
    map = L.map('map', {
        zoomControl: true,
        attributionControl: false
    }).setView([12.9716, 77.5946], 12);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 19,
        subdomains: 'abcd'
    }).addTo(map);

    map.on('click', (e) => {
        const clickedLat = roundCoord(e.latlng.lat);
        const clickedLng = roundCoord(e.latlng.lng);
        
        const weightInput = prompt("Enter Package Weight in kg (Capacities: EV Van = 500kg, Heavy Diesel = 1200kg):", "150");
        const weight = weightInput ? parseInt(weightInput) || 150 : 150;
        
        addPinAtCoordinates(clickedLat, clickedLng, `Custom Stop #${currentTemplateData ? currentTemplateData.deliveries.length + 1 : 1}`, weight);
    });
}

function roundCoord(val) {
    return Math.round(val * 10000) / 10000;
}

// City Card Selection (replaces dropdown)
function selectCity(cityKey) {
    document.querySelectorAll('.city-card').forEach(c => {
        c.classList.remove('active');
        c.setAttribute('aria-selected', 'false');
    });
    const selected = document.querySelector(`.city-card[data-city="${cityKey}"]`);
    if (selected) {
        selected.classList.add('active');
        selected.setAttribute('aria-selected', 'true');
    }
    loadCityTemplate(cityKey);
}

async function loadCityTemplate(cityKey) {
    try {
        const response = await fetch(`/api/templates/${cityKey}`);
        if (!response.ok) throw new Error('Failed to load city template');
        
        currentTemplateData = await response.json();
        activePerturbations = [];
        clearHazardMarkers();
        hideAlert();

        map.setView([currentTemplateData.center.lat, currentTemplateData.center.lng], currentTemplateData.zoom);

        renderMapMarkers();
        renderVehicleList();
        
        const btn = document.getElementById('btnRunEngine');
        if (btn && btn.innerHTML.includes('View Benchmarks')) {
            btn.innerHTML = '<i class="fa-solid fa-play"></i> Run Engine';
            btn.onclick = runBenchmark;
        }
        
        addStatusUpdate("system", `City Hub switched to ${currentTemplateData.name}. Initializing routes...`);
        runBenchmark(false);

    } catch (error) {
        console.error('Error loading template:', error);
    }
}

function renderMapMarkers() {
    mapMarkers.forEach(m => map.removeLayer(m));
    mapMarkers = [];

    if (!currentTemplateData) return;

    // Depots — teal icon
    currentTemplateData.depots.forEach(depot => {
        const depotIcon = L.divIcon({
            className: 'custom-map-icon depot-icon',
            html: `<div style="background:#0f766e; width:32px; height:32px; border-radius:10px; display:flex; align-items:center; justify-content:center; color:#fff; font-weight:bold; font-size:16px; box-shadow:0 2px 8px rgba(15,118,110,0.4);"><i class="fa-solid fa-warehouse"></i></div>`,
            iconSize: [32, 32]
        });
        const marker = L.marker([depot.lat, depot.lng], { icon: depotIcon })
            .bindPopup(`<b>${depot.name}</b><br>Primary Distribution Center`)
            .addTo(map);
        mapMarkers.push(marker);
    });

    // Deliveries — white pins with teal border
    document.getElementById('deliveryCount').innerText = currentTemplateData.deliveries.length;
    currentTemplateData.deliveries.forEach((del, idx) => {
        const delIcon = L.divIcon({
            className: 'custom-map-icon delivery-icon',
            html: `<div style="background:#ffffff; width:26px; height:26px; border-radius:50%; display:flex; align-items:center; justify-content:center; color:#1a1a2e; font-size:11px; font-weight:bold; box-shadow:0 2px 6px rgba(0,0,0,0.15); border:2px solid #0f766e;">${idx + 1}</div>`,
            iconSize: [26, 26]
        });
        const marker = L.marker([del.lat, del.lng], { icon: delIcon })
            .bindPopup(`<b>#${idx + 1} ${del.name}</b><br>Weight: <strong>${del.demand} kg</strong><br><button style="margin-top:6px; padding:4px 10px; font-size:11px; background:#dc2626; color:#fff; border:none; border-radius:6px; cursor:pointer;" onclick="removeDeliveryPin('${del.id}')"><i class="fa-solid fa-trash"></i> Delete</button>`)
            .addTo(map);
        mapMarkers.push(marker);
    });
}

function renderVehicleList(vehicleLoads = {}) {
    const listEl = document.getElementById('vehicleList');
    listEl.innerHTML = currentTemplateData.vehicles.map((v, idx) => {
        const color = ROUTE_COLORS[idx % ROUTE_COLORS.length];
        const loadedWeight = vehicleLoads[v.id] || 0;
        const maxCap = v.capacity;
        const pct = Math.min(Math.round((loadedWeight / maxCap) * 100), 100);
        
        return `
            <div class="vehicle-card" style="border-left-color: ${color};">
                <div class="v-header">
                    <span style="color:${color}; font-weight:600;">${v.name}</span>
                    <span style="color:${pct > 90 ? 'var(--status-danger)' : 'var(--status-success)'}">${pct}%</span>
                </div>
                <div class="v-details">Load: <strong>${loadedWeight} / ${maxCap} kg</strong></div>
            </div>
        `;
    }).join('');
}

// Live Status Feed Updater
function addStatusUpdate(type, msg) {
    const feed = document.getElementById('statusFeed');
    const timeStr = new Date().toLocaleTimeString([], { hour12: false });
    const html = `
        <div class="status-item ${type}">
            <div class="status-time">[${timeStr}]</div>
            <div class="status-msg">${msg}</div>
        </div>
    `;
    feed.innerHTML += html;
    feed.scrollTop = feed.scrollHeight;
}

// Execute Dual Benchmark
async function runBenchmark(appendStatus = true) {
    if (!currentTemplateData) return;

    const btn = document.getElementById('btnRunEngine');
    
    if (btn && btn.innerHTML.includes('View Benchmarks')) {
        switchView('view-analytics');
        return;
    }

    const jobId = 'JOB-' + Math.random().toString(36).substr(2, 6).toUpperCase();
    
    if (btn) {
        btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing...`;
        btn.disabled = true;
    }

    if (appendStatus) addStatusUpdate("system", `Running Dual Benchmark Routing Engine... (Job ID: ${jobId})`);

    try {
        const orToolsTimeLimit = 1; // Fixed at 1 second (slider removed)

        const payload = {
            city_key: currentTemplateData.name,
            depots: currentTemplateData.depots,
            deliveries: currentTemplateData.deliveries,
            vehicles: currentTemplateData.vehicles,
            perturbations: activePerturbations,
            ortools_time_limit: orToolsTimeLimit,
            openai_api_key: userOpenAIApiKey
        };

        const res = await fetch('/api/benchmark', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        lastBenchmarkData = await res.json();
        lastBenchmarkData.job_id = jobId;

        updateTelematicsUI(lastBenchmarkData);
        updateMapRoutesByViewMode();
        updateVehicleLoadsByViewMode();
        
        if (appendStatus) {
            const speedup = lastBenchmarkData.efficiency_gain.latency_speedup;
            addStatusUpdate("success", `Routing complete. Waynex Neural speedup: <b>${speedup}</b>.`);
        }
        
        if (btn) {
            btn.innerHTML = `<i class="fa-solid fa-chart-column"></i> View Benchmarks`;
            btn.disabled = false;
        }
        
        const badge = document.getElementById('benchmarkJobIdBadge');
        const badgeText = document.getElementById('benchmarkJobIdText');
        if (badge && badgeText) {
            badgeText.innerText = jobId;
            badge.classList.remove('hidden');
        }

    } catch (error) {
        console.error('Benchmark error:', error);
        addStatusUpdate("warning", "Routing Engine encountered an error.");
        if (btn) {
            btn.innerHTML = `<i class="fa-solid fa-play"></i> Run Engine`;
            btn.disabled = false;
        }
    }
}

function switchMapViewMode(mode) {
    currentMapViewMode = mode;
    
    document.getElementById('btnViewWaynex').classList.toggle('active', mode === 'waynex');
    document.getElementById('btnViewORTools').classList.toggle('active', mode === 'ortools');

    updateMapRoutesByViewMode();
    updateVehicleLoadsByViewMode();
}

function updateVehicleLoadsByViewMode() {
    if (!lastBenchmarkData || !currentTemplateData) return;

    let routesToUse = [];
    if (currentMapViewMode === 'ortools') {
        routesToUse = lastBenchmarkData.or_tools_baseline.routes;
    } else {
        routesToUse = lastBenchmarkData.waynex_neural_engine.routes;
    }

    const vehicleLoads = {};
    routesToUse.forEach(r => {
        let sumWeight = 0;
        r.route_node_indices.forEach(n_idx => {
            if (n_idx > 0 && currentTemplateData.deliveries[n_idx - 1]) {
                sumWeight += currentTemplateData.deliveries[n_idx - 1].demand;
            }
        });
        vehicleLoads[r.vehicle_id] = sumWeight;
    });

    renderVehicleList(vehicleLoads);
}

function getVehicleColor(vehicleId) {
    if (!currentTemplateData) return ROUTE_COLORS[0];
    const vIdx = currentTemplateData.vehicles.findIndex(v => v.id === vehicleId);
    return ROUTE_COLORS[vIdx >= 0 ? vIdx % ROUTE_COLORS.length : 0];
}

function updateMapRoutesByViewMode() {
    if (!lastBenchmarkData) return;

    routePolylines.forEach(p => map.removeLayer(p));
    vehicleTruckMarkers.forEach(t => map.removeLayer(t));
    routePolylines = [];
    vehicleTruckMarkers = [];

    const waynexRoutes = lastBenchmarkData.waynex_neural_engine.routes;
    const orRoutes = lastBenchmarkData.or_tools_baseline.routes;

    if (currentMapViewMode === 'waynex') {
        waynexRoutes.forEach(r => {
            if (r.polyline && r.polyline.length > 0) {
                const color = getVehicleColor(r.vehicle_id);
                const polyline = L.polyline(r.polyline, {
                    color: color,
                    weight: 4,
                    opacity: 0.9,
                    className: 'route-polyline',
                    smoothFactor: 1
                }).addTo(map);
                routePolylines.push(polyline);

                const midPtIdx = Math.floor(r.polyline.length / 2);
                const truckPt = r.polyline[midPtIdx] || r.polyline[0];
                const truckIcon = L.divIcon({
                    className: 'vehicle-truck-container',
                    html: `<div class="vehicle-truck-icon" style="background:${color};"><i class="fa-solid fa-truck-fast"></i></div>`,
                    iconSize: [28, 28]
                });
                const truckMarker = L.marker(truckPt, { icon: truckIcon })
                    .bindPopup(`<b>⚡ Waynex Neural: ${r.vehicle_name}</b><br>Distance: ${r.distance_km} km<br>Duration: ${r.duration_min} min`)
                    .addTo(map);
                vehicleTruckMarkers.push(truckMarker);
            }
        });
    }

    if (currentMapViewMode === 'ortools') {
        orRoutes.forEach(r => {
            if (r.polyline && r.polyline.length > 0) {
                const color = getVehicleColor(r.vehicle_id);
                const polyline = L.polyline(r.polyline, {
                    color: color,
                    weight: 3,
                    dashArray: '6, 6',
                    opacity: 0.85
                }).addTo(map);
                routePolylines.push(polyline);
                
                const midPtIdx = Math.floor(r.polyline.length / 2);
                const truckPt = r.polyline[midPtIdx] || r.polyline[0];
                const truckIcon = L.divIcon({
                    className: 'vehicle-truck-container',
                    html: `<div class="vehicle-truck-icon" style="background:${color};"><i class="fa-solid fa-truck"></i></div>`,
                    iconSize: [28, 28]
                });
                const truckMarker = L.marker(truckPt, { icon: truckIcon })
                    .bindPopup(`<b>📊 Google OR-Tools: ${r.vehicle_name}</b><br>Distance: ${r.distance_km} km<br>Duration: ${r.duration_min} min`)
                    .addTo(map);
                vehicleTruckMarkers.push(truckMarker);
            }
        });
    }
}

function updateTelematicsUI(data) {
    const waynex = data.waynex_neural_engine;
    const ortools = data.or_tools_baseline;
    const eff = data.efficiency_gain;

    document.getElementById('kpiDistOR').innerText = `${ortools.total_distance_km} km`;
    document.getElementById('kpiDistWaynex').innerText = `${waynex.total_distance_km} km`;
    const distDiff = (waynex.total_distance_km - ortools.total_distance_km).toFixed(2);
    document.getElementById('kpiDistDelta').innerText = `${distDiff > 0 ? '+' : ''}${distDiff} km`;
    document.getElementById('kpiDistDelta').className = distDiff > 0 ? 'text-amber fw-bold' : (distDiff < 0 ? 'text-emerald fw-bold' : 'text-muted fw-bold');

    document.getElementById('kpiSpanOR').innerText = `${ortools.makespan_min} min`;
    document.getElementById('kpiSpanWaynex').innerText = `${waynex.makespan_min} min`;
    const spanDiff = (waynex.makespan_min - ortools.makespan_min).toFixed(2);
    document.getElementById('kpiSpanDelta').innerText = `${spanDiff > 0 ? '+' : ''}${spanDiff} min`;
    document.getElementById('kpiSpanDelta').className = spanDiff > 0 ? 'text-amber fw-bold' : (spanDiff < 0 ? 'text-emerald fw-bold' : 'text-muted fw-bold');

    document.getElementById('kpiUtilOR').innerText = `${ortools.utilization_pct}%`;
    document.getElementById('kpiUtilWaynex').innerText = `${waynex.utilization_pct}%`;
    const utilDiff = (waynex.utilization_pct - ortools.utilization_pct).toFixed(1);
    document.getElementById('kpiUtilDelta').innerText = `${utilDiff > 0 ? '+' : ''}${utilDiff}%`;
    document.getElementById('kpiUtilDelta').className = 'text-teal fw-bold';

    document.getElementById('kpiFuelOR').innerText = `${ortools.fuel_used_gal} gal`;
    document.getElementById('kpiFuelWaynex').innerText = `${waynex.fuel_used_gal} gal`;
    const fuelDiff = (waynex.fuel_used_gal - ortools.fuel_used_gal).toFixed(2);
    document.getElementById('kpiFuelDelta').innerText = `${fuelDiff > 0 ? '+' : ''}${fuelDiff} gal`;
    document.getElementById('kpiFuelDelta').className = fuelDiff > 0 ? 'text-amber fw-bold' : (fuelDiff < 0 ? 'text-emerald fw-bold' : 'text-muted fw-bold');

    document.getElementById('kpiLatOR').innerText = `${ortools.execution_time_ms} ms`;
    document.getElementById('kpiLatWaynex').innerText = `${waynex.execution_time_ms} ms`;
    document.getElementById('kpiLatDelta').innerText = eff.latency_speedup;
}

// Perturbations
async function triggerPerturbation(eventType) {
    const pertRes = await fetch('/api/perturb', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            event_type: eventType,
            affected_location_name: currentTemplateData.name,
            city_key: currentTemplateData.name
        })
    });

    const pertData = await pertRes.json();
    
    activePerturbations.push({ type: eventType, duration_multiplier: 2.0 });

    dropPulsingHazardMarker(currentTemplateData.center, eventType);
    showAlert(pertData.message);
    addStatusUpdate("warning", `Perturbation Injected: ${eventType.toUpperCase()} in ${currentTemplateData.name}. Initiating Rerouting...`);
    
    runBenchmark(true); 
}

function dropPulsingHazardMarker(center, eventType) {
    const icons = {
        festival: 'fa-gopuram',
        accident: 'fa-car-burst',
        rain: 'fa-cloud-showers-heavy'
    };

    const hazardIcon = L.divIcon({
        className: 'hazard-marker-wrapper',
        html: `<div class="hazard-pulse-icon"><i class="fa-solid ${icons[eventType] || 'fa-triangle-exclamation'}"></i></div>`,
        iconSize: [30, 30]
    });

    const marker = L.marker([center.lat + (Math.random() - 0.5) * 0.04, center.lng + (Math.random() - 0.5) * 0.04], { icon: hazardIcon })
        .bindPopup(`<b>Disruption Active</b><br>${eventType.toUpperCase()}`)
        .addTo(map);
    
    hazardMarkers.push(marker);
}

function clearHazardMarkers() {
    hazardMarkers.forEach(h => map.removeLayer(h));
    hazardMarkers = [];
}

function showAlert(msg) {
    const overlay = document.getElementById('alertOverlay');
    document.getElementById('alertMessage').innerText = msg;
    overlay.classList.remove('hidden');
}

function hideAlert() {
    document.getElementById('alertOverlay').classList.add('hidden');
}

// Pin Dropping / Geocoding
function addPinAtCoordinates(lat, lng, name, demandWeight = 150) {
    if (!currentTemplateData) return;

    currentTemplateData.deliveries.push({
        id: `del_custom_${Date.now()}`,
        name: name,
        lat: lat,
        lng: lng,
        demand: demandWeight,
        time_window: [9, 17],
        priority: 'high'
    });

    renderMapMarkers();
    addStatusUpdate("system", `New Delivery Pin dropped at ${name} (${demandWeight} kg). Awaiting Run Engine...`);
    
    const btn = document.getElementById('btnRunEngine');
    if (btn && btn.innerHTML.includes('View Benchmarks')) {
        btn.innerHTML = '<i class="fa-solid fa-play"></i> Run Engine';
        btn.onclick = runBenchmark;
    }
}

function removeDeliveryPin(delId) {
    if (!currentTemplateData) return;
    currentTemplateData.deliveries = currentTemplateData.deliveries.filter(d => d.id !== delId);
    
    map.closePopup();
    
    renderMapMarkers();
    addStatusUpdate("system", `Delivery Pin removed. Awaiting Run Engine...`);
    
    const btn = document.getElementById('btnRunEngine');
    if (btn && btn.innerHTML.includes('View Benchmarks')) {
        btn.innerHTML = '<i class="fa-solid fa-play"></i> Run Engine';
        btn.onclick = runBenchmark;
    }
}

async function handleAutocompleteInput(event) {
    const query = event.target.value.trim();
    const dropdown = document.getElementById('autocompleteDropdown');

    if (!query || query.length < 2) {
        dropdown.classList.add('hidden');
        return;
    }

    try {
        const cityContext = currentTemplateData ? currentTemplateData.name : "";
        const res = await fetch('/api/autocomplete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, city_context: cityContext })
        });

        const data = await res.json();
        if (data.suggestions && data.suggestions.length > 0) {
            dropdown.innerHTML = data.suggestions.map(s => `
                <div class="autocomplete-item" onclick="selectAutocompleteItem(${s.lat}, ${s.lng}, '${s.name.replace(/'/g, "\\'")}')">
                    <i class="fa-solid fa-location-dot text-teal"></i> ${s.name}
                </div>
            `).join('');
            dropdown.classList.remove('hidden');
        } else {
            dropdown.classList.add('hidden');
        }
    } catch (e) { console.error(e); }
}

function selectAutocompleteItem(lat, lng, name) {
    document.getElementById('addressInput').value = name;
    document.getElementById('autocompleteDropdown').classList.add('hidden');
    addPinAtCoordinates(lat, lng, name, 150);
    map.flyTo([lat, lng], 13);
}

async function addCustomAddress() {
    const inputEl = document.getElementById('addressInput');
    const query = inputEl.value.trim();
    if (!query) return;

    try {
        const cityContext = currentTemplateData ? currentTemplateData.name : "";
        const res = await fetch('/api/geocode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, city_context: cityContext })
        });
        
        if (res.ok) {
            const data = await res.json();
            const cleanName = data.display_name.split(',')[0];
            addPinAtCoordinates(data.lat, data.lng, cleanName, 150);
            inputEl.value = '';
            map.flyTo([data.lat, data.lng], 13);
        } else {
            alert(`Address '${query}' could not be located. Try clicking directly on the map!`);
        }
    } catch (e) { console.error(e); }
}

function handleGeocodeKey(event) { if (event.key === 'Enter') addCustomAddress(); }

// Conversational AI Copilot
function toggleCopilotChat() {
    const chatWindow = document.getElementById('copilotChatWindow');
    chatWindow.classList.toggle('hidden');
}

function sendChatPrompt(promptText) {
    document.getElementById('chatInput').value = promptText;
    submitChat();
}

function handleChatKey(event) {
    if (event.key === 'Enter') submitChat();
}

async function submitChat() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (!text) return;

    appendChatMessage('user', text);
    input.value = '';

    chatMessages.push({ role: 'user', content: text });

    appendChatMessage('ai', '...', 'loading-msg');

    try {
        const payload = {
            messages: chatMessages,
            benchmark_context: lastBenchmarkData || {},
            openai_api_key: userOpenAIApiKey
        };

        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        
        const loading = document.getElementById('loading-msg');
        if (loading) loading.remove();

        appendChatMessage('ai', data.reply);
        chatMessages.push({ role: 'assistant', content: data.reply });

    } catch (e) {
        console.error(e);
        const loading = document.getElementById('loading-msg');
        if (loading) loading.remove();
        appendChatMessage('ai', 'Error connecting to Waynex Copilot API.');
    }
}

function appendChatMessage(role, text, id = null) {
    const body = document.getElementById('chatBody');
    
    let formatted = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');

    const html = `
        <div class="chat-msg ${role}" ${id ? `id="${id}"` : ''}>
            <div class="msg-content">${formatted}</div>
        </div>
    `;
    body.innerHTML += html;
    body.scrollTop = body.scrollHeight;
}
