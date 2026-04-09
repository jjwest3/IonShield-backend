"""
IonShield Backend — Space Weather Operational Intelligence API
Serves KML/GeoJSON overlays for ATAK-CIV + location/route risk analysis
"""

from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx
import asyncio
import math
import json
from datetime import datetime, timezone

app = FastAPI(title="IonShield API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
# DATA CACHE
# ═══════════════════════════════════════════════════════════════

cache = {
    "kp": None, "xray": None, "wind": None, "proton": None,
    "last_fetch": None, "fetch_source": "startup",
}

NOAA_ENDPOINTS = {
    "kp": "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json",
    "xray": "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json",
    "wind": "https://services.swpc.noaa.gov/json/solar-wind/plasma-2-hour.json",
    "proton": "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-hour.json",
}

FALLBACK = {"kp": 2.0, "xray_flux": 3e-7, "wind_speed": 400, "proton_flux": 0.5}

BASES = [
    {"name": "Thule AFB, Greenland", "lat": 76.5, "lon": -68.7},
    {"name": "Clear SFS, Alaska", "lat": 64.3, "lon": -149.2},
    {"name": "Schriever SFB, CO", "lat": 38.8, "lon": -104.5},
    {"name": "Vandenberg SFB, CA", "lat": 34.7, "lon": -120.6},
    {"name": "Cape Canaveral, FL", "lat": 28.5, "lon": -80.6},
    {"name": "Diego Garcia", "lat": -7.3, "lon": 72.4},
    {"name": "Ramstein AB, Germany", "lat": 49.4, "lon": 7.6},
    {"name": "Kadena AB, Japan", "lat": 26.4, "lon": 127.8},
    {"name": "Camp Humphreys, ROK", "lat": 36.9, "lon": 127.0},
    {"name": "Al Udeid AB, Qatar", "lat": 25.1, "lon": 51.3},
]


# ═══════════════════════════════════════════════════════════════
# NOAA DATA FETCHER
# ═══════════════════════════════════════════════════════════════

async def fetch_noaa():
    """Fetch all NOAA endpoints with per-feed resilience."""
    async with httpx.AsyncClient(timeout=8.0) as client:
        for key, url in NOAA_ENDPOINTS.items():
            try:
                r = await client.get(url)
                r.raise_for_status()
                cache[key] = r.json()
            except Exception as e:
                print(f"[NOAA] {key} fetch failed: {e}")

    cache["last_fetch"] = datetime.now(timezone.utc).isoformat()
    cache["fetch_source"] = "live" if cache["kp"] else "fallback"


def get_kp() -> float:
    """Extract current Kp from cached data."""
    try:
        data = cache["kp"]
        if data and len(data) > 0:
            return float(data[-1].get("kp_index", data[-1].get("kp", FALLBACK["kp"])))
    except:
        pass
    return FALLBACK["kp"]


def get_xray_class() -> str:
    """Extract current X-ray class."""
    try:
        data = cache["xray"]
        if data and len(data) > 0:
            flux = float(data[-1].get("flux", FALLBACK["xray_flux"]))
            if flux >= 1e-4: return "X"
            if flux >= 1e-5: return "M"
            if flux >= 1e-6: return "C"
            if flux >= 1e-7: return "B"
            return "A"
    except:
        pass
    return "B"


def get_wind_speed() -> float:
    try:
        data = cache["wind"]
        if data and len(data) > 0:
            for entry in reversed(data):
                spd = entry.get("speed")
                if spd and float(spd) > 0:
                    return float(spd)
    except:
        pass
    return FALLBACK["wind_speed"]


# ═══════════════════════════════════════════════════════════════
# PHYSICS / RISK ENGINE
# ═══════════════════════════════════════════════════════════════

def lat_zone(lat: float) -> dict:
    """Classify latitude zone and return multiplier."""
    a = abs(lat)
    if a > 70: return {"zone": "polar", "multiplier": 1.8, "color": "#F59E0B"}
    if a > 55: return {"zone": "sub-auroral", "multiplier": 1.3, "color": "#F59E0B"}
    if a > 25: return {"zone": "mid-latitude", "multiplier": 1.0, "color": "#10B981"}
    return {"zone": "equatorial", "multiplier": 1.4, "color": "#10B981"}


def compute_risk(lat: float, lon: float, kp: float = None) -> dict:
    """Compute full operational risk assessment for a lat/lon."""
    if kp is None:
        kp = get_kp()

    zone = lat_zone(lat)
    m = zone["multiplier"]

    # S4 scintillation (Basu 1988 simplified)
    a = abs(lat)
    if a > 60:
        s4 = min(1.0, (0.1 + kp * 0.08) * (1.5 if _is_night(lon) else 1.0))
    elif a < 20:
        s4 = min(1.0, (0.05 + kp * 0.04) * (1.5 if _is_night(lon) else 1.0))
    else:
        s4 = min(1.0, (0.02 + kp * 0.02) * (1.5 if _is_night(lon) else 1.0))

    # GPS error
    base_err = 1.5 + s4 * 25
    storm_add = max(0, (kp - 5) * 2) if kp > 5 else 0
    gps_error = round((base_err + storm_add) * m, 1)

    # HF absorption (CCIR-888 simplified)
    xray = get_xray_class()
    xray_mult = {"X": 8, "M": 4, "C": 2, "B": 0.5, "A": 0.1}.get(xray, 0.5)
    lt = (datetime.now(timezone.utc).hour + lon / 15) % 24
    is_day = 6 < lt < 18
    hf_abs = round(xray_mult * (0.8 if is_day else 0.1) * m, 1)
    hf_blackout = min(1.0, round(hf_abs / 10, 2))

    # SATCOM (ITU-R P.531 simplified)
    satcom_loss = round(s4 * 12 * m, 1)

    # Radar bias
    radar_bias = round((kp * 0.8 + s4 * 3) * m, 1)

    # Risk score (0-99)
    score = min(99, round(
        min(30, kp * 3.3) +
        min(25, s4 * 50) +
        min(20, hf_abs * 2.5) +
        min(24, gps_error * 0.8)
    ))

    if score < 20:
        level, rec = "NOMINAL", "CLEAR — All systems nominal. Standard operations."
    elif score < 40:
        level, rec = "ELEVATED", "CAUTION — GPS accuracy degraded. Monitor for escalation. Consider backup navigation."
    elif score < 60:
        level, rec = "DEGRADED", "CAUTION — Significant degradation. Delay non-critical GPS-dependent ops. Activate backup comms."
    else:
        level, rec = "SEVERE", "ABORT — Major storm. GPS unreliable. HF likely blacked out. Postpone GPS-dependent operations."

    return {
        "lat": lat,
        "lon": lon,
        "zone": zone["zone"],
        "zone_multiplier": m,
        "kp_current": round(kp, 1),
        "assessment": {
            "gps_error_m": gps_error,
            "gps_error_range": [round(gps_error * 0.75, 1), round(gps_error * 1.5, 1)],
            "hf_absorption_db": hf_abs,
            "hf_blackout_probability": hf_blackout,
            "satcom_loss_db": satcom_loss,
            "radar_bias_m": radar_bias,
            "s4_index": round(s4, 3),
            "risk_score": score,
            "risk_level": level,
            "recommendation": rec,
        },
        "sources": {
            "kp": "[MEASURED] NOAA SWPC planetary_k_index_1m",
            "s4": "[MODELED] Basu 1988 mid-latitude model",
            "gps_error": "[MODELED] IonShield physics engine",
            "hf": "[MODELED] CCIR-888 simplified",
            "satcom": "[MODELED] ITU-R P.531 simplified",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_age_seconds": _data_age(),
    }


def _is_night(lon: float) -> bool:
    lt = (datetime.now(timezone.utc).hour + lon / 15) % 24
    return lt < 6 or lt > 20


def _data_age() -> int:
    if cache["last_fetch"]:
        try:
            dt = datetime.fromisoformat(cache["last_fetch"])
            return int((datetime.now(timezone.utc) - dt).total_seconds())
        except:
            pass
    return 9999


# ═══════════════════════════════════════════════════════════════
# KML GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_kml() -> str:
    """Generate ATAK-compatible KML with risk zones and military bases."""
    kp = get_kp()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    zones = [
        ("Polar North (>70°N)", 70, 90, "polar"),
        ("Sub-Auroral North (55-70°N)", 55, 70, "sub-auroral"),
        ("Mid-Latitude North (25-55°N)", 25, 55, "mid-latitude"),
        ("Equatorial (25°S-25°N)", -25, 25, "equatorial"),
        ("Mid-Latitude South (25-55°S)", -55, -25, "mid-latitude"),
        ("Sub-Auroral South (55-70°S)", -70, -55, "sub-auroral"),
        ("Polar South (<70°S)", -90, -70, "polar"),
    ]

    # KML color format: aabbggrr (alpha, blue, green, red)
    zone_styles = {
        "polar":       {"poly": "26009ef5", "line": "66009ef5"},  # amber
        "sub-auroral":  {"poly": "1a009ef5", "line": "4d009ef5"},  # light amber
        "mid-latitude": {"poly": "0d81b910", "line": "3381b910"},  # green
        "equatorial":   {"poly": "1081b910", "line": "4081b910"},  # light green
    }

    # Scale colors based on Kp
    if kp >= 7:
        zone_styles["mid-latitude"] = {"poly": "26009ef5", "line": "66009ef5"}
        zone_styles["equatorial"] = {"poly": "260000ee", "line": "660000ee"}
        zone_styles["sub-auroral"] = {"poly": "330000ee", "line": "660000ee"}
        zone_styles["polar"] = {"poly": "400000ee", "line": "990000ee"}
    elif kp >= 5:
        zone_styles["sub-auroral"] = {"poly": "260000ee", "line": "660000ee"}
        zone_styles["polar"] = {"poly": "330000ee", "line": "990000ee"}

    styles_kml = ""
    for ztype, colors in zone_styles.items():
        styles_kml += f'''
  <Style id="{ztype}">
    <PolyStyle><color>{colors["poly"]}</color><outline>1</outline></PolyStyle>
    <LineStyle><color>{colors["line"]}</color><width>1</width></LineStyle>
  </Style>'''

    zones_kml = ""
    for name, lat_min, lat_max, ztype in zones:
        risk = compute_risk((lat_min + lat_max) / 2, 0, kp)
        a = risk["assessment"]
        desc = (
            f"<b>IonShield Assessment</b><br/>"
            f"Zone: {risk['zone'].title()}<br/>"
            f"GPS Error: {a['gps_error_m']}m ({a['gps_error_range'][0]}-{a['gps_error_range'][1]}m)<br/>"
            f"HF Absorption: {a['hf_absorption_db']} dB<br/>"
            f"SATCOM Loss: {a['satcom_loss_db']} dB<br/>"
            f"S4 Index: {a['s4_index']}<br/>"
            f"Risk: <b>{a['risk_level']}</b><br/><br/>"
            f"Kp: {kp} | {now}<br/>"
            f"Source: NOAA SWPC + IonShield"
        )
        zones_kml += f'''
    <Placemark>
      <name>{name} — GPS: {a["gps_error_m"]}m | {a["risk_level"]}</name>
      <description><![CDATA[{desc}]]></description>
      <styleUrl>#{ztype}</styleUrl>
      <Polygon><outerBoundaryIs><LinearRing>
        <coordinates>-180,{lat_min},0 -180,{lat_max},0 180,{lat_max},0 180,{lat_min},0 -180,{lat_min},0</coordinates>
      </LinearRing></outerBoundaryIs></Polygon>
    </Placemark>'''

    bases_kml = ""
    for base in BASES:
        risk = compute_risk(base["lat"], base["lon"], kp)
        a = risk["assessment"]
        desc = (
            f"<b>{base['name']}</b><br/>"
            f"Zone: {risk['zone'].title()} ({risk['zone_multiplier']}x)<br/><br/>"
            f"GPS Error: <b>{a['gps_error_m']}m</b><br/>"
            f"HF Absorption: {a['hf_absorption_db']} dB<br/>"
            f"HF Blackout Prob: {int(a['hf_blackout_probability']*100)}%<br/>"
            f"SATCOM Loss: {a['satcom_loss_db']} dB<br/>"
            f"Radar Bias: {a['radar_bias_m']}m<br/>"
            f"S4: {a['s4_index']}<br/>"
            f"Score: {a['risk_score']}/99<br/>"
            f"Risk: <b>{a['risk_level']}</b><br/><br/>"
            f"<i>{a['recommendation']}</i><br/><br/>"
            f"Kp: {kp} | {now}"
        )
        bases_kml += f'''
    <Placemark>
      <name>{base["name"]} — {a["risk_level"]}</name>
      <description><![CDATA[{desc}]]></description>
      <Point><coordinates>{base["lon"]},{base["lat"]},0</coordinates></Point>
    </Placemark>'''

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>IonShield Ionospheric Risk</name>
  <description>Real-time ionospheric risk zones and military base assessments. Updated every 10 minutes from live NOAA SWPC data. Kp: {kp}. Generated: {now}</description>
  {styles_kml}
  <Folder>
    <name>Ionospheric Risk Zones</name>
    {zones_kml}
  </Folder>
  <Folder>
    <name>Military Installations</name>
    {bases_kml}
  </Folder>
</Document>
</kml>'''


# ═══════════════════════════════════════════════════════════════
# GEOJSON GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_geojson() -> dict:
    kp = get_kp()
    now = datetime.now(timezone.utc).isoformat()
    features = []

    zones = [
        ("Polar North", 70, 90), ("Sub-Auroral North", 55, 70),
        ("Mid-Latitude North", 25, 55), ("Equatorial", -25, 25),
        ("Mid-Latitude South", -55, -25), ("Sub-Auroral South", -70, -55),
        ("Polar South", -90, -70),
    ]

    for name, lat_min, lat_max in zones:
        risk = compute_risk((lat_min + lat_max) / 2, 0, kp)
        a = risk["assessment"]
        fill = "#EF4444" if a["risk_level"] == "SEVERE" else "#F97316" if a["risk_level"] == "DEGRADED" else "#F59E0B" if a["risk_level"] == "ELEVATED" else "#10B981"
        features.append({
            "type": "Feature",
            "properties": {
                "name": name,
                "zone": risk["zone"],
                "risk_level": a["risk_level"],
                "gps_error_m": a["gps_error_m"],
                "hf_absorption_db": a["hf_absorption_db"],
                "satcom_loss_db": a["satcom_loss_db"],
                "s4_index": a["s4_index"],
                "risk_score": a["risk_score"],
                "fill": fill,
                "fill-opacity": 0.12,
                "stroke": fill,
                "stroke-opacity": 0.4,
                "stroke-width": 1,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-180, lat_min], [-180, lat_max], [180, lat_max], [180, lat_min], [-180, lat_min]]]
            }
        })

    for base in BASES:
        risk = compute_risk(base["lat"], base["lon"], kp)
        a = risk["assessment"]
        features.append({
            "type": "Feature",
            "properties": {
                "name": base["name"],
                "marker-color": "#00D4FF",
                **a,
            },
            "geometry": {"type": "Point", "coordinates": [base["lon"], base["lat"]]}
        })

    return {
        "type": "FeatureCollection",
        "name": "IonShield Ionospheric Risk",
        "generated": now,
        "kp_current": kp,
        "features": features,
    }


# ═══════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    await fetch_noaa()
    asyncio.create_task(refresh_loop())


async def refresh_loop():
    while True:
        await asyncio.sleep(300)  # 5 minutes
        await fetch_noaa()


@app.get("/")
async def root():
    return {
        "name": "IonShield API",
        "version": "2.0",
        "description": "Space weather operational intelligence for ATAK",
        "endpoints": [
            "/overlay/risk.kml",
            "/overlay/risk.geojson",
            "/api/risk/location?lat=38.8&lon=-104.5",
            "/api/risk/route",
            "/api/status",
        ]
    }


@app.get("/overlay/risk.kml")
async def overlay_kml():
    kml = generate_kml()
    return Response(content=kml, media_type="application/vnd.google-earth.kml+xml")


@app.get("/overlay/risk.geojson")
async def overlay_geojson():
    return generate_geojson()


@app.get("/api/risk/location")
async def risk_location(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    return compute_risk(lat, lon)


class Waypoint(BaseModel):
    lat: float
    lon: float
    name: Optional[str] = None

class RouteRequest(BaseModel):
    waypoints: List[Waypoint]
    asset_type: Optional[str] = "GPS_L1"

@app.post("/api/risk/route")
async def risk_route(req: RouteRequest):
    kp = get_kp()
    results = []
    worst_idx = 0
    worst_score = 0

    for i, wp in enumerate(req.waypoints):
        risk = compute_risk(wp.lat, wp.lon, kp)
        a = risk["assessment"]
        entry = {
            "index": i,
            "name": wp.name or f"WP{i}",
            "lat": wp.lat,
            "lon": wp.lon,
            "zone": risk["zone"],
            "gps_error_m": a["gps_error_m"],
            "hf_absorption_db": a["hf_absorption_db"],
            "satcom_loss_db": a["satcom_loss_db"],
            "risk_score": a["risk_score"],
            "risk_level": a["risk_level"],
            "risk_color": "#EF4444" if a["risk_level"] == "SEVERE" else "#F97316" if a["risk_level"] == "DEGRADED" else "#F59E0B" if a["risk_level"] == "ELEVATED" else "#10B981",
        }
        results.append(entry)
        if a["risk_score"] > worst_score:
            worst_score = a["risk_score"]
            worst_idx = i

    worst = results[worst_idx] if results else None
    if worst_score >= 60:
        route_rec = f"NO-GO — Waypoint {worst_idx} ({worst['name']}) at {worst['risk_level']}. GPS error {worst['gps_error_m']}m. Postpone or re-route."
    elif worst_score >= 40:
        route_rec = f"CAUTION — Waypoint {worst_idx} ({worst['name']}) shows degraded conditions. GPS error {worst['gps_error_m']}m. Consider delay or backup nav."
    elif worst_score >= 20:
        route_rec = f"CAUTION — Elevated risk at waypoint {worst_idx} ({worst['name']}). GPS error {worst['gps_error_m']}m. Monitor conditions."
    else:
        route_rec = "GO — All waypoints nominal. Standard operations."

    return {
        "route_summary": {
            "total_waypoints": len(results),
            "worst_waypoint": worst_idx,
            "worst_gps_error_m": worst["gps_error_m"] if worst else 0,
            "max_risk_level": worst["risk_level"] if worst else "NOMINAL",
            "route_recommendation": route_rec,
        },
        "waypoints": results,
        "kp_current": round(kp, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/status")
async def status():
    return {
        "ionshield_version": "2.0",
        "kp_current": round(get_kp(), 1),
        "xray_class": get_xray_class(),
        "solar_wind_km_s": round(get_wind_speed()),
        "global_risk": "NOMINAL" if get_kp() < 4 else "ELEVATED" if get_kp() < 5 else "DEGRADED" if get_kp() < 7 else "SEVERE",
        "data_source": "NOAA SWPC",
        "last_fetch": cache["last_fetch"],
        "data_age_seconds": _data_age(),
        "fetch_source": cache["fetch_source"],
    }
