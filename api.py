"""
Cloud-Based Flight Delay Analytics System using Big Data
Backend Cloud Microservice: FastAPI + Uvicorn
Data Engine: Apache Hadoop HDFS + Apache Spark (PySpark)
Prescriptive Analytics: Dynamic Time-Sensitive Carrier Recommendation Engine
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import time

app = FastAPI(
    title="Cloud-Based Flight Delay Analytics System using Big Data",
    description="Decoupled Cloud Microservice querying PySpark distributed aggregations over 333,675 flight logs in Hadoop HDFS with Dynamic Time-Sensitive Airline Recommendations."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MART_PATH = "flight_analytics_mart.json"
analytics_mart = None

CARRIER_MAP = {
    0: "9E", 1: "AA", 2: "AS", 3: "B6", 4: "DL", 5: "EV",
    6: "F9", 7: "FL", 8: "HA", 9: "MQ", 10: "OO", 11: "UA",
    12: "US", 13: "VX", 14: "WN", 15: "YV"
}

def load_or_generate_mart():
    global analytics_mart
    if not os.path.exists(MART_PATH):
        print(f"--> [INFO] {MART_PATH} not found. Triggering Big Data Analytics generation...")
        try:
            import pyspark_analytics
            pyspark_analytics.run_distributed_aggregation()
        except Exception as e:
            print(f"--> [ERROR] Failed to run distributed aggregation: {e}")
            raise RuntimeError(f"Could not initialize analytics mart: {e}")

    with open(MART_PATH, "r") as f:
        analytics_mart = json.load(f)
    print(f"--> [SUCCESS] Loaded Big Data Analytics Mart ({analytics_mart['metadata']['total_records_processed']:,} HDFS records)")
    return analytics_mart

try:
    load_or_generate_mart()
except Exception as err:
    print(f"--> [WARNING] Analytics mart deferred: {err}")

class FlightData(BaseModel):
    airline: int
    day: int
    time: int
    distance: int

class RecommendationQuery(BaseModel):
    day: int
    time: int
    distance: int = 1000

@app.get("/")
def root():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return root_info()

@app.get("/api/info")
def root_info():
    mart = analytics_mart or {}
    meta = mart.get("metadata", {})
    return {
        "service": "Cloud-Based Flight Delay Analytics System using Big Data",
        "status": "online",
        "cloud_architecture": "Decoupled Microservice (FastAPI + PySpark + Hadoop HDFS)",
        "prescriptive_engine": "Dynamic Time-Sensitive Carrier Recommendation Optimizer",
        "dataset_records_in_hdfs": meta.get("total_records_processed", 333675),
        "cluster_delay_rate": meta.get("cluster_delay_rate", 21.54),
        "cluster_avg_delay_mins": meta.get("cluster_avg_delay_mins", 66.3),
        "docs_url": "/docs"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "analytics_mart_loaded": analytics_mart is not None,
        "engine": "PySpark Big Data Analytics & Prescriptive Recommendation Engine",
        "cluster_status": "active"
    }

@app.get("/cluster-metrics")
def get_cluster_metrics():
    if not analytics_mart:
        load_or_generate_mart()
    return analytics_mart

def calculate_top_recommendations(day: int, time_val: int, distance: int):
    global analytics_mart
    if not analytics_mart:
        load_or_generate_mart()

    hour = min(23, max(0, time_val // 100))
    day_str = str(day)
    hour_str = str(hour)

    tier = "medium"
    if distance < 500:
        tier = "short"
    elif distance > 1500:
        tier = "long"

    carrier_hourly_all = analytics_mart.get("carrier_hourly_stats", {})
    carrier_day_all = analytics_mart.get("carrier_day_stats", {})
    carrier_stats_all = analytics_mart.get("carrier_stats", {})
    tier_data = analytics_mart.get("distance_tier_stats", {}).get(tier, {"delay_rate": 21.0})
    tier_rate = tier_data["delay_rate"]

    all_ranked = []
    badges = [
        "🏆 Top Pick: Lowest Departure Delay in this Window",
        "⭐ Highly Recommended: Consistent On-Time Buffer",
        "✨ Solid Alternative: Minimal Gate Hold"
    ]

    for c_code, c_info in carrier_stats_all.items():
        base_rate = c_info.get("delay_rate", 20.0)
        
        # 1. Carrier Specific Delay Rate at THIS EXACT HOUR
        c_hourly_entry = carrier_hourly_all.get(c_code, {}).get(hour_str)
        if c_hourly_entry and c_hourly_entry.get("volume", 0) >= 15:
            hour_rate = c_hourly_entry["delay_rate"]
            window_flights = c_hourly_entry["volume"]
        elif c_hourly_entry:
            # Small sample smoothing with carrier baseline
            vol = c_hourly_entry.get("volume", 1)
            hour_rate = (c_hourly_entry["delay_rate"] * (vol / 15.0)) + (base_rate * (1.0 - (vol / 15.0)))
            window_flights = vol
        else:
            # Carrier does not typically fly at this hour, penalize or use fallback
            hour_rate = base_rate * 1.15
            window_flights = 0

        # 2. Carrier Specific Delay Rate on THIS DAY OF WEEK
        c_day_entry = carrier_day_all.get(c_code, {}).get(day_str)
        if c_day_entry:
            day_rate = c_day_entry.get("delay_rate", base_rate)
        else:
            day_rate = base_rate

        # 3. Dynamic Multi-Dimensional Empirical Delay Risk
        # 60% weight on carrier's performance at THIS HOUR, 25% on day-of-week, 15% distance tier
        empirical_risk = (hour_rate * 0.60) + (day_rate * 0.25) + (tier_rate * 0.15)
        empirical_risk = round(float(empirical_risk), 2)
        on_time = round(max(0.0, 100.0 - empirical_risk), 1)

        # Only recommend carriers with active routes in HDFS
        if c_info.get("total_flights", 0) > 100:
            all_ranked.append({
                "carrier_code": c_code,
                "carrier_id": c_info.get("carrier_id", -1),
                "carrier_name": c_info.get("name", c_code),
                "delay_risk_pct": empirical_risk,
                "on_time_pct": on_time,
                "window_flights_in_hdfs": window_flights,
                "total_flights_in_hdfs": c_info.get("total_flights", 0),
                "avg_delay_mins": c_info.get("avg_delay", 0.0)
            })

    # Sort strictly by lowest delay risk for THIS hour and day
    all_ranked.sort(key=lambda x: x["delay_risk_pct"])

    top_3 = []
    for idx, item in enumerate(all_ranked[:3]):
        item_copy = dict(item)
        item_copy["rank"] = idx + 1
        item_copy["badge"] = badges[idx] if idx < len(badges) else "Recommended Carrier"
        top_3.append(item_copy)

    return top_3

@app.post("/recommend-airlines")
def recommend_airlines(query: RecommendationQuery):
    top_3 = calculate_top_recommendations(query.day, query.time, query.distance)
    return {
        "status": "success",
        "day": query.day,
        "departure_time": query.time,
        "recommendations": top_3
    }

@app.post("/predict")
@app.post("/analyze")
def analyze_delay(data: FlightData):
    global analytics_mart
    if not analytics_mart:
        load_or_generate_mart()

    carrier_code = CARRIER_MAP.get(data.airline, "UA")
    hour = min(23, max(0, data.time // 100))
    day_str = str(data.day)
    hour_str = str(hour)

    tier = "medium"
    if data.distance < 500:
        tier = "short"
    elif data.distance > 1500:
        tier = "long"

    carrier_hourly_all = analytics_mart.get("carrier_hourly_stats", {})
    carrier_day_all = analytics_mart.get("carrier_day_stats", {})
    carrier_stats_all = analytics_mart.get("carrier_stats", {})
    hour_stats_all = analytics_mart.get("hourly_stats", {})

    carrier_data = carrier_stats_all.get(carrier_code, {"name": carrier_code, "delay_rate": 21.0, "avg_delay": 35.0, "total_flights": 10000})
    base_rate = carrier_data["delay_rate"]

    c_hourly_entry = carrier_hourly_all.get(carrier_code, {}).get(hour_str)
    if c_hourly_entry and c_hourly_entry.get("volume", 0) >= 15:
        hour_rate = c_hourly_entry["delay_rate"]
        sample_size = c_hourly_entry["volume"]
    elif c_hourly_entry:
        vol = c_hourly_entry.get("volume", 1)
        hour_rate = (c_hourly_entry["delay_rate"] * (vol / 15.0)) + (base_rate * (1.0 - (vol / 15.0)))
        sample_size = vol
    else:
        hour_rate = base_rate * 1.15
        sample_size = hour_stats_all.get(hour_str, {}).get("volume", 10000)

    c_day_entry = carrier_day_all.get(carrier_code, {}).get(day_str)
    day_rate = c_day_entry.get("delay_rate", base_rate) if c_day_entry else base_rate

    tier_data = analytics_mart.get("distance_tier_stats", {}).get(tier, {"delay_rate": 21.0})
    tier_rate = tier_data["delay_rate"]

    cluster_avg = analytics_mart["metadata"]["cluster_delay_rate"]

    empirical_risk = (hour_rate * 0.60) + (day_rate * 0.25) + (tier_rate * 0.15)
    risk_percentage = round(float(empirical_risk), 2)
    delayed = risk_percentage > 30.0

    base_delay_mins = carrier_data.get("avg_delay", 35.0)
    expected_delay_mins = round(float(base_delay_mins * (risk_percentage / cluster_avg)), 1) if delayed else 0.0

    hour_congestion = hour_stats_all.get(hour_str, {}).get("congestion", "Moderate")

    # Dynamic Top 3 Recommendations for this exact departure slot
    top_3_recommendations = calculate_top_recommendations(data.day, data.time, data.distance)

    return {
        "risk_percentage": risk_percentage,
        "delayed": delayed,
        "sample_size": sample_size,
        "expected_delay_minutes": expected_delay_mins,
        "congestion_level": hour_congestion,
        "carrier_name": carrier_data.get("name", carrier_code),
        "engine": "Apache Spark (PySpark) Empirical Aggregation (HDFS)",
        "top_recommended_airlines": top_3_recommendations
    }
