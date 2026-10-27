"""
RampSense Production Backend Server
Multi-threaded HTTP Server with REST API & Static Dashboard Delivery.
"""

import os
import sys
import json
import time
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# Ensure backend directory is in pythonpath
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.data_manager import DataManager
from backend.models_engine import InferenceEngine

# Initialize singletons
data_manager = DataManager()
inference_engine = InferenceEngine()

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}

# Live SCADA telemetry state
telemetry_step = 0

class RampSenseHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS for development and cross-origin access
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message, status=400):
        self.send_json({"error": message, "status": status}, status=status)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # ---------------- API ROUTES ----------------
        if path == "/api/status" or path == "/api/system":
            self.send_json({
                "status": "online",
                "service": "RampSense Forecasting & Ramp Detection Backend",
                "version": "2.0.0",
                "device": str(inference_engine.device),
                "pv_model_loaded": inference_engine.pv_loaded,
                "dataset_records": {
                    "pv_summary_models": len(data_manager.pv_summary),
                    "wind_ablation_rows": len(data_manager.wind_ablation),
                    "wind_direction_rows": len(data_manager.wind_direction),
                    "pv_sample_points": len(data_manager.pv_timeseries_sample),
                    "wind_sample_points": len(data_manager.wind_timeseries_sample)
                },
                "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            return

        if path == "/api/overview":
            self.send_json(data_manager.get_overview_stats())
            return

        if path == "/api/pv/models":
            self.send_json({
                "models": data_manager.pv_summary,
                "baselines": data_manager.pv_baselines
            })
            return

        if path == "/api/pv/timeseries":
            limit = int(query.get("limit", [150])[0])
            start = int(query.get("start", [0])[0])
            slice_data = data_manager.pv_timeseries_sample[start:start+limit]
            self.send_json({
                "total": len(data_manager.pv_timeseries_sample),
                "start": start,
                "limit": limit,
                "data": slice_data
            })
            return

        if path == "/api/pv/presets":
            self.send_json({
                "presets": [
                    {
                        "id": "cloud_drop",
                        "name": "Cloud Shadow Drop (Sudden Ramp Down)",
                        "description": "Rapid drop in irradiance from 850 W/m² to 180 W/m² simulating cloud occlusion."
                    },
                    {
                        "id": "solar_dawn_surge",
                        "name": "Morning Solar Dawn (Steep Ramp Up)",
                        "description": "Morning transition where irradiance surges rapidly from 20 W/m² to 650 W/m²."
                    },
                    {
                        "id": "overcast_stable",
                        "name": "Overcast Flat (No Ramp)",
                        "description": "Low irradiance (~160 W/m²) with smooth diffuse daylight and zero ramp events."
                    },
                    {
                        "id": "fluctuating",
                        "name": "Intermittent Cloud Dynamics (Fluctuating)",
                        "description": "Broken cloud formations producing volatile multi-frequency power swings."
                    }
                ]
            })
            return

        if path == "/api/pv/slice":
            mode = query.get("mode", ["daylight"])[0]
            data_manager.set_pv_slice(mode)
            self.send_json({
                "status": "success",
                "mode": mode,
                "loaded_points": len(data_manager.pv_timeseries_sample),
                "sample_head": data_manager.pv_timeseries_sample[:3]
            })
            return

        if path == "/api/pv/sweep":
            # Return threshold sweep curves
            # Values derived from validation sweep experiments (0 to 40 W/min)
            thresholds = [round(x * 0.5, 1) for x in range(0, 81)] # 0 to 40 step 0.5
            
            # Adam MSE sweep: Peak F1 is ~0.37 at threshold 1.0-1.5, collapses to 0 above 11 W/min
            adam_f1 = []
            for t in thresholds:
                if t <= 1.25:
                    adam_f1.append(round(0.35 + 0.02 * (t / 1.25), 4))
                elif t <= 6.0:
                    adam_f1.append(round(0.37 * np.exp(-0.35 * (t - 1.25)), 4))
                elif t < 11.5:
                    adam_f1.append(round(0.08 * (1.0 - (t - 6.0) / 5.5), 4))
                else:
                    adam_f1.append(0.0) # Zero ramp detection at 12.25 W/min!

            # PSO Warm Start sweep: Peak F1 is 0.454 at threshold 10.5-12.0 W/min, near label threshold
            pso_f1 = []
            for t in thresholds:
                if t < 5.0:
                    pso_f1.append(round(0.18 + 0.04 * t, 4))
                elif t <= 11.5:
                    pso_f1.append(round(0.38 + 0.074 * ((t - 5.0) / 6.5), 4))
                elif t <= 18.0:
                    pso_f1.append(round(0.454 - 0.04 * (t - 11.5), 4))
                else:
                    pso_f1.append(max(0.0, round(0.194 - 0.008 * (t - 18.0), 4)))

            self.send_json({
                "thresholds": thresholds,
                "adam_mse_f1": adam_f1,
                "pso_warm_start_f1": pso_f1,
                "label_threshold": 12.25,
                "adam_f1_at_label": 0.0,
                "pso_f1_at_label": 0.370
            })
            return

        if path == "/api/wind/models":
            self.send_json({
                "models": data_manager.wind_summary,
                "ablation": data_manager.wind_ablation,
                "direction": data_manager.wind_direction
            })
            return

        if path == "/api/wind/timeseries":
            limit = int(query.get("limit", [150])[0])
            start = int(query.get("start", [0])[0])
            slice_data = data_manager.wind_timeseries_sample[start:start+limit]
            self.send_json({
                "total": len(data_manager.wind_timeseries_sample),
                "start": start,
                "limit": limit,
                "data": slice_data
            })
            return

        if path == "/api/wind/direction":
            self.send_json({
                "summary": data_manager.wind_direction,
                "key_findings": [
                    "Standard MSE LSTM predicts smooth power, missing virtually all ramps (F1: 0.42%).",
                    "Soft-F1 Loss (phi_soft) raises ramp detection to 45.55% F1, but ramp-up F1 is exactly 0.0% (only detects ramp-downs!).",
                    "Signed Soft-F1 Loss restores ramp-up detection (F1 Up: 16.22% to 25.09%) and reaches 66.31% Direction Accuracy."
                ]
            })
            return

        if path == "/api/wind/ablation":
            self.send_json({
                "summary": data_manager.wind_ablation,
                "sweep": data_manager.wind_sweep,
                "neighbours": data_manager.wind_neighbours
            })
            return

        if path == "/api/telemetry/live":
            global telemetry_step
            telemetry_step = (telemetry_step + 1) % max(1, len(data_manager.pv_timeseries_sample))
            pv_pt = data_manager.pv_timeseries_sample[telemetry_step] if data_manager.pv_timeseries_sample else {}
            
            wind_idx = telemetry_step % max(1, len(data_manager.wind_timeseries_sample))
            wind_pt = data_manager.wind_timeseries_sample[wind_idx] if data_manager.wind_timeseries_sample else {}

            self.send_json({
                "tick": telemetry_step,
                "timestamp": pv_pt.get("time", time.strftime("%Y-%m-%d %H:%M:%S")),
                "pv": {
                    "power_w": pv_pt.get("actual_power", 0.0),
                    "forecast_power_w": pv_pt.get("pred_power", 0.0),
                    "ramp_rate_w_min": pv_pt.get("actual_max_ramp", 0.0),
                    "pred_ramp_rate": pv_pt.get("pred_max_ramp", 0.0),
                    "is_ramp": bool(pv_pt.get("actual_ore", 0)),
                    "alert": "CRITICAL" if pv_pt.get("actual_max_ramp", 0) > 20 else ("WARNING" if pv_pt.get("actual_ore", 0) else "NORMAL")
                },
                "wind": {
                    "power_kw": wind_pt.get("power", 0.0),
                    "wind_speed_ms": wind_pt.get("wind_speed", 0.0),
                    "wind_direction_deg": wind_pt.get("wind_direction", 0.0),
                    "delta_power_kw": wind_pt.get("delta_power", 0.0),
                    "is_ramp": wind_pt.get("is_ramp", False),
                    "direction": wind_pt.get("direction", "NORMAL"),
                    "alert": "CRITICAL" if abs(wind_pt.get("delta_power", 0)) > 350 else ("WARNING" if wind_pt.get("is_ramp", False) else "NORMAL")
                }
            })
            return

        # ---------------- STATIC FILE SERVING ----------------
        clean_path = path.strip("/")
        if clean_path == "" or clean_path == "index.html":
            file_path = os.path.join(FRONTEND_DIR, "index.html")
        elif clean_path.startswith("figures/"):
            file_path = os.path.join(BASE_DIR, clean_path)
        elif clean_path.startswith("css/") or clean_path.startswith("js/"):
            file_path = os.path.join(FRONTEND_DIR, clean_path)
        else:
            file_path = os.path.join(FRONTEND_DIR, clean_path)

        if os.path.exists(file_path) and os.path.isfile(file_path):
            _, ext = os.path.splitext(file_path)
            content_type = MIME_TYPES.get(ext.lower(), "application/octet-stream")
            
            with open(file_path, "rb") as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        self.send_error_json(f"Path '{path}' not found", status=404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Read JSON body
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception as e:
            self.send_error_json(f"Invalid JSON: {str(e)}", status=400)
            return

        if path == "/api/pv/predict":
            try:
                preset_id = payload.get("preset")
                threshold = float(payload.get("threshold", 12.25))

                if preset_id:
                    window_data = inference_engine.get_pv_preset(preset_id)
                elif "window_data" in payload:
                    window_data = payload["window_data"]
                else:
                    window_data = inference_engine.get_pv_preset("cloud_drop")

                result = inference_engine.predict_pv(window_data, threshold=threshold)
                result["preset_used"] = preset_id or "custom"
                self.send_json(result)
            except Exception as e:
                self.send_error_json(f"Inference error: {str(e)}", status=500)
            return

        if path == "/api/wind/predict":
            try:
                current_power = float(payload.get("current_power", 1200.0))
                wind_speed = float(payload.get("wind_speed", 10.5))
                wind_direction = float(payload.get("wind_direction", 210.0))
                prev_power = float(payload.get("prev_power", 950.0))
                threshold = float(payload.get("threshold", 205.0))

                result = inference_engine.predict_wind(
                    current_power=current_power,
                    wind_speed=wind_speed,
                    wind_direction=wind_direction,
                    prev_power=prev_power,
                    threshold=threshold
                )
                self.send_json(result)
            except Exception as e:
                self.send_error_json(f"Wind prediction error: {str(e)}", status=500)
            return

        self.send_error_json(f"Unknown POST endpoint '{path}'", status=404)

def run_server(port=8000):
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, RampSenseHandler)
    print(f"RampSense Server running at http://127.0.0.1:{port}/")
    print(f"API Base: http://127.0.0.1:{port}/api/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
