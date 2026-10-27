"""
RampSense Data Manager
Loads and prepares PV and Wind datasets, summaries, sweep data, and ablation results.
"""

import os
import json
import csv
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PV_DIR = os.path.join(BASE_DIR, "PV dataset")
WIND_DIR = os.path.join(BASE_DIR, "wind dataset")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
OPTIMIZER_OUTPUTS = os.path.join(PV_DIR, "sequences", "optimizer", "outputs")

class DataManager:
    def __init__(self):
        self.pv_summary = []
        self.pv_baselines = {}
        self.wind_summary = []
        self.wind_ablation = []
        self.wind_direction = []
        self.wind_sweep = []
        self.wind_neighbours = []
        self.pv_timeseries_sample = []
        self.wind_timeseries_sample = []
        self.load_all()

    def load_all(self):
        self._load_pv_results()
        self._load_wind_results()
        self._load_timeseries_samples()

    def _load_pv_results(self):
        pv_per_run_path = os.path.join(OPTIMIZER_OUTPUTS, "pv_results_per_run.csv")
        if os.path.exists(pv_per_run_path):
            with open(pv_per_run_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.pv_summary = list(reader)

        baselines_path = os.path.join(OPTIMIZER_OUTPUTS, "baselines_test_once_report.json")
        if os.path.exists(baselines_path):
            try:
                with open(baselines_path, mode="r", encoding="utf-8") as f:
                    self.pv_baselines = json.load(f)
            except Exception as e:
                print("Error loading baselines json:", e)

    def _load_wind_results(self):
        w_final = os.path.join(WIND_DIR, "final_summary.csv")
        if os.path.exists(w_final):
            with open(w_final, mode="r", encoding="utf-8") as f:
                self.wind_summary = list(csv.DictReader(f))

        w_dir = os.path.join(WIND_DIR, "direction_summary.csv")
        if os.path.exists(w_dir):
            with open(w_dir, mode="r", encoding="utf-8") as f:
                self.wind_direction = list(csv.DictReader(f))

        w_abl = os.path.join(WIND_DIR, "ablation_final_summary.csv")
        if os.path.exists(w_abl):
            with open(w_abl, mode="r", encoding="utf-8") as f:
                self.wind_ablation = list(csv.DictReader(f))

        w_sweep = os.path.join(WIND_DIR, "sweep_summary.csv")
        if os.path.exists(w_sweep):
            with open(w_sweep, mode="r", encoding="utf-8") as f:
                self.wind_sweep = list(csv.DictReader(f))

        w_neigh = os.path.join(WIND_DIR, "neighbours_summary.csv")
        if os.path.exists(w_neigh):
            with open(w_neigh, mode="r", encoding="utf-8") as f:
                self.wind_neighbours = list(csv.DictReader(f))

    def _load_timeseries_samples(self, pv_start_idx=23200):
        # Load sample slice of PV test predictions
        # Index 23200 corresponds to Nov 21, 2017: 09:15 AM to 15:30 PM (bright daylight with frequent ramp events up to 211 W)
        pv_pred_csv = os.path.join(PV_DIR, "sequences", "evaluation", "pv_multistep_ore_pre_predictions.csv")
        self.pv_timeseries_sample = []
        if os.path.exists(pv_pred_csv):
            try:
                with open(pv_pred_csv, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    count = 0
                    for idx, row in enumerate(reader):
                        if idx >= pv_start_idx and count < 350:
                            act_p = float(row.get("actual_t_plus_7", 0.0))
                            pred_p = float(row.get("pred_t_plus_7", 0.0))
                            act_ramp = float(row.get("actual_max_ramp_W_per_min", 0.0))
                            pred_ramp = float(row.get("predicted_max_ramp_W_per_min", 0.0))
                            self.pv_timeseries_sample.append({
                                "time": row.get("time", ""),
                                "actual_power": round(act_p, 2),
                                "pred_power": round(pred_p, 2),
                                "actual_max_ramp": round(act_ramp, 2),
                                "pred_max_ramp": round(pred_ramp, 2),
                                "actual_ore": int(row.get("actual_ORE", 0)),
                                "pred_ore": int(row.get("predicted_ORE", 0))
                            })
                            count += 1
            except Exception as e:
                print("Error loading PV timeseries sample:", e)

        # Load sample slice of Wind WT15 (active wind power and ramp events)
        wind_csv = os.path.join(WIND_DIR, "WT15_2017_2018_correct.csv")
        self.wind_timeseries_sample = []
        if os.path.exists(wind_csv):
            try:
                with open(wind_csv, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    prev_p = None
                    for idx, row in enumerate(reader):
                        if idx >= 50 and idx < 400:
                            p = float(row.get("power", 0.0))
                            ws = float(row.get("wind_speed", 0.0))
                            wd = float(row.get("wind_direction", 0.0))
                            delta_p = (p - prev_p) if prev_p is not None else 0.0
                            prev_p = p
                            is_ramp = abs(delta_p) >= 205.0
                            direction = "UP" if delta_p >= 205.0 else ("DOWN" if delta_p <= -205.0 else "NORMAL")
                            self.wind_timeseries_sample.append({
                                "time": row.get("timestamp", ""),
                                "power": round(p, 2),
                                "wind_speed": round(ws, 2),
                                "wind_direction": round(wd, 1),
                                "delta_power": round(delta_p, 2),
                                "is_ramp": is_ramp,
                                "direction": direction
                            })
            except Exception as e:
                print("Error loading Wind sample:", e)

    def set_pv_slice(self, mode="daylight"):
        """Change the active PV slice for demoing different solar conditions."""
        if mode == "daylight": # High sun & active ramps (Nov 21, 10:00 - 15:00)
            self._load_timeseries_samples(pv_start_idx=23200)
        elif mode == "cloud_drop": # Cloud occlusion ramp event (Nov 06, 09:30 - 14:00)
            self._load_timeseries_samples(pv_start_idx=4550)
        elif mode == "night": # Nighttime baseline (00:00 - 05:00, zero irradiance)
            self._load_timeseries_samples(pv_start_idx=0)
        else:
            self._load_timeseries_samples(pv_start_idx=23200)

    def get_overview_stats(self):
        return {
            "project_name": "RampSense",
            "full_title": "Ramp-Aware LSTM Forecasting for Photovoltaic and Wind Power",
            "base_paper": "S. E. Kostoglou et al., IEEE Access 2026",
            "status": "Production Ready",
            "pv_system": {
                "name": "SolarTech Lab PV System",
                "nominal_rating": "245 Wp",
                "sampling_rate": "1-minute",
                "total_rows": 525601,
                "history_window": "60 minutes (60 steps)",
                "forecast_horizon": "8 steps (t+6 to t+13 min, h=10, Δt=3)",
                "ramp_threshold": "12.25 W/min (5% of nominal)",
                "best_model": "PSO Warm Start (Adam→PSO)",
                "best_test_f1": 0.347,
                "best_test_ap": 0.290,
                "baseline_f1_mse": 0.257,
                "baseline_ap_mse": 0.167
            },
            "wind_system": {
                "name": "Penmanshiel WT15 Turbine",
                "nominal_rating": "2050 kW (2.05 MW)",
                "sampling_rate": "10-minute",
                "total_rows": 105120,
                "history_window": "10-60 min (Lag 1 to Lag 6)",
                "forecast_horizon": "10 minutes ahead",
                "ramp_threshold": "205.0 kW (10% of nominal)",
                "best_model": "Signed Soft-F1 LSTM (Lag 6 + Diff)",
                "best_test_f1": 45.55,
                "direction_accuracy": "66.31%",
                "mse_baseline_f1": 0.42,
                "mse_baseline_f1_up": 0.0
            }
        }
