# config.py
WIN = 10
OVERREACH_MIN_SEC = 3.0
INACTIVITY_SEC = 4.0
BALANCE_JITTER = 0.08

# OSHA/ergonomics-aligned heuristics
THRESHOLDS = {
    "neck_tilt_deg": 35,
    "spine_bend_deg": 25,
    "arm_elev_deg": 140,
    "arm_warn_deg": 70,
    "arm_ideal_max": 40,
    "knee_bend_deg": 110,
    "twist_deg": 30
}
