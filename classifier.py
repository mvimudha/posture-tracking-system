# classifier.py
import time
from collections import deque
import numpy as np
from config import WIN, OVERREACH_MIN_SEC, INACTIVITY_SEC, BALANCE_JITTER, THRESHOLDS
from helpers import median_ignore_none

DEBUG = False  # set True while tuning to print problem frames

def clamp_angle(x):
    """Clamp to sensible range and ensure numeric"""
    if x is None:
        return None
    try:
        xv = float(x)
    except Exception:
        return None
    # angles should be between 0 and 180 in our definitions
    if np.isnan(xv) or np.isinf(xv):
        return None
    return max(0.0, min(180.0, xv))

class PostureClassifier:
    def __init__(self):
        # buffers for smoothing angles
        self.buf = {k: deque(maxlen=WIN) for k in ["neck_tilt","spine_bend","arm_elev","knee_bend","twist"]}
        self.timestamps = deque(maxlen=WIN)
        self.last_motion_time = time.time()
        self.prev_keypoints = None
        self.balance_jitter = deque(maxlen=WIN)
        self.dur_start = {}

    def update(self, feats, t_now):
        """
        feats: dict expected to contain keys:
          "neck_tilt","spine_bend","arm_elev","knee_bend","twist",
          optional: "shoulder_mid","hip_mid","l_ankle","r_ankle"
        """
        # sanitize/clamp and append
        clamped = {}
        for k in self.buf.keys():
            clamped[k] = clamp_angle(feats.get(k))
            self.buf[k].append(clamped[k])
        self.timestamps.append(t_now)

        # Track movement for inactivity (requires coordinate points)
        if all(key in feats and feats[key] is not None for key in ["shoulder_mid", "hip_mid", "l_ankle", "r_ankle"]):
            try:
                kps = np.array([
                    feats["shoulder_mid"][:2],
                    feats["hip_mid"][:2],
                    feats["l_ankle"][:2],
                    feats["r_ankle"][:2],
                ], dtype=float)
                if self.prev_keypoints is not None:
                    disp = np.linalg.norm(kps - self.prev_keypoints, axis=1).mean()
                    self.balance_jitter.append(disp)
                    if disp > 0.005:
                        self.last_motion_time = t_now
                self.prev_keypoints = kps
            except Exception:
                # keep previous prev_keypoints if conversion fails
                pass

    def smoothed(self, key):
        vals = [v for v in self.buf[key] if v is not None]
        if not vals:
            return None
        # median is robust to outliers
        med = median_ignore_none(vals)
        return med

    def sustained(self, key, cond, now, threshold=2.5):
        if cond:
            if key not in self.dur_start or self.dur_start[key] is None:
                self.dur_start[key] = now
            elif (now - self.dur_start[key]) >= threshold:
                return True
        else:
            self.dur_start[key] = None
        return False

    def classify(self):
        now = time.time()
        out = []

        # Smoothed absolute angles (clamped)
        angles = {k: self.smoothed(k) for k in self.buf}

        neck = angles["neck_tilt"]
        spine = angles["spine_bend"]
        arm = angles["arm_elev"]
        knee = angles["knee_bend"]
        twist = angles["twist"]

        # Quick debug print to catch odd values
        if DEBUG:
            print(f"[DEBUG] angles n:{neck} s:{spine} a:{arm} k:{knee} t:{twist}")

        # guard: if we have no meaningful angles, return neutral
        if all(v is None for v in (neck, spine, arm, knee, twist)):
            return [("Neutral (no pose)", "Safe")], angles

        # Minimum triggers to avoid jitter / marginal triggers
        NECK_MIN_TRIGGER = THRESHOLDS.get("neck_min_deg", 20)   # must exceed this AND the official tilt threshold
        ARM_MIN_TRIGGER = THRESHOLDS.get("arm_warn_deg", 45)   # require above caution to consider raised
        SPINE_UPRIGHT_MAX = THRESHOLDS.get("spine_bend_deg", 20) / 2.0  # small forward lean allowed while still 'upright'

        # Posture checks (only mark if sustained)
        # 1) Incorrect Lifting Posture: strong spine bend & not much knee bend
        cond_spine_lift = (spine is not None and spine > THRESHOLDS["spine_bend_deg"] and 
                           (knee is None or knee < THRESHOLDS["knee_bend_deg"]))
        if self.sustained("spine_lift", cond_spine_lift, now, threshold=1.0):
            out.append(("Incorrect Lifting Posture", "High Injury Risk"))

        # 2) Slouching: neck forward plus some torso bend (both required)
        cond_slouch = (neck is not None and neck > NECK_MIN_TRIGGER and neck > THRESHOLDS["neck_tilt_deg"] and
                       spine is not None and spine > (THRESHOLDS["spine_bend_deg"] * 0.4))
        if self.sustained("slouch", cond_slouch, now, threshold=2.0):
            out.append(("Slouching", "Ergonomic Hazard"))

        # 3) Overreaching: arm raised high AND torso fairly upright (so standing with arms down won't trigger)
        # Note: arm angle semantics: 0 = down, 90 = horizontal, > 120 = high reach
        cond_overreach = (arm is not None and arm >= ARM_MIN_TRIGGER and arm >= THRESHOLDS["arm_elev_deg"] and
                          spine is not None and spine < SPINE_UPRIGHT_MAX)
        if self.sustained("overreach", cond_overreach, now, threshold=OVERREACH_MIN_SEC):
            out.append(("Overreaching", "Overexertion Risk"))

        # 4) Torso twist
        cond_twist = (twist is not None and twist > THRESHOLDS["twist_deg"])
        if self.sustained("twist", cond_twist, now, threshold=1.5):
            # if spine significantly bent while twisting, flag twisting-while-lifting
            label = "Twisting While Lifting" if (spine is not None and spine > THRESHOLDS["spine_bend_deg"]) else "Torso Twisting"
            out.append((label, "Spinal Strain Risk"))

        # 5) Kneeling/Crouching
        cond_kneel = (knee is not None and knee > THRESHOLDS["knee_bend_deg"])
        if self.sustained("kneel", cond_kneel, now, threshold=2.0):
            out.append(("Kneeling/Crouching", "Medium Risk"))

        # Inactivity
        if (now - self.last_motion_time) > INACTIVITY_SEC:
            out.append(("Unusual Inactivity", "Fatigue/Injury Suspect"))

        # Balance
        if len(self.balance_jitter) >= WIN and np.mean(self.balance_jitter) > BALANCE_JITTER:
            out.append(("Loss of Balance", "Slip/Fall Hazard"))

        # If nothing triggered, produce safe/monitoring labels with conservative thresholds
        if not out:
            # If arm is low (< warn) AND neck/spine small => safe
            if ((arm is None or arm < THRESHOLDS["arm_warn_deg"]) and
                (neck is None or neck < THRESHOLDS["neck_tilt_deg"]) and
                (spine is None or spine < THRESHOLDS["spine_bend_deg"])):
                out.append(("Safe Standing Posture", "Safe"))
            # If arm moderately raised, but no other risk -> Monitor
            elif arm is not None and arm >= THRESHOLDS["arm_warn_deg"]:
                out.append(("Arms Elevated", "Monitor"))
            else:
                out.append(("Neutral Posture", "Safe"))

        # Deduplicate & choose highest risk label per category
        pref = {
            "High Injury Risk": 4,
            "Spinal Strain Risk": 3,
            "Overexertion Risk": 3,
            "Ergonomic Hazard": 2,
            "Slip/Fall Hazard": 3,
            "Fatigue/Injury Suspect": 2,
            "Medium Risk": 1,
            "Monitor": 0.5,
            "Safe": 0
        }
        best = {}
        for label, risk in out:
            if label not in best or pref[risk] > pref[best[label]]:
                best[label] = risk
        ranked = sorted(best.items(), key=lambda x: pref[x[1]], reverse=True)

        # optional debug: print reason if false positive persists
        if DEBUG and ranked and ranked[0][1] != "Safe":
            print(f"[DEBUG] triggered {ranked[:2]} angles n:{neck} s:{spine} a:{arm} k:{knee} t:{twist}")

        return ranked[:2], angles
