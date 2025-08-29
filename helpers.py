# helpers.py
import numpy as np
import math

def get_xyz(lm):
    """Convert MediaPipe landmark to numpy array"""
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

def vec(a, b):
    """Vector from a to b"""
    return b - a

def angle_between(u, v):
    """Angle in degrees between vectors u and v"""
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return None
    cosang = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def line_angle_2d(u):
    """2D angle of vector (x,y) in degrees"""
    if np.linalg.norm(u[:2]) < 1e-6:
        return None
    return math.degrees(math.atan2(u[1], u[0]))

def median_ignore_none(vals):
    """Median of list ignoring None values"""
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None

def visible(lms, idxs, min_vis=0.5):
    """Check if all landmarks in idxs are sufficiently visible"""
    return all(lms[i].visibility >= min_vis for i in idxs)