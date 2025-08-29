import numpy as np
import mediapipe as mp
from helpers import median_ignore_none, angle_between, get_xyz, visible

mp_pose = mp.solutions.pose

def distance(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))

def neck_tilt_angle(shoulder_mid, nose, hip_mid):
    """
    Forward neck tilt relative to vertical torso axis.
    0° = neutral/upright, increases as head moves forward.
    """
    torso_vec = shoulder_mid - hip_mid        # torso axis (upwards)
    neck_vec = nose - shoulder_mid            # neck direction
    angle = angle_between(neck_vec, torso_vec)
    return angle if angle is not None else None

def spine_bend_angle(shoulder_mid, hip_mid):
    """
    Deviation of torso from vertical axis.
    0° = upright, increases when bending forward/backward.
    """
    torso_vec = shoulder_mid - hip_mid
    vertical_vec = np.array([0, -1], dtype=np.float32)  # straight up (Y axis negative in image coords)
    angle = angle_between(torso_vec, vertical_vec)
    return angle if angle is not None else None

def arm_elevation_angle(shoulder, elbow, wrist):
    """
    Arm elevation relative to vertical (hanging down).
    0° = arm hanging straight down, 90° = horizontal, 180° = overhead.
    """
    upper_arm_vec = elbow - shoulder
    vertical_vec = np.array([0, 1], dtype=np.float32)   # gravity direction (down)
    angle = angle_between(upper_arm_vec, vertical_vec)
    return angle if angle is not None else None

def knee_bend_angle(hip, knee, ankle):
    """
    Knee bend relative to straight leg.
    0° = fully straight, increases as knee bends.
    """
    thigh_vec = hip - knee
    shank_vec = ankle - knee
    angle = angle_between(thigh_vec, shank_vec)
    if angle is None:
        return None
    # A straight knee = ~180°, we convert to bend (0° straight, >0° bent)
    return abs(180 - angle)

def torso_twist_angle(l_sh, r_sh, l_hip, r_hip):
    """
    Relative rotation between shoulder line and hip line.
    0° = perfectly aligned, increases as torso twists.
    """
    sh_vec = np.array(r_sh, dtype=np.float32) - np.array(l_sh, dtype=np.float32)
    hip_vec = np.array(r_hip, dtype=np.float32) - np.array(l_hip, dtype=np.float32)
    return angle_between(sh_vec, hip_vec)

def compute_angles(lms, min_vis=0.5):
    L = mp_pose.PoseLandmark
    pts = {}

    # Required landmarks
    landmark_names = [
        "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE"
    ]

    # Bail if core points are not visible
    core_idxs = [L.NOSE.value, L.LEFT_SHOULDER.value, L.RIGHT_SHOULDER.value,
                 L.LEFT_HIP.value, L.RIGHT_HIP.value]
    if not visible(lms, core_idxs, min_vis=min_vis):
        return None

    # Extract points as np.float32
    for name in landmark_names:
        idx = getattr(L, name).value
        pts[name] = get_xyz(lms[idx]).astype(np.float32)

    # Midpoints
    shoulder_mid = 0.5 * (pts["LEFT_SHOULDER"] + pts["RIGHT_SHOULDER"])
    hip_mid = 0.5 * (pts["LEFT_HIP"] + pts["RIGHT_HIP"])

    # Angles
    neck_tilt = neck_tilt_angle(shoulder_mid[:2], pts["NOSE"][:2], hip_mid[:2])
    spine_bend = spine_bend_angle(shoulder_mid[:2], hip_mid[:2])

    l_arm_elev = arm_elevation_angle(pts["LEFT_SHOULDER"][:2], pts["LEFT_ELBOW"][:2], pts["LEFT_WRIST"][:2])
    r_arm_elev = arm_elevation_angle(pts["RIGHT_SHOULDER"][:2], pts["RIGHT_ELBOW"][:2], pts["RIGHT_WRIST"][:2])
    arm_elev = median_ignore_none([l_arm_elev, r_arm_elev])

    l_knee = knee_bend_angle(pts["LEFT_HIP"][:2], pts["LEFT_KNEE"][:2], pts["LEFT_ANKLE"][:2])
    r_knee = knee_bend_angle(pts["RIGHT_HIP"][:2], pts["RIGHT_KNEE"][:2], pts["RIGHT_ANKLE"][:2])
    knee_bend = median_ignore_none([l_knee, r_knee])

    twist = torso_twist_angle(
        pts["LEFT_SHOULDER"][:2], pts["RIGHT_SHOULDER"][:2],
        pts["LEFT_HIP"][:2], pts["RIGHT_HIP"][:2]
    )

    return {
        "neck_tilt": neck_tilt,
        "spine_bend": spine_bend,
        "arm_elev": arm_elev,
        "l_arm_elev": l_arm_elev,
        "r_arm_elev": r_arm_elev,
        "knee_bend": knee_bend,
        "twist": twist,
        "shoulder_mid": shoulder_mid,
        "hip_mid": hip_mid,
        "l_ankle": pts["LEFT_ANKLE"],
        "r_ankle": pts["RIGHT_ANKLE"],
    }
