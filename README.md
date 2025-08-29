# Posture & Safety Monitoring with Pose Estimation

This project combines **MediaPipe Pose** and **YOLOv8** to analyze human posture and detect unsafe or risky movements in real time.  
It can classify postures, track body stability, and monitor activity levels — making it useful for **workplace safety, ergonomics, fitness tracking, and robotics applications**.

---

## 📌 Features
- Real-time **pose detection** using MediaPipe
- **Posture classification** (safe, bending, reaching, inactive, unstable, etc.)
- **Multi-person tracking** with rolling buffers
- Detection of **inactivity, overreaching, and balance jitter**
- Customizable **thresholds and safety rules** in `config.py`
- Lightweight and works with a standard webcam

---

## 🚀 Use Cases
- 🏭 **Workplace Safety** – Detect workers bending or overreaching in factories/warehouses  
- 💻 **Ergonomics** – Monitor desk posture for office workers  
- 🏋️ **Fitness/Training** – Track exercise posture & form correction  
- 🤖 **Robotics** – Provide human posture feedback for collaborative robots  

---

## 🛠️ Installation

Clone the repository and install the requirements:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt
```

---

## 📂 File Structure

```
YOUR_REPO_NAME/
│── main.py                # Main entry point for running the system
│── helpers.py             # Utility math & landmark functions
│── angles.py              # Angle calculations between joints
│── classifier.py          # Posture classifier logic
│── tracker.py             # Multi-person tracker
│── config.py              # Configuration and thresholds
│── requirements.txt       # Python dependencies
```

---

## ▶️ Usage

Run the system with:

```bash
python main.py
```

You can modify thresholds and sensitivity in `config.py` to adapt it for your use case.

---

## ⚙️ How It Works

1. **Pose Estimation**  
   - MediaPipe extracts 3D body landmarks (shoulders, hips, ankles, etc.).

2. **Feature Extraction**  
   - Vectors and angles are calculated (e.g., torso bend, arm reach).

3. **Posture Classification**  
   - The `PostureClassifier` maps movements into categories:  
     * `safe`  
     * `bending`  
     * `reaching`  
     * `inactive`  
     * `unstable`  

4. **Tracking & Alerts**  
   - Movement is tracked with rolling buffers.  
   - Unsafe postures are flagged only if sustained for a configurable duration.  

---

## 📊 Example Categories

By default, the system detects:

- ✅ **Safe posture**  
- 🔻 **Bending forward**  
- 📏 **Overreaching**  
- 💤 **Inactive / idle**  
- ⚠️ **Unstable / jittery balance**  

These can be extended or customized in `classifier.py`.

---

## 👤 Author

Developed by **Vimudha**.  
Feel free to contribute, open issues, or suggest improvements!
