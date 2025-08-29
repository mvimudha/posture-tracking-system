# tracker.py
import numpy as np

def iou(a, b):
    """
    Compute intersection-over-union of two boxes.
    Boxes: [x1, y1, x2, y2]
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    
    return inter / union

class MultiPersonTracker:
    def __init__(self, iou_thresh=0.3, max_age=30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks = {}  # id: {"bbox":[x1,y1,x2,y2], "age":int, "miss":int}
    
    def update(self, detections):
        """
        Update tracks with current frame detections.
        detections: list of [x1,y1,x2,y2]
        Returns: dict of {track_id: bbox}
        """
        # Age all tracks
        for t in self.tracks.values():
            t["age"] += 1
            t["miss"] += 1
        
        # Match by IOU
        unmatched = set(range(len(detections)))
        for tid, t in list(self.tracks.items()):
            best_j = -1
            best_iou = self.iou_thresh
            for j in unmatched:
                i = iou(t["bbox"], detections[j])
                if i > best_iou:
                    best_iou = i
                    best_j = j
            
            if best_j >= 0:
                self.tracks[tid]["bbox"] = detections[best_j]
                self.tracks[tid]["miss"] = 0
                unmatched.remove(best_j)
        
        # Create new tracks for unmatched detections
        for j in unmatched:
            self.tracks[self.next_id] = {
                "bbox": detections[j],
                "age": 1,
                "miss": 0
            }
            self.next_id += 1
        
        # Remove stale tracks
        drop_ids = [tid for tid, t in self.tracks.items() if t["miss"] > self.max_age]
        for tid in drop_ids:
            del self.tracks[tid]
        
        return {tid: t["bbox"] for tid, t in self.tracks.items()}