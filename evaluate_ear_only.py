
import os
import glob
import math
import cv2
import dlib
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE        = r"D:\kd project\test\datasets"
FF_ROOT     = os.path.join(BASE, "FF")
CDF_ROOT    = os.path.join(BASE, "CDFv2")
PREDICTOR   = os.path.join(BASE, "shape_predictor_68_face_landmarks.dat")

FF_FAKE_SUBSETS = ["DF", "F2F", "FS", "NT"]

# ──────────────────────────────────────────────
# EAR
# ──────────────────────────────────────────────
LEFT_EYE  = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

def eye_aspect_ratio(landmarks, indices):
    pts = np.array([[landmarks.part(i).x, landmarks.part(i).y]
                    for i in indices], dtype=np.float32)
    A = math.dist(pts[1], pts[5])
    B = math.dist(pts[2], pts[4])
    C = math.dist(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def video_ear(frame_paths, detector, predictor):
    ears = []
    for fp in frame_paths:
        img = cv2.imread(fp)
        if img is None:
            continue
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        if not faces:
            continue
        shape = predictor(gray, faces[0])
        ear   = (eye_aspect_ratio(shape, LEFT_EYE) +
                 eye_aspect_ratio(shape, RIGHT_EYE)) / 2.0
        ears.append(ear)
    return float(np.mean(ears)) if ears else None

# ──────────────────────────────────────────────
# DATASET LOADERS
# ──────────────────────────────────────────────
def iter_videos(root, subset):
    folder = os.path.join(root, subset)
    if not os.path.isdir(folder):
        print(f"  [WARN] not found: {folder}")
        return
    for vid in sorted(os.listdir(folder)):
        vid_dir = os.path.join(folder, vid)
        if not os.path.isdir(vid_dir):
            continue
        frames = sorted(glob.glob(os.path.join(vid_dir, "*.png")))
        if frames:
            yield vid, frames

def load_ff(detector, predictor):
    scores, labels = [], []
    print("\n[FF++] real …")
    for _, frames in iter_videos(FF_ROOT, "real"):
        e = video_ear(frames, detector, predictor)
        if e is not None:
            scores.append(e); labels.append(0)
    for sub in FF_FAKE_SUBSETS:
        print(f"[FF++] {sub} …")
        for _, frames in iter_videos(FF_ROOT, sub):
            e = video_ear(frames, detector, predictor)
            if e is not None:
                scores.append(e); labels.append(1)
    return np.array(scores), np.array(labels)

def load_cdf(detector, predictor):
    scores, labels = [], []
    for sub, lbl in [("Celeb-real", 0), ("YouTube-real", 0), ("Celeb-synthesis", 1)]:
        print(f"[CDFv2] {sub} …")
        for _, frames in iter_videos(CDF_ROOT, sub):
            e = video_ear(frames, detector, predictor)
            if e is not None:
                scores.append(e); labels.append(lbl)
    return np.array(scores), np.array(labels)

# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────
def evaluate(scores, labels, name):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Videos : {len(scores)}  (real={sum(labels==0)}, fake={sum(labels==1)})")

    auc = roc_auc_score(labels, -scores)
    print(f"  AUC    : {auc:.4f}")

    best_tau, best_ba = None, -1
    for tau in np.arange(0.15, 0.36, 0.005):
        preds = (scores < tau).astype(int)
        tp = np.sum((preds==1)&(labels==1))
        tn = np.sum((preds==0)&(labels==0))
        fp = np.sum((preds==1)&(labels==0))
        fn = np.sum((preds==0)&(labels==1))
        ba = ((tp/(tp+fn+1e-9)) + (tn/(tn+fp+1e-9))) / 2
        if ba > best_ba:
            best_ba, best_tau = ba, tau

    preds = (scores < best_tau).astype(int)
    print(f"  Best τ : {best_tau:.3f}")
    print(f"  Bal.Acc: {best_ba:.4f}")
    print(f"  Acc    : {accuracy_score(labels, preds):.4f}")
    print(f"  Prec   : {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"  Recall : {recall_score(labels, preds, zero_division=0):.4f}")
    print(f"  F1     : {f1_score(labels, preds, zero_division=0):.4f}")
    print(f"  CM     :\n{confusion_matrix(labels, preds)}")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    if not os.path.exists(PREDICTOR):
        print(f"\n[ERROR] Landmarks model not found:\n  {PREDICTOR}")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract and place the .dat file at the path above.")
        return

    print(f"Loading dlib predictor from:\n  {PREDICTOR}")
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR)
    print("OK")

    ff_scores, ff_labels = load_ff(detector, predictor)
    np.save(os.path.join(BASE, "ff_scores.npy"), ff_scores)
    np.save(os.path.join(BASE, "ff_labels.npy"), ff_labels)
    evaluate(ff_scores, ff_labels, "FF++ (all subsets)")

    cdf_scores, cdf_labels = load_cdf(detector, predictor)
    np.save(os.path.join(BASE, "cdf_scores.npy"), cdf_scores)
    np.save(os.path.join(BASE, "cdf_labels.npy"), cdf_labels)
    evaluate(cdf_scores, cdf_labels, "CDFv2")

    print("\nDone. .npy files saved to", BASE)

if __name__ == "__main__":
    main()
