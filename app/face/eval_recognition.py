"""
scripts/eval_recognition.py
----------------------------
Offline evaluation of the face-recognition pipeline (YuNet -> gate -> align -> MobileFaceNet).
Use it to choose the similarity threshold from data instead of guessing.

What it does
------------
1. Embeds every usable image in datasets/faces/known/<person>/.
2. Leave-one-out: each image is scored against every identity's mean template, where its
   own identity's template is rebuilt WITHOUT that image (so it is never compared to itself).
3. Optionally scores "unknown" people from datasets/faces/unknown/ (flat images or sub-folders),
   compared against the full templates. These are the images that really tell you the
   false-accept rate -- add 20-30 different people, ideally captured with the Pi camera.
4. Sweeps thresholds and prints, for each:
     correct_id   genuine image accepted AND named correctly
     reject       genuine image not accepted (device stays silent / says "unknown")
     wrong_id     image accepted but named as the WRONG person  (worst case for the user)
     unk_accept   unknown person accepted as someone known     (also bad)
5. Prints a suggested threshold = highest impostor score seen + margin.

Usage
-----
    python scripts/eval_recognition.py
    python scripts/eval_recognition.py --unknown datasets/faces/unknown --margin 0.05
    python scripts/eval_recognition.py --no-gate        # evaluate without quality gates
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from app.face.face_alignment import align_face
from app.face.face_quality import QualityConfig, check_face

import importlib.util as _ilu  # load enroll_faces.py by path so no scripts/__init__.py is needed
_spec = _ilu.spec_from_file_location("enroll_faces", Path(__file__).resolve().parent / "enroll_faces.py")
_enroll = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_enroll)
detect_largest_face, embed_face, load_onnx_session = (
    _enroll.detect_largest_face, _enroll.embed_face, _enroll.load_onnx_session)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate face recognition and pick a threshold.")
    p.add_argument("--known", default=str(_ROOT / "datasets/faces/known"))
    p.add_argument("--unknown", default=str(_ROOT / "datasets/faces/unknown"))
    p.add_argument("--model", default=str(_ROOT / "models/face_recognition/mobilefacenet.onnx"))
    p.add_argument("--detector", default=str(_ROOT / "models/face_detection/face_detection_yunet.onnx"))
    p.add_argument("--min-score", type=float, default=0.80)
    p.add_argument("--min-face-px", type=int, default=60)
    p.add_argument("--no-gate", action="store_true", help="disable quality gates")
    p.add_argument("--margin", type=float, default=0.05, help="safety margin above worst impostor")
    return p.parse_args()


def embed_images(paths: List[Path], detector, session, qcfg, gate: bool) -> np.ndarray:
    out = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        row = detect_largest_face(detector, img)
        if row is None:
            continue
        if gate and not check_face(row, qcfg)[0]:
            continue
        aligned = align_face(img, row, output_size=112)
        if aligned is None or aligned.size == 0:
            continue
        out.append(embed_face(session, aligned, 112))
    return np.array(out) if out else np.zeros((0, 512), np.float32)


def list_images(root: Path) -> List[Path]:
    return sorted(f for f in root.rglob("*") if f.suffix.lower() in _IMAGE_EXTS)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main() -> None:
    a = parse_args()
    qcfg = QualityConfig(min_score=a.min_score, min_face_px=a.min_face_px)
    gate = not a.no_gate
    det = cv2.FaceDetectorYN.create(a.detector, "", (320, 320), 0.6, 0.3, top_k=10)
    sess = load_onnx_session(a.model)

    known: Dict[str, np.ndarray] = {}
    for d in sorted(Path(a.known).iterdir()):
        if d.is_dir():
            E = embed_images(list_images(d), det, sess, qcfg, gate)
            if len(E) >= 3:
                known[d.name] = E
    if len(known) < 2:
        sys.exit("Need at least 2 identities with >=3 usable images each.")
    names = list(known)
    print(f"Known identities (usable images): " + ", ".join(f"{n}={len(known[n])}" for n in names))

    # ---- genuine samples: leave-one-out against all templates -------------------------------
    # each record: (true_name, best_name, best_sim, own_sim)
    genuine: List[Tuple[str, str, float, float]] = []
    for n in names:
        for i in range(len(known[n])):
            tmpl = {}
            for k in names:
                X = np.delete(known[n], i, 0) if k == n else known[k]
                tmpl[k] = unit(X.mean(0))
            sims = {k: float(known[n][i] @ tmpl[k]) for k in names}
            best = max(sims, key=sims.get)
            genuine.append((n, best, sims[best], sims[n]))

    # ---- unknown samples ----------------------------------------------------------------------
    full = {k: unit(known[k].mean(0)) for k in names}
    unk_best: List[float] = []
    up = Path(a.unknown)
    if up.exists():
        U = embed_images(list_images(up), det, sess, qcfg, gate)
        for e in U:
            unk_best.append(max(float(e @ t) for t in full.values()))
        print(f"Unknown-person images used: {len(U)}")
    else:
        print(f"[!] No unknown set at {up} -> false-accept numbers below only reflect "
              f"enrolled-vs-enrolled confusion. Add 20-30 other people to get a real FAR.")

    n_gen = len(genuine)
    own = np.array([g[3] for g in genuine])
    print(f"\nTop-1 identity accuracy (ignoring threshold): "
          f"{sum(g[0] == g[1] for g in genuine)}/{n_gen}")
    print(f"Own-template similarity: mean {own.mean():.3f}  p5 {np.percentile(own, 5):.3f}  min {own.min():.3f}")

    # impostor scores = best similarity to a WRONG template (enrolled people) + best sim of unknowns
    # cross-identity impostor: each genuine image vs the OTHER people's templates
    cross_max = [max(float(known[n][i] @ full[k]) for k in names if k != n)
                 for n in names for i in range(len(known[n]))]
    imp_all = np.array(list(unk_best) + cross_max)
    worst = float(imp_all.max())

    print(f"\n{'thr':>5} | {'correct_id':>10} {'reject':>7} {'wrong_id':>8} | {'unk_accept':>10}")
    for t in np.arange(0.25, 0.71, 0.05):
        correct = sum(1 for g in genuine if g[1] == g[0] and g[2] >= t) / n_gen
        wrong = sum(1 for g in genuine if g[1] != g[0] and g[2] >= t) / n_gen
        rej = 1 - correct - wrong
        ua = (np.mean(np.array(unk_best) >= t) if unk_best else float("nan"))
        print(f"{t:5.2f} | {correct:10.3f} {rej:7.3f} {wrong:8.3f} | {ua:10.3f}")

    sugg = min(0.95, worst + a.margin)
    acc_at = np.mean([g[1] == g[0] and g[2] >= sugg for g in genuine])
    print(f"\nWorst impostor similarity seen: {worst:.3f}"
          f" ({'incl. unknown set' if unk_best else 'enrolled-vs-enrolled only'})")
    print(f"Suggested threshold = worst + margin({a.margin}) = {sugg:.2f}"
          f"  -> genuine accepted & correct: {acc_at:.1%}")
    if not unk_best:
        print("Treat this as PROVISIONAL until an unknown-person set is evaluated.")


if __name__ == "__main__":
    main()
