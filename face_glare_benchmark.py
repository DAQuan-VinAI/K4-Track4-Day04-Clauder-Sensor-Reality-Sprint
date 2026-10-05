"""
Face detection under glare - Sensor Reality Sprint (T1: Camera degradation, xe ADAS)
Tinh nang : Phat hien mat tai xe (buoc dau tien cua he thong canh bao buon ngu / mat tap trung)
Sensor    : Camera cabin RGB (webcam / dien thoai)
Failure   : Nang chieu vao mat -> vung mat chay sang + loa toan khung

Cach chay (chi can OpenCV, khong tai model):
    pip install "opencv-python<5" numpy pandas matplotlib
    python face_glare_benchmark.py --images ./images --out ./results
    python face_glare_benchmark.py --video ./videos/clip.mp4 --out ./results

Baseline : anh goc (glare level 0)
Degraded : CUNG anh + glare 4 muc. Giu nguyen detector va tham so.
Pseudo-GT: mat phat hien duoc tren anh goc (khong co nhan that).
"""
import argparse
import glob
import os
import platform
import sys
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GLARE_LEVELS = [0, 1, 2, 3, 4]
GLARE_PEAK = [0, 60, 120, 180, 240]   # do sang cong them o tam vet loa (0-255)
VEIL = [0, 10, 20, 30, 45]             # loa phu toan khung
IOU_MATCH = 0.5
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")   # so khop khong phan biet hoa/thuong
DETECTOR_PARAMS = dict(scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

if hasattr(cv2, "CascadeClassifier"):
    DETECTOR_NAME = "OpenCV Haar cascade frontalface_default"
    _CASCADE = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

    def _detect(gray):
        return _CASCADE.detectMultiScale(gray, **DETECTOR_PARAMS)
else:
    # OpenCV 5 da bo Haar cascade -> dung LBP cascade co san trong scikit-image (khong can tai model)
    try:
        from skimage import data as _skdata
        from skimage.feature import Cascade as _SkCascade
    except ImportError:
        sys.exit(f"OpenCV {cv2.__version__} khong co Haar cascade. Chay: pip install \"opencv-python<5\" "
                 "hoac pip install scikit-image")
    DETECTOR_NAME = "scikit-image LBP frontal face cascade"
    _SK = _SkCascade(_skdata.lbp_frontal_face_cascade_filename())

    def _detect(gray):
        dets = _SK.detect_multi_scale(img=gray, scale_factor=1.2, step_ratio=1, min_size=DETECTOR_PARAMS["minSize"],
                                      max_size=(gray.shape[1], gray.shape[0]), min_neighbor_number=4)
        return [(d["c"], d["r"], d["width"], d["height"]) for d in dets]


def add_glare(img, level, box=None):
    """Vet loa hinh Gaussian dat lech sang nua trai khuon mat + loa phu toan anh."""
    if level == 0:
        return img.copy()
    h, w = img.shape[:2]
    if box is not None:
        x, y, bw, bh = box
        cx, cy, r = x + 0.3 * bw, y + 0.45 * bh, 0.6 * bw
    else:
        cx, cy, r = 0.4 * w, 0.4 * h, 0.2 * w
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2)))
    out = img.astype(np.float32) + (GLARE_PEAK[level] * blob + VEIL[level])[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = time.perf_counter()
    faces = _detect(gray)
    return [tuple(map(int, f)) for f in faces], (time.perf_counter() - t) * 1000


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def face_stats(img, box):
    if box is None:
        return np.nan, np.nan
    x, y, w, h = box
    roi = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    return float(roi.mean()), float((roi >= 250).mean() * 100)


def load_inputs(args):
    items, skipped = [], []
    if args.images:
        # liet ke 1 lan roi loc theo duoi viet thuong -> khong sot .PNG/.JPEG, khong trung file tren Windows
        files = sorted(p for p in glob.glob(os.path.join(args.images, "*")) if os.path.isfile(p))
        paths = [p for p in files if p.lower().endswith(IMG_EXTS)]
        skipped = [os.path.basename(p) for p in files
                   if not p.lower().endswith(IMG_EXTS) and not os.path.basename(p).startswith(".")]
        for p in paths[: args.max]:
            im = cv2.imread(p)
            if im is not None:
                items.append((os.path.basename(p), im))
            else:
                skipped.append(os.path.basename(p))
    if args.video:
        cap = cv2.VideoCapture(args.video)
        i = 0
        while len(items) < args.max:
            ok, f = cap.read()
            if not ok:
                break
            if i % args.step == 0:
                items.append((f"{os.path.basename(args.video)}#{i}", f))
            i += 1
        cap.release()
    out = []
    for name, im in items:      # dua ve cung chieu rong de cong bang
        if im.shape[1] != 640:
            im = cv2.resize(im, (640, int(im.shape[0] * 640 / im.shape[1])))
        out.append((name, im))
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images")
    ap.add_argument("--video")
    ap.add_argument("--out", default="results")
    ap.add_argument("--step", type=int, default=10, help="video: lay 1 frame moi N frame")
    ap.add_argument("--max", type=int, default=100)
    args = ap.parse_args()
    if not (args.images or args.video):
        sys.exit("Can --images hoac --video")
    os.makedirs(args.out, exist_ok=True)

    log = open(os.path.join(args.out, "run_log.txt"), "w", encoding="utf-8")
    def say(m):
        print(m); log.write(m + "\n")

    items, skipped = load_inputs(args)
    say(f"Run time : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"Env      : Python {platform.python_version()} | OpenCV {cv2.__version__} | numpy {np.__version__}")
    say(f"Command  : {' '.join(sys.argv)}")
    say(f"Detector : {DETECTOR_NAME} | {DETECTOR_PARAMS}")
    say(f"Glare    : peak={GLARE_PEAK} veil={VEIL} | IoU match >= {IOU_MATCH}")
    say(f"So anh   : {len(items)}")
    if skipped:
        say(f"Bo qua   : {len(skipped)} file khong doc duoc (chi nhan {'/'.join(IMG_EXTS)}; "
            f"anh HEIC can xuat sang JPG): {', '.join(skipped[:10])}")
    if not items:
        sys.exit("Khong co anh/frame nao")

    rows, examples = [], []
    for name, img in items:
        base_faces, _ = detect_faces(img)
        gt = max(base_faces, key=lambda b: b[2] * b[3]) if base_faces else None   # mat lon nhat = tai xe
        for lvl in GLARE_LEVELS:
            g = add_glare(img, lvl, gt)
            faces, ms = detect_faces(g)
            best = max(faces, key=lambda b: iou(b, gt)) if (faces and gt) else None
            best_iou = iou(best, gt) if best else 0.0
            bright, sat = face_stats(g, gt)
            rows.append(dict(image=name, level=lvl, base_has_face=gt is not None, n_faces=len(faces),
                             detected=bool(faces), matched=best_iou >= IOU_MATCH if gt else np.nan,
                             iou=best_iou if gt else np.nan,
                             # nhan nham = box khong trung voi BAT KY mat nao o anh goc (mat nguoi thu hai
                             # khong bi tinh); muc 0 luon bang 0 vi anh goc chinh la moc so sanh
                             false_pos=sum(1 for f in faces if all(iou(f, b) < IOU_MATCH for b in base_faces)),
                             face_brightness=bright, face_sat_pct=sat, latency_ms=ms))
            if gt is not None and (not examples or examples[0][0] == name):
                vis = g.copy()
                cv2.rectangle(vis, gt[:2], (gt[0] + gt[2], gt[1] + gt[3]), (255, 0, 0), 1)
                for f in faces:
                    cv2.rectangle(vis, f[:2], (f[0] + f[2], f[1] + f[3]), (0, 255, 0), 2)
                cv2.putText(vis, f"L{lvl} {'OK' if best_iou >= IOU_MATCH else 'MISS'} sat={sat:.0f}%",
                            (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                examples.append((name, lvl, vis))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "results.csv"), index=False)

    valid = df[df.base_has_face]
    say(f"Anh co mat o baseline: {valid.image.nunique()}/{df.image.nunique()} (chi tinh recall tren cac anh nay)")
    summary = valid.groupby("level").agg(
        recall_proxy_pct=("matched", lambda s: 100 * s.mean()),
        mean_iou=("iou", "mean"),
        face_sat_pct=("face_sat_pct", "mean"),
        face_brightness=("face_brightness", "mean"),
        false_pos_per_img=("false_pos", "mean"),
        latency_ms=("latency_ms", "mean"),
    ).round(2)
    summary.to_csv(os.path.join(args.out, "summary.csv"))
    say("\n=== KET QUA theo muc loa ===")
    say(summary.to_string())

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (m, t) in zip(axes, [("recall_proxy_pct", "Recall proxy (%)"), ("mean_iou", "Mean IoU"),
                                 ("face_sat_pct", "Pixel bao hoa tren mat (%)")]):
        ax.plot(summary.index, summary[m], "o-", color="#c0392b")
        ax.set_title(t); ax.set_xlabel("Muc loa"); ax.set_xticks(GLARE_LEVELS); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "metrics_vs_glare.png"), dpi=130)

    # Anh minh hoa: 1 anh x 5 muc loa
    if examples:
        first = examples[0][0]
        tiles = [v for n, _, v in examples if n == first]
        h = 220
        tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
        cv2.imwrite(os.path.join(args.out, "glare_grid.jpg"), np.hstack(tiles))

    # Failure case: anh mat som nhat (o muc loa thap nhat)
    miss = valid[valid.matched == False]
    if len(miss):
        first_miss = miss.groupby("image")["level"].min().sort_values()
        say("\n=== FAILURE CASE: anh bi mat mat o muc loa thap nhat ===")
        say(first_miss.head(5).to_string())
    say(f"\nOutput: {os.path.abspath(args.out)}")
    log.close()


if __name__ == "__main__":
    main()
