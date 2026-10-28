"""
MediaPipe frontend + temporal decision layer + graduated alerting.

Same CLI and same CSV schema as run_yolo.py, so the two frontends are
directly comparable.

Pipeline
--------
    frame -> FaceLandmarker (478 landmarks)
          -> EAR / MAR geometric ratios
          -> DrowsinessMonitor (PERCLOS, microsleeps, yawns)
          -> AlertManager (chime + speech, non-blocking)

Why EAR instead of a classifier
-------------------------------
    EAR (Eye Aspect Ratio)   = vertical eye opening / horizontal eye width
    MAR (Mouth Aspect Ratio) = vertical mouth opening / mouth width

Both are scale-invariant: they do not change when the driver moves closer
to or further from the camera.

Per-driver calibration, in a separate pass
------------------------------------------
A fixed EAR threshold does not work: eye shape varies a lot between people,
so the 0.21 copied from a 2016 paper fires constantly for some faces and
never for others.

An earlier version calibrated inline, which meant the opening seconds were
reported as UNKNOWN — on a 10s clip that discarded 30% of the data. This
version runs a short calibration pass first, then rewinds and processes
every frame with the threshold already known. Nothing is thrown away.

This is equivalent to what a live system does after its first seconds; the
only difference is that offline we can apply the result retroactively.

Usage
-----
    python src/run_mediapipe.py --video data/clip.mp4 --tag demo --alerts
    python src/run_mediapipe.py --video data/clip.mp4 --no-display --max-width 640
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from drowsiness_state import DrowsinessMonitor, Config, EyeState, AlertLevel
from alerts import AlertManager, AlertConfig

# Face Mesh landmark indices.
# EAR points in the order p1..p6 of the standard formula.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)
# mouth: inner upper lip, inner lower lip, left corner, right corner
MOUTH = (13, 14, 78, 308)

COLORS = {
    AlertLevel.AWAKE:   (0, 200, 0),
    AlertLevel.WARNING: (0, 165, 255),
    AlertLevel.DROWSY:  (0, 0, 255),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="face_landmarker.task")
    p.add_argument("--video", default=None,
    help="Path to a video file; omit to use the webcam")
    p.add_argument("--camera", type=int, default=0,
    help="Webcam device index when --video is omitted")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--tag", default="mediapipe")
    p.add_argument("--max-width", type=int, default=960)
    p.add_argument("--calib-s", type=float, default=3.0,
    help="seconds used to learn this driver's open-eye EAR")
    p.add_argument("--ear-ratio", type=float, default=0.72,
    help="closed threshold = ratio x calibrated open EAR")
    p.add_argument("--mar-thr", type=float, default=0.50)
    p.add_argument("--window", type=float, default=60.0,help="PERCLOS window in seconds")
    p.add_argument("--min-window", type=float, default=30.0,help="PERCLOS is not trusted below this observed span")
    p.add_argument("--alerts", action="store_true",
    help="enable chime and voice alerts")
    p.add_argument("--no-voice", action="store_true",
    help="chime only, no speech")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def ear(pts, idx, w, h):
    """Eye Aspect Ratio.

    Landmarks are normalised to [0,1] on each axis *independently*, so x must
    be scaled by width and y by height before measuring. Skipping this makes
    EAR depend on the video's aspect ratio — a common and silent bug.
    """
    p = [np.array([pts[i].x * w, pts[i].y * h]) for i in idx]
    horiz = np.linalg.norm(p[0] - p[3])
    if horiz < 1e-6:
        return None
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2.0 * horiz)


def mar(pts, idx, w, h):
    p = [np.array([pts[i].x * w, pts[i].y * h]) for i in idx]
    width = np.linalg.norm(p[2] - p[3])
    if width < 1e-6:
        return None
    return np.linalg.norm(p[0] - p[1]) / width


def open_capture(path, max_width):
    source = int(path) if str(path).isdigit() else path
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise SystemExit(f"Cannot open video or camera: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if sw <= 0 or sh <= 0:
        cap.release()
        raise SystemExit(f"Invalid video dimensions for input: {path}")

    scale = min(1.0, max_width / sw)

    return cap, fps, sw, sh, n, scale, int(sw * scale), int(sh * scale)


def calibrate(args, fps, w, h, scale):
    """Pass 1: measure this driver's open-eye EAR. Writes nothing."""
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
    )
    cap = cv2.VideoCapture(args.video)
    vals, idx = [], 0
    with vision.FaceLandmarker.create_from_options(opts) as lmk:
        while idx / fps < args.calib_s:
            ok, frame = cap.read()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if res.face_landmarks:
                lm = res.face_landmarks[0]
                e = [v for v in (ear(lm, RIGHT_EYE, w, h), ear(lm, LEFT_EYE, w, h))
                     if v is not None]
                if e:
                    vals.append(float(np.mean(e)))
            idx += 1
    cap.release()

    if len(vals) < 10:
        raise SystemExit(
            f"Calibration failed: only {len(vals)} usable frames in the first "
            f"{args.calib_s}s. Extend it with --calib-s, or use a clip where "
            f"the face is visible from the start."
        )
    # 75th percentile ~ the open-eye baseline, robust to blinks that happen
    # to fall inside the calibration segment.
    base = float(np.percentile(vals, 75))
    thr = base * args.ear_ratio
    print(f"Calibration: {len(vals)} frames, open EAR {base:.3f} "
          f"-> closed threshold {thr:.3f}")
    return thr


def draw(frame, st, fps, latency_ms, ear_val, thr, last_alert):
    h, w = frame.shape[:2]
    color = COLORS[st.level]
    cv2.rectangle(frame, (0, 0), (w, 92), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)

    cv2.putText(frame, st.level.name, (14, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    e = f"{ear_val:.3f}" if ear_val is not None else "  -  "
    pc = f"{st.perclos:.1%}" if st.perclos_valid else "n/a"
    cv2.putText(frame, f"PERCLOS {pc:>6}   closure {st.current_closure_s:4.1f}s"
                       f"   micro {st.microsleeps}   yawns {st.yawns_in_window}"
                       f"   EAR {e}/{thr:.3f}",
                (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    cv2.putText(frame, f"{fps:5.1f} FPS   {latency_ms:5.1f} ms   "
                       f"coverage {st.coverage:.0%}   window {st.observed_s:4.0f}s",
                (14, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

    bar_w, x, y = 260, w - 280, 20
    cv2.rectangle(frame, (x, y), (x + bar_w, y + 14), (60, 60, 60), -1)
    if st.perclos_valid:
        cv2.rectangle(frame, (x, y),
                      (x + int(bar_w * min(st.perclos, 1.0)), y + 14), color, -1)
    cv2.line(frame, (x + int(bar_w * .15), y - 3), (x + int(bar_w * .15), y + 17),
             (255, 255, 255), 1)

    if st.reasons:
        cv2.putText(frame, " | ".join(st.reasons[:2]), (14, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if last_alert:
        cv2.putText(frame, f"ALERT: {last_alert}", (14, h - 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if not os.path.exists(a.model):
        raise SystemExit(f"Not found: {a.model}")

    live = a.video is None
    source = a.camera if live else a.video

    cap, fps, sw, sh, n_frames, scale, w, h = open_capture(
        source, a.max_width
    )

    if live:
        fps = 30.0
        n_frames = 0
        print(f"Webcam {a.camera}: {sw}x{sh}; calibrating for {a.calib_s}s")
    else:
        if not os.path.exists(a.video):
            cap.release()
            raise SystemExit(f"Not found: {a.video}")
        print(f"Video: {sw}x{sh} -> {w}x{h}, {fps:.1f} fps, "
              f"{n_frames} frames")

    def read_frame():
        ok, frame = cap.read()
        if not ok:
            return None
        if scale < 1.0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        return frame

    # Calibration uses a separate pass for files and the initial frames
    # for a live camera.
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=a.model),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
    )

    vals = []
    if live:
        calib_start = time.perf_counter()
        with vision.FaceLandmarker.create_from_options(opts) as lmk:
            while time.perf_counter() - calib_start < a.calib_s:
                frame = read_frame()
                if frame is None:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = lmk.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                )
                if res.face_landmarks:
                    lm = res.face_landmarks[0]
                    eyes = [
                        v for v in (
                            ear(lm, RIGHT_EYE, w, h),
                            ear(lm, LEFT_EYE, w, h)
                        ) if v is not None
                    ]
                    if eyes:
                        vals.append(float(np.mean(eyes)))
                if not a.no_display:
                    cv2.putText(frame, "CALIBRATING: keep eyes open",
                                (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 255), 2)
                    cv2.imshow("drowsiness", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
    else:
        cap.release()
        ear_thr = calibrate(a, fps, w, h, scale)
        cap, *_ = open_capture(source, a.max_width)

    if live:
        if len(vals) < 10:
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit(
                "Calibration failed: face not detected consistently. "
                "Improve lighting and keep your face visible, then retry."
            )
        baseline = float(np.percentile(vals, 75))
        ear_thr = baseline * a.ear_ratio
        print(f"Calibration complete: EAR={baseline:.3f}, "
              f"closed threshold={ear_thr:.3f}")

    writer = cv2.VideoWriter(
        os.path.join(a.out_dir, f"{a.tag}_annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    csv_path = os.path.join(a.out_dir, f"{a.tag}_metrics.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_file)
    csv_w.writerow([
        "frame", "t_video", "eye_state", "mouth_open", "level",
        "perclos", "perclos_valid", "observed_s", "coverage",
        "closure_s", "microsleeps", "yawns", "alert",
        "latency_ms", "ear", "mar"
    ])

    monitor = DrowsinessMonitor(
        Config(window_s=a.window, perclos_min_window_s=a.min_window)
    )
    alerts = (
        AlertManager(AlertConfig(enable_voice=not a.no_voice))
        if a.alerts else None
    )

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=a.model),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
    )

    latencies = []
    idx = 0
    disp_fps = 0.0
    last_alert = ""
    wall_start = time.perf_counter()

    try:
        with vision.FaceLandmarker.create_from_options(opts) as landmarker:
            while True:
                frame = read_frame()
                if frame is None:
                    break

                t_video = idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb
                )

                t0 = time.perf_counter()
                res = landmarker.detect_for_video(
                    mp_img, int(t_video * 1000)
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                latencies.append(latency_ms)

                ear_val = mar_val = None
                eye = EyeState.UNKNOWN
                mouth_open = False

                if res.face_landmarks:
                    lm = res.face_landmarks[0]
                    eye_vals = [
                        v for v in (
                            ear(lm, RIGHT_EYE, w, h),
                            ear(lm, LEFT_EYE, w, h)
                        ) if v is not None
                    ]
                    if eye_vals:
                        ear_val = float(np.mean(eye_vals))
                        eye = (
                            EyeState.CLOSED
                            if ear_val < ear_thr else EyeState.OPEN
                        )

                    mar_val = mar(lm, MOUTH, w, h)
                    if mar_val is not None:
                        mouth_open = mar_val > a.mar_thr

                    color = (
                        (0, 0, 255) if eye is EyeState.CLOSED
                        else (0, 220, 0)
                    )
                    for i in RIGHT_EYE + LEFT_EYE + MOUTH:
                        cv2.circle(
                            frame,
                            (int(lm[i].x * w), int(lm[i].y * h)),
                            2, color, -1
                        )

                st = monitor.update(t_video, eye, mouth_open)
                fired = ""

                if alerts is not None:
                    severity = alerts.notify(t_video, st)
                    if severity is not None:
                        fired = severity.name
                        last_alert = fired
                        print(
                            f"t={t_video:.2f}s {st.level.name} "
                            f"-> ALERT {fired}"
                        )

                if idx % 10 == 0:
                    disp_fps = 1000 / max(
                        float(np.mean(latencies[-30:])), 1e-6
                    )

                frame = draw(
                    frame, st, disp_fps, latency_ms,
                    ear_val, ear_thr, last_alert
                )
                writer.write(frame)

                csv_w.writerow([
                    idx, f"{t_video:.3f}", eye.value, int(mouth_open),
                    st.level.name, f"{st.perclos:.4f}",
                    int(st.perclos_valid), f"{st.observed_s:.2f}",
                    f"{st.coverage:.3f}", f"{st.current_closure_s:.3f}",
                    st.microsleeps, st.yawns_in_window, fired,
                    f"{latency_ms:.2f}",
                    f"{ear_val:.4f}" if ear_val is not None else "",
                    f"{mar_val:.4f}" if mar_val is not None else ""
                ])

                if not a.no_display:
                    cv2.imshow("drowsiness", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("a"):
                        monitor.acknowledge(t_video)
                        last_alert = "acknowledged"

                idx += 1

    finally:
        cap.release()
        writer.release()
        csv_file.close()
        cv2.destroyAllWindows()

        if alerts is not None:
            time.sleep(1)
            print("\n" + alerts.report())
            alerts.close()

    if latencies:
        arr = np.asarray(latencies)
        wall = time.perf_counter() - wall_start
        print(f"\n=== {a.tag.upper()} ===")
        print(f"Frames processed: {idx}")
        print(f"Inference p50: {np.percentile(arr, 50):.2f} ms")
        print(f"Inference p95: {np.percentile(arr, 95):.2f} ms")
        print(f"End-to-end: {idx / max(wall, 1e-6):.1f} FPS")
        print(f"Video: {os.path.join(a.out_dir, a.tag + '_annotated.mp4')}")
        print(f"CSV:   {csv_path}")

if __name__ == "__main__":
    main()