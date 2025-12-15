import cv2
import numpy as np
import argparse
import os

ANIM_TAIL_FRAMES = 30


def analyze_rally(video_path, roi, rally_start, rally_end, debug=False):
    effective_end = max(rally_start, rally_end - ANIM_TAIL_FRAMES)

    band_counts, white_positions_for_band, hit_sequence = process_rally_white_and_bands(
        video_path,
        roi,
        rally_start,
        effective_end,
        initial_white=None,
        debug=debug
    )

    total_bands = sum(band_counts.values())

    ball_positions = track_balls_for_rally(
        video_path,
        roi,
        rally_start,
        effective_end
    )

    collisions_white_red = detect_collisions_between(
        ball_positions, "white", "red"
    )
    collisions_white_yellow = detect_collisions_between(
        ball_positions, "white", "yellow"
    )

    hit_red = len(collisions_white_red) > 0
    hit_yellow = len(collisions_white_yellow) > 0

    success = (total_bands >= 3) and hit_red and hit_yellow

    result = {
        "start": rally_start,
        "end": rally_end,
        "effective_end": effective_end,
        "band_counts": band_counts,
        "band_hits": hit_sequence,
        "white_red_collisions": collisions_white_red,
        "white_yellow_collisions": collisions_white_yellow,
        "hit_red": hit_red,
        "hit_yellow": hit_yellow,
        "success": success,
        "white_track": ball_positions.get("white", {}),
    }

    return result


def play_full_video_with_overlays(video_path, roi, rallies, rally_results,
                                  window_name="Bilardo Analiz", play_delay=20):
    bands = roi["bands"]

    frame_events = {}
    frame_white_box = {}
    summary_frames = {}

    for ridx, ((r_start, r_end), result) in enumerate(zip(rallies, rally_results)):
        effective_end = result.get("effective_end", r_end)
        band_hits = result.get("band_hits", [])
        coll_red = result.get("white_red_collisions", [])
        coll_yellow = result.get("white_yellow_collisions", [])
        white_track = result.get("white_track", {})

        for f, (cx, cy, r) in white_track.items():
            frame_white_box[f] = (cx, cy, r)

        for side, f in band_hits:
            if f > effective_end:
                continue
            frame_events.setdefault(f, []).append(f"BAND_{side.upper()}")

        for f in coll_red:
            if f > effective_end:
                continue
            frame_events.setdefault(f, []).append("WHITE-RED COLLISION")

        for f in coll_yellow:
            if f > effective_end:
                continue
            frame_events.setdefault(f, []).append("WHITE-YELLOW COLLISION")

        summary_frames[effective_end] = ridx

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis = frame.copy()
        h, w = vis.shape[:2]

        ix, iy, iw, ih = roi["inner"]
        cv2.rectangle(vis, (ix, iy), (ix + iw, iy + ih), (0, 255, 0), 1)

        cv2.line(vis, (bands["left"], bands["top"]), (bands["left"], bands["bottom"]), (255, 0, 0), 2)
        cv2.line(vis, (bands["right"], bands["top"]), (bands["right"], bands["bottom"]), (0, 0, 255), 2)
        cv2.line(vis, (bands["left"], bands["top"]), (bands["right"], bands["top"]), (0, 255, 0), 2)
        cv2.line(vis, (bands["left"], bands["bottom"]), (bands["right"], bands["bottom"]), (0, 255, 255), 2)

        if frame_idx in frame_white_box:
            cx, cy, r = frame_white_box[frame_idx]
            pad = int(max(12, 2 * r))
            x1 = max(0, int(cx - pad))
            y1 = max(0, int(cy - pad))
            x2 = min(w - 1, int(cx + pad))
            y2 = min(h - 1, int(cy + pad))
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.putText(
            vis,
            f"frame {frame_idx}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"frame {frame_idx}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        events = frame_events.get(frame_idx, [])

        summary_for = summary_frames.get(frame_idx, None)

        if events:
            overlay = vis.copy()
            cv2.rectangle(overlay, (40, 60), (w - 40, 180), (0, 0, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

            y0 = 90
            for e in events:
                cv2.putText(
                    vis,
                    e,
                    (60, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y0 += 30

            cv2.imshow(window_name, vis)
            print(f"[DEBUG VIS] frame {frame_idx}: events={events}")
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                break

        elif summary_for is not None:
            result = rally_results[summary_for]
            band_counts = result["band_counts"]
            total_bands = sum(band_counts.values())
            hit_red = result["hit_red"]
            hit_yellow = result["hit_yellow"]
            success = result["success"]

            overlay = vis.copy()
            cv2.rectangle(overlay, (40, 60), (w - 40, 220), (0, 0, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

            y0 = 90
            lines = [
                f"Ralli {summary_for + 1}",
                f"Bands: L={band_counts['left']}, R={band_counts['right']}, T={band_counts['top']}, B={band_counts['bottom']}  (sum={total_bands})",
                f"Hit RED   : {'YES' if hit_red else 'NO'}",
                f"Hit YELLOW: {'YES' if hit_yellow else 'NO'}",
                f"RESULT: {'SUCCESS' if success else 'FAIL'}",
                "Devam icin bir tusa basin (ESC cikis)",
            ]
            for line in lines:
                cv2.putText(
                    vis,
                    line,
                    (60, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y0 += 25

            cv2.imshow(window_name, vis)
            print(f"[DEBUG VIS] frame {frame_idx}: Ralli {summary_for + 1} ozeti")
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                break

        else:
            cv2.imshow(window_name, vis)
            key = cv2.waitKey(play_delay) & 0xFF
            if key == 27:
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def is_separator_frame_template(frame, template_gray, match_threshold=0.8, debug=False) -> bool:
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fh, fw = frame_gray.shape
    th, tw = template_gray.shape

    tpl = template_gray
    if th > fh or tw > fw:
        scale = min(fh / th, fw / tw) * 0.9
        new_size = (max(1, int(tw * scale)), max(1, int(th * scale)))
        tpl = cv2.resize(template_gray, new_size)
        th, tw = tpl.shape

    res = cv2.matchTemplate(frame_gray, tpl, cv2.TM_CCOEFF_NORMED)
    score = res.max()

    if debug:
        print(f"[DEBUG] template_match_score = {score:.3f}")

    return score >= match_threshold


def find_rallies(
        video_path: str,
        template_path: str,
        match_threshold: float = 0.8,
        expand_frames: int = 29,
        min_rally_length: int = 80,
        debug_every_n: int = 0
):
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Separator template bulunamadı: {template_path}")

    template_bgr = cv2.imread(template_path)
    if template_bgr is None:
        raise RuntimeError(f"Separator template okunamadı: {template_path}")

    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    frame_idx = 0
    core_sep_flags = []
    core_sep_indices = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        debug = (debug_every_n > 0 and frame_idx % debug_every_n == 0)

        is_core_sep = is_separator_frame_template(
            frame,
            template_gray,
            match_threshold=match_threshold,
            debug=debug
        )

        core_sep_flags.append(is_core_sep)
        if is_core_sep:
            core_sep_indices.append(frame_idx)

        frame_idx += 1

    total_frames = frame_idx
    cap.release()

    if not core_sep_indices:
        print("Uyarı: Hiç core separator frame bulunamadı.")
        return [(0, total_frames - 1)], [], total_frames

    is_sep = [False] * total_frames
    for i, flag in enumerate(core_sep_flags):
        if flag:
            start = max(0, i - expand_frames)
            end = min(total_frames - 1, i)
            for j in range(start, end + 1):
                is_sep[j] = True

    final_sep_indices = [i for i, f in enumerate(is_sep) if f]

    rallies = []
    in_rally = False
    rally_start = 0

    for i in range(total_frames):
        if not is_sep[i]:
            if not in_rally:
                in_rally = True
                rally_start = i
        else:
            if in_rally:
                in_rally = False
                rally_end = i - 1
                length = rally_end - rally_start + 1
                if length >= min_rally_length:
                    rallies.append((rally_start, rally_end))

    if in_rally:
        rally_end = total_frames - 30
        length = rally_end - rally_start + 1
        if length >= min_rally_length:
            rallies.append((rally_start, rally_end))

    return rallies, final_sep_indices, total_frames


def detect_table_roi(frame, margin: int = 8, band_inset: int = 12):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_table = np.array([85, 80, 80], dtype=np.uint8)
    upper_table = np.array([110, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_table, upper_table)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Masa konturu bulunamadı, HSV aralığını güncellemek gerekebilir.")

    table_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(table_contour)

    ix = x + margin
    iy = y + margin
    iw = max(1, w - 2 * margin)
    ih = max(1, h - 2 * margin)

    bx = ix + band_inset
    by = iy + band_inset
    bw = max(1, iw - 2 * band_inset)
    bh = max(1, ih - 2 * band_inset)

    roi = {
        "outer": (x, y, w, h),
        "inner": (ix, iy, iw, ih),
        "bands": {
            "left": bx,
            "right": bx + bw - 1,
            "top": by,
            "bottom": by + bh - 1,
        },
    }

    return roi, mask


def get_frame_at(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"{frame_idx} indexli frame okunamadı.")

    return frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="3 top bilardo videosunda ralli sınırlarını separator template'ine göre bulur."
    )
    parser.add_argument("video")
    parser.add_argument("--template", default="frame_000000.jpg")
    parser.add_argument("--match-threshold", type=float, default=0.8)
    parser.add_argument("--expand-frames", type=int, default=29)
    parser.add_argument("--min-rally-length", type=int, default=90)
    parser.add_argument("--debug-n", type=int, default=0)
    return parser.parse_args()


def detect_balls_in_frame(frame, roi):
    ix, iy, iw, ih = roi["inner"]
    table_crop = frame[iy:iy + ih, ix:ix + iw].copy()

    hsv = cv2.cvtColor(table_crop, cv2.COLOR_BGR2HSV)

    balls = {
        "red": None,
        "yellow": None,
        "white": None,
    }

    def find_largest_contour(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area <= 0:
            return None, None, None
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        return x, y, radius

    lower_red1 = np.array([0, 120, 120], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([160, 120, 120], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)

    kernel = np.ones((3, 3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)

    rx, ry, rr = find_largest_contour(mask_red)
    if rx is not None:
        balls["red"] = {
            "center": (int(ix + rx), int(iy + ry)),
            "radius": float(rr),
        }

    lower_yellow = np.array([15, 150, 150], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE, kernel, iterations=2)

    yx, yy, yr = find_largest_contour(mask_y)
    if yx is not None:
        balls["yellow"] = {
            "center": (int(ix + yx), int(iy + yy)),
            "radius": float(yr),
        }

    lower_white = np.array([0, 0, 210], dtype=np.uint8)
    upper_white = np.array([180, 80, 255], dtype=np.uint8)

    mask_w = cv2.inRange(hsv, lower_white, upper_white)
    mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cand_center = None
    cand_radius = None
    max_score = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)

        if not (5 < radius < 20):
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        circularity = 4.0 * np.pi * area / (peri * peri)
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        if rw == 0 or rh == 0:
            continue
        aspect = max(rw, rh) / float(min(rw, rh))

        if circularity < 0.6:
            continue
        if aspect > 1.8:
            continue

        score = area * circularity
        if score > max_score:
            max_score = score
            cand_center = (x, y)
            cand_radius = radius

    if cand_center is not None:
        wx, wy = cand_center
        balls["white"] = {
            "center": (int(ix + wx), int(iy + wy)),
            "radius": float(cand_radius),
        }

    return balls


def process_rally_white_and_bands(video_path, roi, rally_start, rally_end,
                                  initial_white=None, debug=False):
    bands = roi["bands"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, rally_start)

    prev_center = None
    prev_radius = 8.0
    if initial_white is not None:
        prev_center = initial_white["center"]
        prev_radius = initial_white["radius"]

    positions = {}

    for idx in range(rally_start, rally_end + 1):
        ret, frame = cap.read()
        if not ret:
            break

        balls = detect_balls_in_frame(frame, roi)
        info = balls["white"]

        if info is not None:
            cx, cy = info["center"]
            r = info["radius"]
            prev_center = (cx, cy)
            prev_radius = r
        elif prev_center is not None:
            cx, cy = prev_center
            r = prev_radius
        else:
            continue

        positions[idx] = (cx, cy, r)

    cap.release()

    if not positions:
        return {s: 0 for s in ["left", "right", "top", "bottom"]}, {}, []

    frames_sorted = sorted(positions.keys())
    distances_per_side = {side: [] for side in ["left", "right", "top", "bottom"]}

    for f in frames_sorted:
        cx, cy, r = positions[f]

        dist_left = cx - bands["left"]
        dist_right = bands["right"] - cx
        dist_top = cy - bands["top"]
        dist_bottom = bands["bottom"] - cy

        distances_per_side["left"].append((f, float(max(dist_left, 0.0))))
        distances_per_side["right"].append((f, float(max(dist_right, 0.0))))
        distances_per_side["top"].append((f, float(max(dist_top, 0.0))))
        distances_per_side["bottom"].append((f, float(max(dist_bottom, 0.0))))

    band_counts = {side: 0 for side in distances_per_side}
    hit_sequence = []

    BALL_R = 10.0

    ENTER_TH = BALL_R + 3.0
    EXIT_TH = BALL_R + 7.0
    APPROACH_DELTA = 6.0
    MIN_FRAME_GAP = 10

    for side, series in distances_per_side.items():
        if len(series) < 2:
            continue

        state = "far"
        last_far_dist = None
        last_hit_frame = -99999

        for f, d in series:
            if state == "far":
                if last_far_dist is None or d > last_far_dist:
                    last_far_dist = d

                if d <= ENTER_TH and last_far_dist is not None:
                    if (last_far_dist - d) >= APPROACH_DELTA and (f - last_hit_frame) >= MIN_FRAME_GAP:
                        band_counts[side] += 1
                        last_hit_frame = f
                        state = "near"
                        if debug:
                            print(f"[DEBUG] HIT {side} at frame {f} (last_far={last_far_dist:.1f}, d={d:.1f})")
                        hit_sequence.append((side, f))
                        last_far_dist = None

            else:
                if d >= EXIT_TH:
                    state = "far"
                    last_far_dist = d

    hit_sequence.sort(key=lambda x: x[1])

    return band_counts, positions, hit_sequence


def track_balls_for_rally(video_path, roi, rally_start, rally_end):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, rally_start)

    positions = {
        "white": {},
        "yellow": {},
        "red": {},
    }

    prev = {
        "white": None,
        "yellow": None,
        "red": None,
    }

    for idx in range(rally_start, rally_end + 1):
        ret, frame = cap.read()
        if not ret:
            break

        balls = detect_balls_in_frame(frame, roi)

        for name in ["white", "yellow", "red"]:
            info = balls.get(name)
            if info is not None:
                cx, cy = info["center"]
                r = info["radius"]
                prev[name] = (cx, cy, r)
                positions[name][idx] = (cx, cy, r)
            else:
                pass

    cap.release()
    positions = interpolate_ball_tracks(positions, max_gap=3)

    return positions


def interpolate_ball_tracks(ball_positions, max_gap=3):
    new_tracks = {}

    for name, track in ball_positions.items():
        frames = sorted(track.keys())
        if len(frames) < 2:
            new_tracks[name] = dict(track)
            continue

        filled = dict(track)

        for i in range(len(frames) - 1):
            f1 = frames[i]
            f2 = frames[i + 1]
            gap = f2 - f1

            if 1 < gap <= max_gap:
                x1, y1, r1 = track[f1]
                x2, y2, r2 = track[f2]

                for f in range(f1 + 1, f2):
                    t = (f - f1) / float(gap)
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    r = r1 + t * (r2 - r1)
                    filled[f] = (x, y, r)

        new_tracks[name] = filled

    return new_tracks


def detect_speed_jump_collisions_single(ball_pos,
                                        start_frame,
                                        end_frame,
                                        pre_window=4,
                                        max_pre_speed=1.0,
                                        min_post_speed=8.0):
    frames = sorted(f for f in ball_pos.keys()
                    if start_frame <= f <= end_frame)
    if len(frames) < pre_window + 2:
        return []

    speeds = {}
    for i in range(1, len(frames)):
        f1, f2 = frames[i - 1], frames[i]
        x1, y1, _ = ball_pos[f1]
        x2, y2, _ = ball_pos[f2]
        d = float(np.hypot(x2 - x1, y2 - y1))
        dt = max(1, f2 - f1)
        speeds[f2] = d / dt

    collision_frames = []

    for i in range(pre_window, len(frames)):
        f = frames[i]
        post_speed = speeds.get(f, 0.0)

        prev_fs = frames[i - pre_window:i]
        prev_speeds = [speeds.get(ff, 0.0) for ff in prev_fs if ff in speeds]
        if not prev_speeds:
            continue

        pre_speed_avg = float(np.mean(prev_speeds))

        if pre_speed_avg < max_pre_speed and post_speed >= min_post_speed:
            collision_frames.append(f)
            break

    return collision_frames


def detect_collisions_between(ball_positions, name1, name2,
                              ball_radius=10.0,
                              dist_factor=2.2,
                              min_frame_gap=8,
                              move_window=5,
                              move_thresh=1.5):
    pos1 = ball_positions[name1]
    pos2 = ball_positions[name2]

    common_frames = sorted(set(pos1.keys()) & set(pos2.keys()))
    if len(common_frames) < 2:
        return []

    series = []
    for f in common_frames:
        x1, y1, _ = pos1[f]
        x2, y2, _ = pos2[f]
        d = float(np.hypot(x1 - x2, y1 - y2))
        series.append((f, d))

    base_th = dist_factor * (2.0 * ball_radius)
    if name1 == "white" and name2 == "yellow":
        COLLISION_TH = 3.0 * (2.0 * ball_radius)
    else:
        COLLISION_TH = base_th

    below_idx = [i for i, (_, d) in enumerate(series) if d <= COLLISION_TH]
    candidate_frames = []
    last_candidate_frame = -99999

    if below_idx:
        start = below_idx[0]
        current = [start]

        def add_segment(seg_indices):
            nonlocal last_candidate_frame
            if not seg_indices:
                return
            best_i = seg_indices[0]
            best_d = series[best_i][1]
            for idx in seg_indices[1:]:
                _, d = series[idx]
                if d < best_d:
                    best_d = d
                    best_i = idx
            f_candidate = series[best_i][0]

            if f_candidate - last_candidate_frame >= min_frame_gap:
                candidate_frames.append(f_candidate)
                last_candidate_frame = f_candidate

        for idx in below_idx[1:]:
            if idx == current[-1] + 1:
                current.append(idx)
            else:
                add_segment(current)
                current = [idx]
        add_segment(current)

    def ball_moved_around_frame(ball_pos, frame, window, disp_thresh):
        pre_frames = [f for f in ball_pos.keys() if frame - window <= f < frame]
        post_frames = [f for f in ball_pos.keys() if frame < f <= frame + window]

        if not pre_frames or not post_frames:
            return False

        pre_frames_sorted = sorted(pre_frames)
        post_frames_sorted = sorted(post_frames)

        def avg_pos(frames):
            xs, ys = [], []
            for ff in frames:
                x, y, _ = ball_pos[ff]
                xs.append(x)
                ys.append(y)
            return (float(np.mean(xs)), float(np.mean(ys)))

        def avg_speed(frames):
            if len(frames) < 2:
                return 0.0
            frames_sorted = sorted(frames)
            dists = []
            times = []
            for i in range(1, len(frames_sorted)):
                f1 = frames_sorted[i - 1]
                f2 = frames_sorted[i]
                x1, y1, _ = ball_pos[f1]
                x2, y2, _ = ball_pos[f2]
                d = float(np.hypot(x2 - x1, y2 - y1))
                dt = max(1, f2 - f1)
                dists.append(d)
                times.append(dt)
            return float(sum(dists) / sum(times))

        pre_x, pre_y = avg_pos(pre_frames_sorted)
        post_x, post_y = avg_pos(post_frames_sorted)
        displacement = float(np.hypot(post_x - pre_x, post_y - pre_y))

        pre_speed = avg_speed(pre_frames_sorted)
        post_speed = avg_speed(post_frames_sorted)

        speed_delta = 0.15
        min_post_speed = 0.15

        if displacement < disp_thresh:
            return False
        if post_speed < min_post_speed:
            return False
        if (post_speed - pre_speed) < speed_delta:
            return False

        return True

    collisions = []
    last_collision_frame2 = -99999

    for fc in candidate_frames:
        moved = ball_moved_around_frame(pos2, fc, move_window, move_thresh)
        if not moved:
            continue
        if fc - last_collision_frame2 < min_frame_gap:
            continue
        collisions.append(fc)
        last_collision_frame2 = fc

    if not collisions:
        fb = detect_speed_jump_collisions_single(
            pos2,
            start_frame=common_frames[0],
            end_frame=common_frames[-1],
            pre_window=4,
            max_pre_speed=1.0,
            min_post_speed=8.0
        )
        if fb:
            collisions.extend(fb)

    return collisions


def analyze_rally(video_path, roi, rally_start, rally_end, debug=False):
    band_counts, white_positions_for_band, hit_sequence = process_rally_white_and_bands(
        video_path,
        roi,
        rally_start,
        rally_end,
        initial_white=None,
        debug=debug
    )

    total_bands = sum(band_counts.values())

    ball_positions = track_balls_for_rally(video_path, roi, rally_start, rally_end)

    collisions_white_red = detect_collisions_between(ball_positions, "white", "red")
    collisions_white_yellow = detect_collisions_between(ball_positions, "white", "yellow")

    hit_red = len(collisions_white_red) > 0
    hit_yellow = len(collisions_white_yellow) > 0

    success = (total_bands >= 3) and hit_red and hit_yellow

    result = {
        "band_counts": band_counts,
        "band_hits": hit_sequence,
        "white_red_collisions": collisions_white_red,
        "white_yellow_collisions": collisions_white_yellow,
        "hit_red": hit_red,
        "hit_yellow": hit_yellow,
        "success": success,
        "ball_positions": ball_positions,
    }

    return result


def main():
    args = parse_args()

    rallies, sep_indices, total_frames = find_rallies(
        args.video,
        args.template,
        match_threshold=args.match_threshold,
        expand_frames=args.expand_frames,
        min_rally_length=args.min_rally_length,
        debug_every_n=args.debug_n
    )

    first_start, first_end = rallies[0]
    sample_idx = (first_start + first_end) // 2

    frame = get_frame_at(args.video, sample_idx)
    roi, _ = detect_table_roi(frame, margin=8)

    print("\n=== Tüm ralliler için özet analiz ===")
    total_rallies = len(rallies)
    success_count = 0
    total_bands_sum = 0
    all_results = []

    rally_results = []
    for i, (start, end) in enumerate(rallies, start=1):
        result = analyze_rally(args.video, roi, start, end, debug=False)
        rally_results.append(result)
        all_results.append(result)

        band_counts = result["band_counts"]
        band_hits = result["band_hits"]
        coll_red = result["white_red_collisions"]
        coll_yellow = result["white_yellow_collisions"]

        total_bands = sum(band_counts.values())
        hit_red = result["hit_red"]
        hit_yellow = result["hit_yellow"]
        success = result["success"]

        hit_order = [side for (side, f) in band_hits]

        print(f"\nRalli {i}: frame {start}-{end}")
        print(f"  Band sayıları   : {band_counts} (toplam={total_bands})")
        print(f"  Band sırası     : {hit_order}")
        print(f"  Beyaz-KIRMIZI çarpışma frameleri: {coll_red}")
        print(f"  Beyaz-SARI   çarpışma frameleri: {coll_yellow}")
        print(f"  KIRMIZI'ya çarptı mı? {hit_red}")
        print(f"  SARI'ya çarptı mı?   {hit_yellow}")
        print(f"  -> Bu ralli {'BAŞARILI' if success else 'BAŞARISIZ'}")

        total_bands_sum += total_bands
        if success:
            success_count += 1

    play_full_video_with_overlays(args.video, roi, rallies, all_results)

    print("\n=== GENEL SONUÇLAR ===")
    print(f"Toplam ralli sayısı : {total_rallies}")
    print(f"Başarılı ralli sayısı: {success_count}")


if __name__ == "__main__":
    main()
