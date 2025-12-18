import cv2
import numpy as np
import argparse
import os

ANIM_TAIL_FRAMES = 30
SEPARATOR_FRAMES = 60

def is_separator_frame_template(frame, template_gray, match_threshold=0.8):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fh, fw = frame_gray.shape
    th, tw = template_gray.shape
    tpl = template_gray
    if th > fh or tw > fw:
        scale = min(fh / th, fw / tw) * 0.9
        new_size = (max(1, int(tw * scale)), max(1, int(th * scale)))
        tpl = cv2.resize(template_gray, new_size)
    res = cv2.matchTemplate(frame_gray, tpl, cv2.TM_CCOEFF_NORMED)
    score = res.max()
    return score >= match_threshold

def detect_table_roi(frame, margin=8, band_inset=12):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_table = np.array([85, 80, 80], dtype=np.uint8)
    upper_table = np.array([110, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_table, upper_table)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Masa konturu bulunamadi.")
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

def detect_balls_in_frame(frame, roi):
    ix, iy, iw, ih = roi["inner"]
    table_crop = frame[iy:iy + ih, ix:ix + iw].copy()
    hsv = cv2.cvtColor(table_crop, cv2.COLOR_BGR2HSV)
    balls = {"red": None, "yellow": None, "white": None}
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
        balls["red"] = {"center": (int(ix + rx), int(iy + ry)), "radius": float(rr)}
    lower_yellow = np.array([15, 150, 150], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE, kernel, iterations=2)
    yx, yy, yr = find_largest_contour(mask_y)
    if yx is not None:
        balls["yellow"] = {"center": (int(ix + yx), int(iy + yy)), "radius": float(yr)}
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
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw == 0 or bh == 0:
            continue
        aspect = max(bw, bh) / float(min(bw, bh))
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
        balls["white"] = {"center": (int(ix + wx), int(iy + wy)), "radius": float(cand_radius)}
    return balls

class BandHitDetector:
    def __init__(self, bands):
        self.bands = bands
        self.band_counts = {"left": 0, "right": 0, "top": 0, "bottom": 0}
        self.hit_sequence = []
        self.distances_per_side = {side: [] for side in ["left", "right", "top", "bottom"]}
        self.state = {side: "far" for side in ["left", "right", "top", "bottom"]}
        self.last_far_dist = {side: None for side in ["left", "right", "top", "bottom"]}
        self.last_hit_frame = {side: -99999 for side in ["left", "right", "top", "bottom"]}
        self.BALL_R = 10.0
        self.ENTER_TH = self.BALL_R + 3.0
        self.EXIT_TH = self.BALL_R + 7.0
        self.APPROACH_DELTA = 6.0
        self.MIN_FRAME_GAP = 10

    def reset(self):
        self.band_counts = {"left": 0, "right": 0, "top": 0, "bottom": 0}
        self.hit_sequence = []
        self.distances_per_side = {side: [] for side in ["left", "right", "top", "bottom"]}
        self.state = {side: "far" for side in ["left", "right", "top", "bottom"]}
        self.last_far_dist = {side: None for side in ["left", "right", "top", "bottom"]}
        self.last_hit_frame = {side: -99999 for side in ["left", "right", "top", "bottom"]}

    def process_frame(self, frame_idx, cx, cy):
        new_hits = []
        dist_left = cx - self.bands["left"]
        dist_right = self.bands["right"] - cx
        dist_top = cy - self.bands["top"]
        dist_bottom = self.bands["bottom"] - cy
        dists = {
            "left": float(max(dist_left, 0.0)),
            "right": float(max(dist_right, 0.0)),
            "top": float(max(dist_top, 0.0)),
            "bottom": float(max(dist_bottom, 0.0)),
        }
        for side, d in dists.items():
            self.distances_per_side[side].append((frame_idx, d))
            if self.state[side] == "far":
                if self.last_far_dist[side] is None or d > self.last_far_dist[side]:
                    self.last_far_dist[side] = d
                if d <= self.ENTER_TH and self.last_far_dist[side] is not None:
                    if (self.last_far_dist[side] - d) >= self.APPROACH_DELTA and (frame_idx - self.last_hit_frame[side]) >= self.MIN_FRAME_GAP:
                        self.band_counts[side] += 1
                        self.last_hit_frame[side] = frame_idx
                        self.state[side] = "near"
                        self.hit_sequence.append((side, frame_idx))
                        new_hits.append((side, frame_idx))
                        self.last_far_dist[side] = None
            else:
                if d >= self.EXIT_TH:
                    self.state[side] = "far"
                    self.last_far_dist[side] = d
        return new_hits

class CollisionDetector:
    def __init__(self, ball_radius=10.0, dist_factor=2.2, min_frame_gap=8, move_window=5, move_thresh=1.5):
        self.ball_radius = ball_radius
        self.dist_factor = dist_factor
        self.min_frame_gap = min_frame_gap
        self.move_window = move_window
        self.move_thresh = move_thresh
        self.pos_white = {}
        self.pos_red = {}
        self.pos_yellow = {}
        self.last_collision_red = -99999
        self.last_collision_yellow = -99999
        self.collisions_red = []
        self.collisions_yellow = []
        self.pending_red = []
        self.pending_yellow = []
        self.confirmed_red = set()
        self.confirmed_yellow = set()

    def reset(self):
        self.pos_white = {}
        self.pos_red = {}
        self.pos_yellow = {}
        self.last_collision_red = -99999
        self.last_collision_yellow = -99999
        self.collisions_red = []
        self.collisions_yellow = []
        self.pending_red = []
        self.pending_yellow = []
        self.confirmed_red = set()
        self.confirmed_yellow = set()

    def update_positions(self, frame_idx, balls):
        if balls["white"] is not None:
            cx, cy = balls["white"]["center"]
            r = balls["white"]["radius"]
            self.pos_white[frame_idx] = (cx, cy, r)
        if balls["red"] is not None:
            cx, cy = balls["red"]["center"]
            r = balls["red"]["radius"]
            self.pos_red[frame_idx] = (cx, cy, r)
        if balls["yellow"] is not None:
            cx, cy = balls["yellow"]["center"]
            r = balls["yellow"]["radius"]
            self.pos_yellow[frame_idx] = (cx, cy, r)

    def check_collision(self, frame_idx, name2):
        if name2 == "red":
            pos2 = self.pos_red
            last_collision = self.last_collision_red
            collision_th = self.dist_factor * (2.0 * self.ball_radius)
            pending_list = self.pending_red
            confirmed_set = self.confirmed_red
        else:
            pos2 = self.pos_yellow
            last_collision = self.last_collision_yellow
            collision_th = 3.0 * (2.0 * self.ball_radius)
            pending_list = self.pending_yellow
            confirmed_set = self.confirmed_yellow
        if frame_idx in self.pos_white and frame_idx in pos2:
            x1, y1, _ = self.pos_white[frame_idx]
            x2, y2, _ = pos2[frame_idx]
            d = float(np.hypot(x1 - x2, y1 - y2))
            if d <= collision_th:
                already_pending = any(abs(f - frame_idx) < self.min_frame_gap for f in pending_list)
                if not already_pending and frame_idx - last_collision >= self.min_frame_gap:
                    pending_list.append(frame_idx)
        confirmed_frame = None
        new_pending = []
        for pf in pending_list:
            if pf in confirmed_set:
                continue
            frames_after = frame_idx - pf
            if frames_after >= self.move_window:
                is_collision = self.verify_real_collision(self.pos_white, pos2, pf)
                if is_collision:
                    if name2 == "red":
                        if pf - self.last_collision_red >= self.min_frame_gap:
                            self.last_collision_red = pf
                            self.collisions_red.append(pf)
                            confirmed_set.add(pf)
                            if confirmed_frame is None:
                                confirmed_frame = pf
                    else:
                        if pf - self.last_collision_yellow >= self.min_frame_gap:
                            self.last_collision_yellow = pf
                            self.collisions_yellow.append(pf)
                            confirmed_set.add(pf)
                            if confirmed_frame is None:
                                confirmed_frame = pf
            elif frames_after < self.move_window + 10:
                new_pending.append(pf)
        if name2 == "red":
            self.pending_red = new_pending
        else:
            self.pending_yellow = new_pending
        return confirmed_frame

    def get_velocity(self, ball_pos, frames):
        if len(frames) < 2:
            return (0.0, 0.0), 0.0
        frames_sorted = sorted(frames)
        velocities_x = []
        velocities_y = []
        for i in range(1, len(frames_sorted)):
            f1, f2 = frames_sorted[i-1], frames_sorted[i]
            if f1 not in ball_pos or f2 not in ball_pos:
                continue
            x1, y1, _ = ball_pos[f1]
            x2, y2, _ = ball_pos[f2]
            dt = max(1, f2 - f1)
            velocities_x.append((x2 - x1) / dt)
            velocities_y.append((y2 - y1) / dt)
        if not velocities_x:
            return (0.0, 0.0), 0.0
        vx = np.mean(velocities_x)
        vy = np.mean(velocities_y)
        speed = np.hypot(vx, vy)
        return (vx, vy), speed

    def get_direction_angle(self, vx, vy):
        if abs(vx) < 0.01 and abs(vy) < 0.01:
            return None
        return np.arctan2(vy, vx)

    def verify_real_collision(self, pos_white, pos_other, frame, pre_window=6, post_window=8):
        pre_frames_w = sorted([f for f in pos_white.keys() if frame - pre_window <= f < frame])
        post_frames_w = sorted([f for f in pos_white.keys() if frame < f <= frame + post_window])
        pre_frames_o = sorted([f for f in pos_other.keys() if frame - pre_window <= f < frame])
        post_frames_o = sorted([f for f in pos_other.keys() if frame < f <= frame + post_window])
        if len(pre_frames_o) < 2 or len(post_frames_o) < 2:
            return False
        (pre_vx_o, pre_vy_o), pre_speed_o = self.get_velocity(pos_other, pre_frames_o)
        (post_vx_o, post_vy_o), post_speed_o = self.get_velocity(pos_other, post_frames_o)
        other_was_stationary = pre_speed_o < 1.0
        other_started_moving = post_speed_o >= 1.5
        if other_was_stationary and other_started_moving:
            return True
        if len(pre_frames_w) < 2 or len(post_frames_w) < 2:
            return other_was_stationary and other_started_moving
        (pre_vx_w, pre_vy_w), pre_speed_w = self.get_velocity(pos_white, pre_frames_w)
        (post_vx_w, post_vy_w), post_speed_w = self.get_velocity(pos_white, post_frames_w)
        white_was_moving = pre_speed_w >= 1.0
        other_was_moving = pre_speed_o >= 1.0
        if white_was_moving and other_was_moving:
            pre_angle_w = self.get_direction_angle(pre_vx_w, pre_vy_w)
            post_angle_w = self.get_direction_angle(post_vx_w, post_vy_w)
            pre_angle_o = self.get_direction_angle(pre_vx_o, pre_vy_o)
            post_angle_o = self.get_direction_angle(post_vx_o, post_vy_o)
            white_changed_dir = False
            other_changed_dir = False
            if pre_angle_w is not None and post_angle_w is not None:
                angle_diff_w = abs(pre_angle_w - post_angle_w)
                if angle_diff_w > np.pi:
                    angle_diff_w = 2 * np.pi - angle_diff_w
                if angle_diff_w > 0.3:
                    white_changed_dir = True
            if pre_angle_o is not None and post_angle_o is not None:
                angle_diff_o = abs(pre_angle_o - post_angle_o)
                if angle_diff_o > np.pi:
                    angle_diff_o = 2 * np.pi - angle_diff_o
                if angle_diff_o > 0.3:
                    other_changed_dir = True
            speed_change_w = abs(post_speed_w - pre_speed_w)
            speed_change_o = abs(post_speed_o - pre_speed_o)
            white_changed_speed = speed_change_w > 1.5
            other_changed_speed = speed_change_o > 1.5
            if white_changed_dir or other_changed_dir:
                return True
            if white_changed_speed and other_changed_speed:
                return True
            return False
        if white_was_moving and not other_was_moving:
            if other_started_moving:
                return True
            if post_speed_o > pre_speed_o + 0.5:
                return True
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="3 top bilardo videosunda real-time ralli analizi.")
    parser.add_argument("video")
    parser.add_argument("--template", default="template.jpg")
    parser.add_argument("--match-threshold", type=float, default=0.8)
    return parser.parse_args()

def draw_frame_overlay(vis, frame_idx, roi, white_pos, event_history, rally_summary=None):
    h, w = vis.shape[:2]
    bands = roi["bands"]
    ix, iy, iw, ih = roi["inner"]
    cv2.rectangle(vis, (ix, iy), (ix + iw, iy + ih), (0, 255, 0), 1)
    cv2.line(vis, (bands["left"], bands["top"]), (bands["left"], bands["bottom"]), (255, 0, 0), 2)
    cv2.line(vis, (bands["right"], bands["top"]), (bands["right"], bands["bottom"]), (0, 0, 255), 2)
    cv2.line(vis, (bands["left"], bands["top"]), (bands["right"], bands["top"]), (0, 255, 0), 2)
    cv2.line(vis, (bands["left"], bands["bottom"]), (bands["right"], bands["bottom"]), (0, 255, 255), 2)
    if white_pos is not None:
        cx, cy, r = white_pos
        pad = int(max(12, 2 * r))
        x1 = max(0, int(cx - pad))
        y1 = max(0, int(cy - pad))
        x2 = min(w - 1, int(cx + pad))
        y2 = min(h - 1, int(cy + pad))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(vis, f"frame {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, f"frame {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    should_pause = False
    if event_history:
        box_height = min(35 + len(event_history) * 22, h - 100)
        overlay = vis.copy()
        cv2.rectangle(overlay, (w - 320, 50), (w - 10, 50 + box_height), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
        cv2.putText(vis, "TESPIT GECMISI:", (w - 310, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        y0 = 92
        max_display = min(len(event_history), 12)
        start_idx = max(0, len(event_history) - max_display)
        for i in range(start_idx, len(event_history)):
            evt = event_history[i]
            color = (255, 255, 255)
            if "RED" in evt:
                color = (0, 0, 255)
            elif "YELLOW" in evt:
                color = (0, 255, 255)
            elif "BAND" in evt:
                color = (0, 255, 0)
            cv2.putText(vis, evt, (w - 310, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y0 += 20
    if rally_summary is not None:
        overlay = vis.copy()
        cv2.rectangle(overlay, (40, 60), (w - 40, 220), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
        y0 = 90
        for line in rally_summary:
            cv2.putText(vis, line, (60, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y0 += 25
        should_pause = True
    return vis, should_pause

def main():
    args = parse_args()
    if not os.path.exists(args.template):
        raise FileNotFoundError(f"Separator template bulunamadi: {args.template}")
    template_bgr = cv2.imread(args.template)
    if template_bgr is None:
        raise RuntimeError(f"Separator template okunamadi: {args.template}")
    template_small = cv2.resize(template_bgr, (0, 0), fx=0.5, fy=0.5)
    template_gray = cv2.cvtColor(template_small, cv2.COLOR_BGR2GRAY)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        all_frames.append(f)
    cap.release()
    total_frames = len(all_frames)
    roi = None
    frame_idx = 0
    rally_num = 0
    rally_start = SEPARATOR_FRAMES
    in_separator = True
    separator_end_frame = SEPARATOR_FRAMES
    band_detector = None
    collision_detector = None
    prev_white = None
    rally_results = []
    window_name = "Bilardo Analiz"
    separator_cache = {}
    CHECK_INTERVAL = 5
    event_history = []
    while frame_idx < total_frames:
        frame = all_frames[frame_idx]
        if roi is None and frame_idx >= SEPARATOR_FRAMES:
            roi, _ = detect_table_roi(frame)
            band_detector = BandHitDetector(roi["bands"])
            collision_detector = CollisionDetector()
        if frame_idx < separator_end_frame:
            cv2.putText(frame, f"frame {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"frame {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            frame_idx += 1
            continue
        if in_separator:
            in_separator = False
            rally_num += 1
            rally_start = frame_idx
            band_detector.reset()
            collision_detector.reset()
            prev_white = None
            event_history = []
        lookahead_frame = frame_idx + ANIM_TAIL_FRAMES
        is_rally_end = False
        if lookahead_frame < total_frames:
            check_frame = (lookahead_frame // CHECK_INTERVAL) * CHECK_INTERVAL
            if check_frame not in separator_cache:
                if check_frame < total_frames:
                    lookahead = all_frames[check_frame]
                    lookahead_small = cv2.resize(lookahead, (0, 0), fx=0.5, fy=0.5)
                    separator_cache[check_frame] = is_separator_frame_template(lookahead_small, template_gray, args.match_threshold)
            if separator_cache.get(check_frame, False):
                is_rally_end = True
        is_last_rally_end = False
        if not in_separator and frame_idx == total_frames - ANIM_TAIL_FRAMES:
            is_last_rally_end = True
        if frame_idx > total_frames - ANIM_TAIL_FRAMES:
            cv2.putText(frame, f"frame {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"frame {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            frame_idx += 1
            continue
        balls = detect_balls_in_frame(frame, roi)
        collision_detector.update_positions(frame_idx, balls)
        white_pos = None
        if balls["white"] is not None:
            cx, cy = balls["white"]["center"]
            r = balls["white"]["radius"]
            white_pos = (cx, cy, r)
            prev_white = white_pos
        elif prev_white is not None:
            white_pos = prev_white
        if white_pos is not None:
            cx, cy, r = white_pos
            new_band_hits = band_detector.process_frame(frame_idx, cx, cy)
            for side, f in new_band_hits:
                event_history.append(f"F{f}: BAND_{side.upper()}")
        coll_red = collision_detector.check_collision(frame_idx, "red")
        if coll_red is not None:
            event_history.append(f"F{coll_red}: WHITE-RED")
        coll_yellow = collision_detector.check_collision(frame_idx, "yellow")
        if coll_yellow is not None:
            event_history.append(f"F{coll_yellow}: WHITE-YELLOW")
        rally_summary = None
        if is_rally_end or is_last_rally_end:
            rally_end = frame_idx
            band_counts = band_detector.band_counts
            total_bands = sum(band_counts.values())
            hit_red = len(collision_detector.collisions_red) > 0
            hit_yellow = len(collision_detector.collisions_yellow) > 0
            success = (total_bands >= 3) and hit_red and hit_yellow
            rally_results.append({
                "rally_num": rally_num,
                "start": rally_start,
                "end": rally_end,
                "band_counts": band_counts.copy(),
                "hit_red": hit_red,
                "hit_yellow": hit_yellow,
                "success": success,
            })
            rally_summary = [
                f"Ralli {rally_num}",
                f"Bands: L={band_counts['left']}, R={band_counts['right']}, T={band_counts['top']}, B={band_counts['bottom']} (sum={total_bands})",
                f"Hit RED: {'YES' if hit_red else 'NO'}",
                f"Hit YELLOW: {'YES' if hit_yellow else 'NO'}",
                f"RESULT: {'SUCCESS' if success else 'FAIL'}",
                "Devam icin bir tusa basin (ESC cikis)",
            ]
            print(f"Ralli {rally_num}: frame {rally_start}-{rally_end}, Bands={total_bands}, Red={hit_red}, Yellow={hit_yellow}, {'BASARILI' if success else 'BASARISIZ'}")
            if is_rally_end:
                separator_start = lookahead_frame
                separator_end_frame = separator_start + SEPARATOR_FRAMES
                in_separator = True
        vis = frame.copy()
        vis, should_pause = draw_frame_overlay(vis, frame_idx, roi, white_pos, event_history, rally_summary)
        cv2.imshow(window_name, vis)
        if should_pause:
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        frame_idx += 1
    cv2.destroyAllWindows()
    print("\n=== GENEL SONUCLAR ===")
    print(f"Toplam ralli sayisi: {len(rally_results)}")
    success_count = sum(1 for r in rally_results if r["success"])
    print(f"Basarili ralli sayisi: {success_count}")

if __name__ == "__main__":
    main()
