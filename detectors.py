import numpy as np
from config import (
    BAND_BALL_RADIUS, BAND_ENTER_THRESHOLD, BAND_EXIT_THRESHOLD,
    BAND_APPROACH_DELTA, BAND_MIN_FRAME_GAP,
    COLLISION_BALL_RADIUS, COLLISION_DIST_FACTOR,
    COLLISION_MIN_FRAME_GAP, COLLISION_MOVE_WINDOW, COLLISION_MOVE_THRESHOLD
)


class BandHitDetector:
    
    def __init__(self, bands):
        self.bands = bands
        self.band_counts = {"left": 0, "right": 0, "top": 0, "bottom": 0}
        self.hit_sequence = []
        self.distances_per_side = {side: [] for side in ["left", "right", "top", "bottom"]}
        self.state = {side: "far" for side in ["left", "right", "top", "bottom"]}
        self.last_far_dist = {side: None for side in ["left", "right", "top", "bottom"]}
        self.last_hit_frame = {side: -99999 for side in ["left", "right", "top", "bottom"]}
        
        self.BALL_R = BAND_BALL_RADIUS
        self.ENTER_TH = BAND_ENTER_THRESHOLD
        self.EXIT_TH = BAND_EXIT_THRESHOLD
        self.APPROACH_DELTA = BAND_APPROACH_DELTA
        self.MIN_FRAME_GAP = BAND_MIN_FRAME_GAP

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
                    if (self.last_far_dist[side] - d) >= self.APPROACH_DELTA and \
                       (frame_idx - self.last_hit_frame[side]) >= self.MIN_FRAME_GAP:
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
    
    def __init__(self, ball_radius=COLLISION_BALL_RADIUS, dist_factor=COLLISION_DIST_FACTOR,
                 min_frame_gap=COLLISION_MIN_FRAME_GAP, move_window=COLLISION_MOVE_WINDOW,
                 move_thresh=COLLISION_MOVE_THRESHOLD):
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

    def check_collision_by_target_movement(self, frame_idx, name2):
        if name2 == "red":
            pos_target = self.pos_red
            last_collision = self.last_collision_red
            confirmed_set = self.confirmed_red
        else:
            pos_target = self.pos_yellow
            last_collision = self.last_collision_yellow
            confirmed_set = self.confirmed_yellow
        
        if len(pos_target) < 15:
            return None
        
        recent_window = 10
        recent_frames = sorted([f for f in pos_target.keys() if frame_idx - recent_window <= f <= frame_idx])
        
        if len(recent_frames) < 5:
            return None
        
        pre_start = frame_idx - recent_window - 10
        pre_end = frame_idx - recent_window
        pre_frames = sorted([f for f in pos_target.keys() if pre_start <= f < pre_end])
        
        if len(pre_frames) < 5:
            return None
        
        _, pre_speed = self.get_velocity(pos_target, pre_frames)
        _, current_speed = self.get_velocity(pos_target, recent_frames)
        
        was_stationary = pre_speed < 0.8
        now_moving = current_speed >= 2.0
        
        if was_stationary and now_moving:
            movement_start_frame = None
            all_frames_sorted = sorted(pos_target.keys())
            
            for i in range(1, len(all_frames_sorted)):
                f1, f2 = all_frames_sorted[i-1], all_frames_sorted[i]
                if f1 not in pos_target or f2 not in pos_target:
                    continue
                x1, y1, _ = pos_target[f1]
                x2, y2, _ = pos_target[f2]
                dist = np.hypot(x2 - x1, y2 - y1)
                if dist > 2.0 and f2 > pre_end:
                    movement_start_frame = f1
                    break
            
            if movement_start_frame is None:
                movement_start_frame = recent_frames[0]
            
            if movement_start_frame in confirmed_set:
                return None
            if movement_start_frame - last_collision < self.min_frame_gap:
                return None
            
            confirmed_set.add(movement_start_frame)
            if name2 == "red":
                self.last_collision_red = movement_start_frame
                self.collisions_red.append(movement_start_frame)
            else:
                self.last_collision_yellow = movement_start_frame
                self.collisions_yellow.append(movement_start_frame)
            
            return movement_start_frame
        
        return None

    def get_distance_between_balls(self, pos_white, pos_other, frame):
        if frame not in pos_white or frame not in pos_other:
            return None
        x1, y1, _ = pos_white[frame]
        x2, y2, _ = pos_other[frame]
        return np.hypot(x1 - x2, y1 - y2)

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
            pre_distances = []
            for f in pre_frames_w[-4:]:
                d = self.get_distance_between_balls(pos_white, pos_other, f)
                if d is not None:
                    pre_distances.append(d)
            
            post_distances = []
            for f in post_frames_w[:4]:
                d = self.get_distance_between_balls(pos_white, pos_other, f)
                if d is not None:
                    post_distances.append(d)
            
            approaching = False
            if len(pre_distances) >= 2:
                dist_change_pre = pre_distances[-1] - pre_distances[0]
                if dist_change_pre < -3:
                    approaching = True
            
            separating = False
            if len(post_distances) >= 2:
                dist_change_post = post_distances[-1] - post_distances[0]
                if dist_change_post > 3:
                    separating = True
            
            if not approaching and not separating:
                return False

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
                if angle_diff_w > 0.4:
                    white_changed_dir = True
            
            if pre_angle_o is not None and post_angle_o is not None:
                angle_diff_o = abs(pre_angle_o - post_angle_o)
                if angle_diff_o > np.pi:
                    angle_diff_o = 2 * np.pi - angle_diff_o
                if angle_diff_o > 0.4:
                    other_changed_dir = True

            speed_change_w = abs(post_speed_w - pre_speed_w)
            speed_change_o = abs(post_speed_o - pre_speed_o)
            white_changed_speed = speed_change_w > 2.0
            other_changed_speed = speed_change_o > 2.0
            
            if approaching and separating:
                if white_changed_dir or other_changed_dir:
                    return True
                if white_changed_speed and other_changed_speed:
                    return True
            
            if approaching or separating:
                if (white_changed_dir or white_changed_speed) and (other_changed_dir or other_changed_speed):
                    return True
            
            return False

        if white_was_moving and not other_was_moving:
            if other_started_moving:
                return True
            if post_speed_o > pre_speed_o + 0.5:
                return True
        
        return False

