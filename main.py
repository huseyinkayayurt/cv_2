import cv2
import argparse
import os

from config import ANIM_TAIL_FRAMES, SEPARATOR_FRAMES, DEFAULT_MATCH_THRESHOLD
from detection import is_separator_frame_template, detect_table_roi, detect_balls_in_frame
from detectors import BandHitDetector, CollisionDetector
from visualization import draw_frame_overlay


def parse_args():
    parser = argparse.ArgumentParser(description="3 top bilardo videosunda real-time ralli analizi.")
    parser.add_argument("video", help="Video dosyasının yolu")
    parser.add_argument("--template", default="template.jpg", help="Separator template dosyası")
    parser.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD,
                        help="Template eşleştirme eşiği")
    return parser.parse_args()


def load_template(template_path):
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Separator template bulunamadi: {template_path}")
    
    template_bgr = cv2.imread(template_path)
    if template_bgr is None:
        raise RuntimeError(f"Separator template okunamadi: {template_path}")
    
    template_small = cv2.resize(template_bgr, (0, 0), fx=0.5, fy=0.5)
    template_gray = cv2.cvtColor(template_small, cv2.COLOR_BGR2GRAY)
    return template_gray


def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {video_path}")
    
    all_frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        all_frames.append(f)
    cap.release()
    
    return all_frames


def process_rally_end(rally_num, rally_start, rally_end, band_detector, collision_detector):
    band_counts = band_detector.band_counts
    total_bands = sum(band_counts.values())
    hit_red = len(collision_detector.collisions_red) > 0
    hit_yellow = len(collision_detector.collisions_yellow) > 0
    success = (total_bands >= 3) and hit_red and hit_yellow
    
    result = {
        "rally_num": rally_num,
        "start": rally_start,
        "end": rally_end,
        "band_counts": band_counts.copy(),
        "hit_red": hit_red,
        "hit_yellow": hit_yellow,
        "success": success,
    }
    
    summary = [
        f"Ralli {rally_num}",
        f"Bands: L={band_counts['left']}, R={band_counts['right']}, T={band_counts['top']}, B={band_counts['bottom']} (sum={total_bands})",
        f"Hit RED: {'YES' if hit_red else 'NO'}",
        f"Hit YELLOW: {'YES' if hit_yellow else 'NO'}",
        f"RESULT: {'SUCCESS' if success else 'FAIL'}",
        "Devam icin bir tusa basin (ESC cikis)",
    ]
    
    print(f"Ralli {rally_num}: frame {rally_start}-{rally_end}, Bands={total_bands}, "
          f"Red={hit_red}, Yellow={hit_yellow}, {'BASARILI' if success else 'BASARISIZ'}")
    
    return result, summary


def main():
    args = parse_args()

    template_gray = load_template(args.template)
    all_frames = load_video_frames(args.video)
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
                    separator_cache[check_frame] = is_separator_frame_template(
                        lookahead_small, template_gray, args.match_threshold)
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

        if len(collision_detector.collisions_red) == 0:
            coll_red_alt = collision_detector.check_collision_by_target_movement(frame_idx, "red")
            if coll_red_alt is not None:
                event_history.append(f"F{coll_red_alt}: WHITE-RED (hareket)")
        
        if len(collision_detector.collisions_yellow) == 0:
            coll_yellow_alt = collision_detector.check_collision_by_target_movement(frame_idx, "yellow")
            if coll_yellow_alt is not None:
                event_history.append(f"F{coll_yellow_alt}: WHITE-YELLOW (hareket)")

        rally_summary = None
        if is_rally_end or is_last_rally_end:
            result, rally_summary = process_rally_end(
                rally_num, rally_start, frame_idx, band_detector, collision_detector)
            rally_results.append(result)
            
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
    
    # Sonuçları yazdır
    print("\n=== GENEL SONUCLAR ===")
    print(f"Toplam ralli sayisi: {len(rally_results)}")
    success_count = sum(1 for r in rally_results if r["success"])
    print(f"Basarili ralli sayisi: {success_count}")


if __name__ == "__main__":
    main()
