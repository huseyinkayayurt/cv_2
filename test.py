import cv2
import argparse
import os
from collections import deque

START_FRAME = 61
LOOKAHEAD_FRAMES = 30  # i -> i+30'a bak
SEP_LEN_FRAMES = 60  # separator bölgesi uzunluğu (işlenmeyecek)
ANIM_TAIL_FRAMES = 30  # son animasyon kuyrukları analiz dışı


def detect_separator(frame_bgr, template_gray, threshold=0.80, roi=None, downscale=1.0):
    if roi is not None:
        x, y, w, h = roi
        frame_bgr = frame_bgr[y:y + h, x:x + w]

    if downscale != 1.0:
        frame_bgr = cv2.resize(frame_bgr, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)

    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    if (template_gray.shape[0] > frame_gray.shape[0]) or (template_gray.shape[1] > frame_gray.shape[1]):
        return False, 0.0

    res = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return (max_val >= threshold), float(max_val)


def segment_rallies(video_path, template_path, threshold=0.80, start_frame=START_FRAME,
                    lookahead=LOOKAHEAD_FRAMES, sep_len=SEP_LEN_FRAMES,
                    anim_tail=ANIM_TAIL_FRAMES, roi=None, downscale=1.0):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template bulunamadı: {template_path}")

    tpl_bgr = cv2.imread(template_path)
    if tpl_bgr is None:
        raise RuntimeError("Template okunamadı (cv2.imread None döndü).")

    if downscale != 1.0:
        tpl_bgr = cv2.resize(tpl_bgr, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)

    template_gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Video açılamadı.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video frame sayısı okunamadı.")

    if start_frame >= total_frames:
        print(f"[WARN] start_frame ({start_frame}) video uzunluğunu aşıyor ({total_frames}).")
        cap.release()
        return

    window = deque(maxlen=lookahead + 1)

    def read_frame_at_current_pos():
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, fr = cap.read()
        if not ret:
            return None, None
        return idx, fr

    rally_start = start_frame
    rally_id = 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def finalize_last_rally_if_needed():
        nonlocal rally_id, rally_start
        last_possible_end = (total_frames - 1) - anim_tail  # senin dediğin: 3697-30=3667
        if last_possible_end >= rally_start:
            print(f"RALLY {rally_id}: {rally_start}-{last_possible_end}  | (no separator, EOF fallback)")

    while True:
        # pencereyi doldur (lookahead+1 frame)
        while len(window) < (lookahead + 1):
            idx, fr = read_frame_at_current_pos()
            if fr is None:
                # EOF: separator bulunamadıysa son ralliyi yazdır
                finalize_last_rally_if_needed()
                cap.release()
                cv2.destroyAllWindows()
                return
            window.append((idx, fr))

        # pencerenin sonu: (i+lookahead) -> separator aday frame
        sep_idx, sep_frame = window[-1]
        found, score = detect_separator(sep_frame, template_gray, threshold=threshold, roi=roi, downscale=1.0)

        if found:
            rally_end = window[0][0]  # i
            print(f"RALLY {rally_id}: {rally_start}-{rally_end}  | sep@{sep_idx}  score={score:.3f}")

            # yeni ralli başlangıcı = sep_idx + 60  == rally_end + 30 + 60
            next_start = sep_idx + sep_len

            rally_id += 1
            rally_start = next_start

            window.clear()
            if next_start >= total_frames:
                cap.release()
                return

            cap.set(cv2.CAP_PROP_POS_FRAMES, next_start)
            continue

        # separator yoksa pencereyi kaydır
        window.popleft()


def parse_roi(text):
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError("ROI formatı x,y,w,h olmalı")
    return tuple(int(p.strip()) for p in parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bilardo rallilerini separator template ile ayırma (EOF fallback)")
    ap.add_argument("video", help="Video yolu (örn: input.mp4)")
    ap.add_argument("template", help="Separator template görsel yolu (örn: template.jpg)")
    ap.add_argument("--threshold", type=float, default=0.80, help="Template match eşiği (0-1)")
    ap.add_argument("--start", type=int, default=START_FRAME, help="Analize başlanacak frame index")
    ap.add_argument("--lookahead", type=int, default=LOOKAHEAD_FRAMES, help="Kaç frame ileriye bakılacak")
    ap.add_argument("--sep-len", type=int, default=SEP_LEN_FRAMES, help="Separator bölgesi uzunluğu (frame)")
    ap.add_argument("--anim-tail", type=int, default=ANIM_TAIL_FRAMES, help="Ralli sonu animasyon kuyrukları (frame)")
    ap.add_argument("--roi", type=str, default=None, help="Opsiyonel ROI: x,y,w,h")
    ap.add_argument("--downscale", type=float, default=1.0, help="Hız için küçültme (örn 0.5)")
    args = ap.parse_args()

    roi = parse_roi(args.roi) if args.roi else None

    segment_rallies(
        video_path=args.video,
        template_path=args.template,
        threshold=args.threshold,
        start_frame=args.start,
        lookahead=args.lookahead,
        sep_len=args.sep_len,
        anim_tail=args.anim_tail,
        roi=roi,
        downscale=args.downscale
    )
