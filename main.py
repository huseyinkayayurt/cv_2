import cv2
import numpy as np
import argparse
import os


def is_separator_frame_template(frame, template_gray, match_threshold=0.8, debug=False) -> bool:
    """
    Ayrı verilen separator görseline (template) göre frame'in separator olup
    olmadığını tespit eder.
    - Template ve frame gri formda eşleştiriliyor.
    - Template frame'den büyükse, frame'e sığacak şekilde ölçekleniyor.
    """

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fh, fw = frame_gray.shape
    th, tw = template_gray.shape

    # Template frame'den büyükse biraz küçült
    tpl = template_gray
    if th > fh or tw > fw:
        scale = min(fh / th, fw / tw) * 0.9
        new_size = (max(1, int(tw * scale)), max(1, int(th * scale)))
        tpl = cv2.resize(template_gray, new_size)
        th, tw = tpl.shape

    # Template matching
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
    # Separator görselini yükle
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
    core_sep_flags = []  # her frame için: core separator mı? (template direkt eşleşen frame)
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
        # Yine de tüm videoyu tek ralli olarak döndürelim
        return [(0, total_frames - 1)], [], total_frames

    # 1D "genişletme": Template görünen framelerin etrafındaki ±expand_frames
    # aralığı da separator kabul ediliyor. Böylece animasyon + beyaz geçişler
    # aynı separator bloğuna dahil olmuş oluyor.
    is_sep = [False] * total_frames
    for i, flag in enumerate(core_sep_flags):
        if flag:
            start = max(0, i - expand_frames)
            end = min(total_frames - 1, i)
            for j in range(start, end + 1):
                is_sep[j] = True

    # Debug için: final separator framelerini listlemek istersek
    final_sep_indices = [i for i, f in enumerate(is_sep) if f]

    # Şimdi separator blokları arasındaki aralıkları ralli olarak çıkaralım
    rallies = []
    in_rally = False
    rally_start = 0

    for i in range(total_frames):
        if not is_sep[i]:  # oyun frameleri
            if not in_rally:
                in_rally = True
                rally_start = i
        else:  # separator frameleri
            if in_rally:
                in_rally = False
                rally_end = i - 1
                length = rally_end - rally_start + 1
                if length >= min_rally_length:
                    rallies.append((rally_start, rally_end))
        # separator ise ralli otomatik kapanmış olur

    # video separator ile bitmediyse son ralliyi kapat
    if in_rally:
        rally_end = total_frames - 1
        length = rally_end - rally_start + 1
        if length >= min_rally_length:
            rallies.append((rally_start, rally_end))

    return rallies, final_sep_indices, total_frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="3 top bilardo videosunda ralli sınırlarını separator template'ine göre bulur."
    )
    parser.add_argument("video")
    parser.add_argument("--template",default="frame_000000.jpg")
    parser.add_argument("--match-threshold",type=float,default=0.8)
    parser.add_argument("--expand-frames",type=int,default=29)
    parser.add_argument("--min-rally-length",type=int,default=90)
    parser.add_argument("--debug-n",type=int,default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    rallies, sep_indices, total_frames = find_rallies(
        args.video,
        args.template,
        match_threshold=args.match_threshold,
        expand_frames=args.expand_frames,
        debug_every_n=args.debug_n
    )
    print(f"Bulunan ralli sayısı: {len(rallies)}")

    for i, (start, end) in enumerate(rallies, start=1):
        print(f"Ralli {i}: start={start}, end={end}, uzunluk={end - start + 1} frame")


if __name__ == "__main__":
    main()
