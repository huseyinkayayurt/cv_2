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


def detect_table_roi(frame, margin: int = 8):
    """
    Bilardo masasının oyun alanını (turkuaz kumaş kısmı) otomatik tespit eder.
    Dönen:
        roi = {
            "outer": (x, y, w, h),   # kumaşın boundingRect'i
            "inner": (ix, iy, iw, ih),  # margin kadar içeri kaydırılmış dikdörtgen
            "bands": {
                "left":   ix,
                "right":  ix + iw - 1,
                "top":    iy,
                "bottom": iy + ih - 1,
            }
        }
        mask : masa maskesi (debug için)
    """

    # BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Masa rengi: turkuaz/mavi. Bu aralık video için test edilip seçildi.
    # Gerekirse bu sınırlarla oynayabiliriz.
    lower_table = np.array([85, 80, 80], dtype=np.uint8)
    upper_table = np.array([110, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_table, upper_table)

    # Gürültüyü azaltmak için morfoloji
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # En büyük konturu masa kabul ediyoruz
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Masa konturu bulunamadı, HSV aralığını güncellemek gerekebilir.")

    table_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(table_contour)

    # Ahşap kenarlara değmemek için margin kadar içeri giriyoruz
    ix = x + margin
    iy = y + margin
    iw = max(1, w - 2 * margin)
    ih = max(1, h - 2 * margin)

    roi = {
        "outer": (x, y, w, h),
        "inner": (ix, iy, iw, ih),
        "bands": {
            "left": ix,
            "right": ix + iw - 1,
            "top": iy,
            "bottom": iy + ih - 1,
        },
    }

    return roi, mask

def get_frame_at(video_path: str, frame_idx: int):
    """Belirli indexteki frame'i okur."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"{frame_idx} indexli frame okunamadı.")

    return frame


def visualize_table_roi(video_path: str, sample_idx: int, roi):
    """ROI'yi çizip ekranda gösterir (sadece debug amaçlı)."""
    frame = get_frame_at(video_path, sample_idx)

    vis = frame.copy()
    ox, oy, ow, oh = roi["outer"]
    ix, iy, iw, ih = roi["inner"]

    # Dış boundingRect (kumaş alanı) - kırmızı
    cv2.rectangle(vis, (ox, oy), (ox + ow, oy + oh), (0, 0, 255), 2)
    # İç oyun alanı - yeşil
    cv2.rectangle(vis, (ix, iy), (ix + iw, iy + ih), (0, 255, 0), 2)

    cv2.imshow("Table ROI (outer=red, inner=green)", vis)
    print("ROI görüntülendi. Pencereyi kapatmak için herhangi bir tuşa bas.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
        min_rally_length=args.min_rally_length,
        debug_every_n=args.debug_n
    )
    print(f"Bulunan ralli sayısı: {len(rallies)}")

    for i, (start, end) in enumerate(rallies, start=1):
        print(f"Ralli {i}: start={start}, end={end}, uzunluk={end - start + 1} frame")
    if not rallies:
        print("Ralli bulunamadı, ROI tespiti yapmıyorum.")
        return

    # 2) Masa ROI'sini tespit etmek için bir örnek frame seç
    #    1. rallinin ortasındaki frame'i alıyorum:
    first_start, first_end = rallies[0]
    sample_idx = (first_start + first_end) // 2

    frame = get_frame_at(args.video, sample_idx)
    roi, _ = detect_table_roi(frame, margin=8)

    ox, oy, ow, oh = roi["outer"]
    ix, iy, iw, ih = roi["inner"]
    bands = roi["bands"]

    print("\n--- Masa ROI Bilgileri ---")
    print(f"Dış (kumaş) dikdörtgen: x={ox}, y={oy}, w={ow}, h={oh}")
    print(f"İç (oyun alanı) dikdörtgen: x={ix}, y={iy}, w={iw}, h={ih}")
    print("Band sınırları:")
    print(f"  left   = {bands['left']}")
    print(f"  right  = {bands['right']}")
    print(f"  top    = {bands['top']}")
    print(f"  bottom = {bands['bottom']}")

    # 3) Debug: ROI'yi çizip göster
    visualize_table_roi(args.video, sample_idx, roi)

if __name__ == "__main__":
    main()
