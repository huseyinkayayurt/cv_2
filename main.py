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


def detect_table_roi(frame, margin: int = 8, band_inset: int = 12):
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

    # Band çizgilerini içeri doğru biraz daha çekiyoruz
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
    parser.add_argument("--template", default="frame_000000.jpg")
    parser.add_argument("--match-threshold", type=float, default=0.8)
    parser.add_argument("--expand-frames", type=int, default=29)
    parser.add_argument("--min-rally-length", type=int, default=90)
    parser.add_argument("--debug-n", type=int, default=0)
    return parser.parse_args()


def detect_balls_in_frame(frame, roi):
    """
    Masa ROI'si içinde kırmızı, sarı ve beyaz topları HSV tabanlı olarak tespit eder.
    Dönüş:
        balls = {
            "red":    {"center": (x, y), "radius": r}  veya None,
            "yellow": {"center": (x, y), "radius": r}  veya None,
            "white":  {"center": (x, y), "radius": r}  veya None,
        }
    Koordinatlar full-frame piksel koordinatlarıdır.
    """

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
            return None, None, None  # cx, cy, radius
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area <= 0:
            return None, None, None
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        return x, y, radius

    # --- KIRMIZI TOP (H ~ 160–180, S ve V yüksek) ---
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

    # --- SARI TOP (H ~ 18–24, S ve V yüksek) ---
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
    # --- BEYAZ TOP ---
    # Mantık: düşük S, çok yüksek V, küçük-orta boy parlak dairesel bölge.
    # H aralığını serbest bırakıyoruz.
    lower_white = np.array([0, 0, 210], dtype=np.uint8)
    upper_white = np.array([180, 80, 255], dtype=np.uint8)

    mask_w = cv2.inRange(hsv, lower_white, upper_white)
    mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand_center = None
    cand_radius = None
    max_area = 0.0

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)

        # Top boyutuna göre kaba filtre (bu videoda ~8-9 px civarı)
        if 5 < radius < 20 and area > max_area:
            max_area = area
            cand_center = (x, y)
            cand_radius = radius

    if cand_center is not None:
        wx, wy = cand_center
        balls["white"] = {
            "center": (int(ix + wx), int(iy + wy)),
            "radius": float(cand_radius),
        }

    return balls


def visualize_balls_on_frame(frame, roi, balls):
    vis = frame.copy()

    # Masa iç ROI'yi hafifçe çizelim (debug için)
    ix, iy, iw, ih = roi["inner"]
    cv2.rectangle(vis, (ix, iy), (ix + iw, iy + ih), (0, 255, 0), 1)

    colors_bgr = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "white": (255, 255, 255),
    }

    for name, info in balls.items():
        if info is None:
            continue
        cx, cy = info["center"]
        r = int(info["radius"])
        cv2.circle(vis, (cx, cy), r, colors_bgr[name], 2)
        cv2.circle(vis, (cx, cy), 2, (0, 0, 0), -1)
        cv2.putText(
            vis,
            name,
            (cx + 5, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors_bgr[name],
            1,
            cv2.LINE_AA,
        )

    cv2.imshow("Ball detection debug", vis)
    print("Top tespiti görüntülendi. Kapatmak için bir tuşa bas.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_rally_white_and_bands(video_path, roi, rally_start, rally_end,
                                  initial_white=None, debug=False):
    """
    Verilen ralli aralığında beyaz topu takip eder, her frame için merkezini hesaplar,
    sonra band mesafelerine threshold + "uzaktan gelme" mantığıyla bakarak
    kaç banda kaç kere ve hangi sırayla çarptığını bulur.

    Dönen:
        band_counts:  { 'left': n, 'right': n, 'top': n, 'bottom': n }
        positions:    { frame_idx: (cx, cy, r) }
        hit_sequence: [ ('bottom', frame_idx), ('left', frame_idx), ... ]
    """

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

    # 1) Önce tüm ralliyi bir kere geçip beyaz top merkezlerini çıkaralım
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
            # henüz hiç beyaz top görmediysek bu frame'i es geçiyoruz
            continue

        positions[idx] = (cx, cy, r)

    cap.release()

    if not positions:
        # Hiç beyaz top tespit edilemediyse
        return {s: 0 for s in ["left", "right", "top", "bottom"]}, {}, []

    # 2) Her band için distance zaman serisini çıkaralım
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

    # 3) Threshold + "uzaktan gelme" + hysteresis ile hit tespit

    band_counts = {side: 0 for side in distances_per_side}
    hit_sequence = []

    BALL_R = 10.0

    # Yaklaşma/uzaklaşma eşikleri
    ENTER_TH = BALL_R + 3.0  # mesafe buranın altına inince "banda çok yakın" say
    EXIT_TH = BALL_R + 7.0  # mesafe buranın üstüne çıkınca tekrar "far" moda dön
    APPROACH_DELTA = 6.0  # uzaktan gelmiş saymak için: prev_far_dist - cur_dist >= bu
    MIN_FRAME_GAP = 10  # aynı banda iki hit arasında minimum frame
    # İstersen bunlarla oynayabilirsin

    for side, series in distances_per_side.items():
        if len(series) < 2:
            continue

        state = "far"  # 'far' veya 'near'
        last_far_dist = None  # 'far' durumundayken gördüğümüz en büyük mesafe
        last_hit_frame = -99999

        for f, d in series:
            # Stabilite için negatifler sıfıra kırpıldı; burada d >= 0 olmalı
            if state == "far":
                # far durumundayken "ne kadar uzaktaydık" bilgisini topla
                if last_far_dist is None or d > last_far_dist:
                    last_far_dist = d

                # Hit adayı: yeterince yakına girdik mi?
                if d <= ENTER_TH and last_far_dist is not None:
                    # Uzaktan gerçekten gelmiş mi?
                    if (last_far_dist - d) >= APPROACH_DELTA and (f - last_hit_frame) >= MIN_FRAME_GAP:
                        band_counts[side] += 1
                        last_hit_frame = f
                        state = "near"
                        if debug:
                            print(f"[DEBUG] HIT {side} at frame {f} (last_far={last_far_dist:.1f}, d={d:.1f})")
                        hit_sequence.append((side, f))
                        # near'a geçtikten sonra far mesafeyi sıfırla
                        last_far_dist = None

            else:  # state == 'near'
                # Banda yakınken uzağa çıkmayı bekliyoruz
                if d >= EXIT_TH:
                    state = "far"
                    last_far_dist = d  # yeni far başlangıç mesafesi

    # frame sırasına göre sıralayalım
    hit_sequence.sort(key=lambda x: x[1])

    return band_counts, positions, hit_sequence


def visualize_rally_hits(video_path, roi, rally_start, rally_end, positions, hit_sequence,
                         window_name="Rally debug", play_delay=20):
    """
    Verilen ralli aralığını oynatır, beyaz topu ve band çizgilerini çizer,
    process_rally_white_and_bands'ten gelen hit_sequence'e göre çarpmaları
    video üzerinde gösterir.

    hit_sequence: [('bottom', frame_idx), ('left', frame_idx), ...]
    positions:    { frame_idx: (cx, cy, r) }

    Her hit olduğu framede görüntüyü durdurur (waitKey(0)).
    Normal framelerde play_delay ms bekler (ör: 20 ms).
    """

    bands = roi["bands"]

    # Hit'leri frame -> [side1, side2, ...] map'ine dönüştürelim
    hits_by_frame = {}
    for side, f in hit_sequence:
        hits_by_frame.setdefault(f, []).append(side)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    # Ralli başlangıcına sar
    cap.set(cv2.CAP_PROP_POS_FRAMES, rally_start)

    # Band çizgilerini çizerken kullanacağımız renkler
    # BGR formatı
    band_colors = {
        "left": (255, 0, 0),  # mavi
        "right": (0, 0, 255),  # kırmızı
        "top": (0, 255, 0),  # yeşil
        "bottom": (0, 255, 255),  # sarı
    }

    while True:
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_pos > rally_end:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = current_pos

        vis = frame.copy()

        # --- Masa iç oyun alanını hafifçe çiz (isteğe bağlı) ---
        ix, iy, iw, ih = roi["inner"]
        cv2.rectangle(vis, (ix, iy), (ix + iw, iy + ih), (0, 128, 0), 1)

        # --- Band çizgilerini çiz ---
        # Solda dikey çizgi
        cv2.line(vis, (bands["left"], bands["top"]), (bands["left"], bands["bottom"]), band_colors["left"], 2)
        # Sağda dikey çizgi
        cv2.line(vis, (bands["right"], bands["top"]), (bands["right"], bands["bottom"]), band_colors["right"], 2)
        # Üstte yatay çizgi
        cv2.line(vis, (bands["left"], bands["top"]), (bands["right"], bands["top"]), band_colors["top"], 2)
        # Altta yatay çizgi
        cv2.line(vis, (bands["left"], bands["bottom"]), (bands["right"], bands["bottom"]), band_colors["bottom"], 2)

        # --- Beyaz topu çiz ---
        if frame_idx in positions:
            cx, cy, r = positions[frame_idx]
            cv2.circle(vis, (int(cx), int(cy)), int(r), (255, 255, 255), 2)
            cv2.circle(vis, (int(cx), int(cy)), 2, (0, 0, 0), -1)

        # --- Bu framede hit varsa highlight et ---
        hits_now = hits_by_frame.get(frame_idx, [])
        if hits_now:
            # Hit olan taraflar için büyük, gözükür highlight
            text_lines = []
            for side in hits_now:
                color = band_colors.get(side, (0, 0, 0))
                text_lines.append(f"HIT: {side.upper()}")
                # Band çizgisine yakın bir yere büyük daire çizelim
                if side == "left":
                    hx, hy = bands["left"] + 20, (bands["top"] + bands["bottom"]) // 2
                elif side == "right":
                    hx, hy = bands["right"] - 20, (bands["top"] + bands["bottom"]) // 2
                elif side == "top":
                    hx, hy = (bands["left"] + bands["right"]) // 2, bands["top"] + 20
                else:  # bottom
                    hx, hy = (bands["left"] + bands["right"]) // 2, bands["bottom"] - 20

                cv2.circle(vis, (hx, hy), 25, color, 3)

            # Ekranın üstüne yazı
            y0 = 30
            for line in text_lines:
                cv2.putText(
                    vis,
                    line,
                    (30, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis,
                    line,
                    (30, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                y0 += 30

            # Ayrıca frame indexini de gösterelim
            cv2.putText(
                vis,
                f"frame {frame_idx}",
                (30, y0 + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"frame {frame_idx}",
                (30, y0 + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, vis)
            print(f"[DEBUG VIS] frame {frame_idx}: hits={hits_now}")
            # Çarptığı framede durdur, sen space/ herhangi tuş ile ilerlet
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC ile çık
                break
        else:
            # Normal frame, akış halinde göster
            cv2.imshow(window_name, vis)
            key = cv2.waitKey(play_delay) & 0xFF
            if key == 27:  # ESC ile çık
                break

    cap.release()
    cv2.destroyAllWindows()


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

    # 3) Aynı framede topları tespit et
    balls = detect_balls_in_frame(frame, roi)

    print("\n--- Top bilgileri (örnek frame) ---")
    for name, info in balls.items():
        print(f"{name}: {info}")

    # 4) Debug: topları çiz ve göster
    visualize_balls_on_frame(frame, roi, balls)

    print("\n=== İlk 3 ralli için band çarpma analizi ===")
    for ridx in range(3):
        start, end = rallies[ridx]
        band_counts, _, hit_sequence = process_rally_white_and_bands(
            args.video,
            roi,
            start,
            end,
            initial_white=None,
            debug=False  # istersen burayı True yapıp debug logları da görebilirsin
        )
        total_bands = sum(band_counts.values())

        # Sıralamayı sadece band isimleri şeklinde yazalım:
        hit_order = [side for (side, f) in hit_sequence]

        print(f"Ralli {ridx + 1}: frame {start}-{end}")
        print(f"  band_counts = {band_counts}, total = {total_bands}")
        print(f"  hit order   = {hit_order}")

    # --- DEBUG: 2. rallinin band hit'lerini video üzerinde göster ---
    # Örneğin ikinci ralliyi inceleyelim:
    ridx = 2  # 0: 1. ralli, 1: 2. ralli, 2: 3. ralli ...

    start, end = rallies[ridx]
    band_counts, positions, hit_sequence = process_rally_white_and_bands(
        args.video,
        roi,
        start,
        end,
        initial_white=None,
        debug=False  # istersen True yapıp konsolda log da görebilirsin
    )

    print(f"\n[DEBUG] Ralli {ridx + 1} için görsel band hit debug başlıyor...")
    print(f"  band_counts = {band_counts}")
    print(f"  hit_sequence = {hit_sequence}")

    visualize_rally_hits(
        args.video,
        roi,
        start,
        end,
        positions,
        hit_sequence,
        window_name=f"Rally {ridx + 1} debug",
        play_delay=20  # normal akışta frameler arası bekleme (ms)
    )


if __name__ == "__main__":
    main()
