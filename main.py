import cv2
import numpy as np

INPUT_VIDEO  = "input.mp4"
OUTPUT_VIDEO = ""   # İstersen boş bırakıp kaydetmeyebilirsin

def order_points(pts):
    """
    pts: (N,2) noktalar. Çıktı sırası: tl, tr, br, bl
    Daha sağlam yöntem:
      1) y'ye göre sırala (üst 2, alt 2)
      2) her grubu x'e göre sırala (sol, sağ)
    """
    pts = np.array(pts, dtype="float32")

    # y'ye göre sırala (küçük y = üst)
    idx_by_y = np.argsort(pts[:, 1])
    pts_sorted = pts[idx_by_y]

    top = pts_sorted[:2]
    bottom = pts_sorted[2:]

    # üsttekileri x'e göre sırala -> tl, tr
    top = top[np.argsort(top[:, 0])]
    tl, tr = top

    # alttakileri x'e göre sırala -> bl, br
    bottom = bottom[np.argsort(bottom[:, 0])]
    bl, br = bottom

    return np.array([tl, tr, br, bl], dtype="float32")

def detect_table(frame):
    """İlk frame üzerinde MAVİ masayı tespit eder, 4 köşe döner."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mavi/turkuaz bilardo masası için HSV aralığı
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Masa konturu bulunamadı.")
        return None

    table_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(table_contour, True)
    approx = cv2.approxPolyDP(table_contour, epsilon, True)

    if len(approx) < 4:
        rect = cv2.minAreaRect(table_contour)
        box = cv2.boxPoints(rect)
        approx = np.int0(box)

    pts = approx.reshape(-1, 2)

    if pts.shape[0] > 4:
        hull = cv2.convexHull(pts)
        if hull.shape[0] >= 4:
            pts = hull.reshape(-1, 2)
        pts = pts[:4]

    if pts.shape[0] != 4:
        print("4 köşe bulunamadı, masayı dikdörtgen gibi modelleyemedik.")
        return None

    ordered = order_points(pts)
    print("[DEBUG] Masa köşeleri (tl, tr, br, bl):")
    print(ordered)
    return ordered

def compute_perspective_transform(table_corners):
    """
    Masa köşelerinden homografi matrisi ve çıktı boyutlarını hesaplar.
    table_corners: (4,2) float32, sırası: tl, tr, br, bl
    """
    (tl, tr, br, bl) = table_corners

    # Genişlikler
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    # Yükseklikler
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    if maxWidth < 10 or maxHeight < 10:
        print(f"[WARN] Hesaplanan masa boyutları çok küçük: {maxWidth}x{maxHeight}")
        return None, None, None

    print(f"[DEBUG] Warp size: {maxWidth} x {maxHeight}")

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(table_corners, dst)
    return M, maxWidth, maxHeight

def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("Video açılamadı:", INPUT_VIDEO)
        return

    # İlk frame: masayı tespit etmek için
    ret, first_frame = cap.read()
    if not ret:
        print("İlk frame okunamadı.")
        cap.release()
        return

    table_corners = detect_table(first_frame)
    if table_corners is None:
        print("Bilardo masası tespit edilemedi, video işaretlenmeyecek.")
        cap.release()
        return

    # Homografi ve çıktı boyutları
    M, warp_w, warp_h = compute_perspective_transform(table_corners)
    if M is None:
        print("Homografi hesaplanamadı, perspektif düzeltme iptal.")
        cap.release()
        return

    # Video özellikleri
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if OUTPUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # Videoyu başa sar
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    table_corners_int = table_corners.astype(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Orijinal görüntüde masayı işaretle
        cv2.polylines(frame, [table_corners_int], isClosed=True,
                      color=(0, 0, 255), thickness=3)
        for (x, y) in table_corners_int:
            cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)

        # 2) Tepeden görünüm: masa perspektif düzeltme
        warped = cv2.warpPerspective(frame, M, (warp_w, warp_h))

        cv2.imshow("Bilardo Masasi Isaretli Video", frame)
        cv2.imshow("Masa Tepeden (Warped)", warped)

        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if writer is not None:
        print("Isaretlenmis video kaydedildi ->", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
