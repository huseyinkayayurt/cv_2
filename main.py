import cv2
import numpy as np

INPUT_VIDEO  = "input.mp4"
OUTPUT_VIDEO = ""   # İstersen boş bırakıp kaydetmeyebilirsin

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

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

    ordered = order_points(pts.astype("float32"))
    return ordered

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

    # Video özellikleri
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter (istersen kapatmak için OUTPUT_VIDEO = "" yapabilirsin)
    writer = None
    if OUTPUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # İlk frame’i tekrar kullanmak için pozisyonu başa sar
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    table_corners_int = table_corners.astype(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Masayı video akışı bozulmadan aynı köşelerle çiz
        cv2.polylines(frame, [table_corners_int], isClosed=True,
                      color=(0, 0, 255), thickness=3)

        for (x, y) in table_corners_int:
            cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)

        # Ekrana göster (q ile çık)
        cv2.imshow("Bilardo Masasi Isaretli Video", frame)
        if writer is not None:
            writer.write(frame)

        # 1 ms bekle – fps’yi çok bozmaz, ESC için kontrol
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
