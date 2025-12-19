import cv2
import numpy as np
import math

# --- AYARLAR ---
DEBUG_MODE = True
DEBUG_LIMIT_SEC = 240


class BilardoAnalyzer:
    def __init__(self):
        self.reset_rally()
        self.table_mask = None
        self.table_rect = None
        self.prev_yellow_pos = None
        self.prev_red_pos = None
        # YENİ: Çarpışma bekleyen durumlar
        self.yellow_collision_pending = 0  # Kaç frame daha izlenecek
        self.red_collision_pending = 0
        self.yellow_initial_pos = None  # Çarpışma anındaki konum
        self.red_initial_pos = None

    def reset_rally(self):
        self.banda_sayisi = 0
        self.sari_carpti = False
        self.kirmizi_carpti = False
        self.banda_cooldown = 0
        self.prev_yellow_pos = None
        self.prev_red_pos = None
        self.yellow_collision_pending = 0
        self.red_collision_pending = 0
        self.yellow_initial_pos = None
        self.red_initial_pos = None

    def detect_table_structure(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([80, 100, 50])
        upper_blue = np.array([140, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) < 1000:
                return None, None

            x, y, w, h = cv2.boundingRect(largest)
            scale_factor = 2
            offset = 15
            rect_scaled = (x * scale_factor + offset, y * scale_factor + offset,
                           w * scale_factor - 2 * offset, h * scale_factor - 2 * offset)

            h_orig, w_orig = frame.shape[:2]
            full_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
            largest_scaled = largest * scale_factor
            cv2.drawContours(full_mask, [largest_scaled], -1, (255), thickness=cv2.FILLED)
            dilate_kernel = np.ones((10, 10), np.uint8)
            full_mask = cv2.dilate(full_mask, dilate_kernel, iterations=1)

            return rect_scaled, full_mask
        return None, None

    def detect_balls(self, frame, table_mask):
        blurred = cv2.GaussianBlur(frame, (9, 9), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv, (0, 0, 130), (180, 60, 255))
        yellow_mask = cv2.inRange(hsv, (15, 80, 80), (45, 255, 255))
        r_mask1 = cv2.inRange(hsv, (0, 140, 80), (10, 255, 255))
        r_mask2 = cv2.inRange(hsv, (170, 140, 80), (180, 255, 255))
        red_mask = r_mask1 + r_mask2

        if table_mask is not None:
            white_mask = cv2.bitwise_and(white_mask, white_mask, mask=table_mask)
            yellow_mask = cv2.bitwise_and(yellow_mask, yellow_mask, mask=table_mask)
            red_mask = cv2.bitwise_and(red_mask, red_mask, mask=table_mask)

        return white_mask, yellow_mask, red_mask

    def is_red_screen(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 80])
        upper_red1 = np.array([20, 255, 255])
        lower_red2 = np.array([160, 100, 80])
        upper_red2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        red_pixels = cv2.countNonZero(mask)
        total_pixels = small_frame.shape[0] * small_frame.shape[1]
        return (red_pixels / total_pixels) > 0.04

    def get_center_and_box(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30: continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5: continue
            if area > max_area:
                max_area = area
                best_cnt = cnt
        if best_cnt is not None:
            M = cv2.moments(best_cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(best_cnt)
                return (cx, cy), (x, y, w, h)
        return None, None

    def check_cushion_collision(self, ball_pos, table_rect):
        if not ball_pos or not table_rect: return
        bx, by = ball_pos
        tx, ty, tw, th = table_rect
        threshold = 25
        if self.banda_cooldown > 0:
            self.banda_cooldown -= 1
            return
        hit_left = abs(bx - tx) < threshold
        hit_right = abs(bx - (tx + tw)) < threshold
        hit_top = abs(by - ty) < threshold
        hit_bottom = abs(by - (ty + th)) < threshold
        if hit_left or hit_right or hit_top or hit_bottom:
            self.banda_sayisi += 1
            self.banda_cooldown = 20

    def check_ball_collision(self, white_pos, white_radius, target_pos, target_radius, target_color):
        """
        YENİ YAKLAŞIM: Çarpışma algılandığında 10 frame boyunca hareketi izle
        """
        if not white_pos or not target_pos:
            return 0

        distance = math.sqrt((white_pos[0] - target_pos[0]) ** 2 + (white_pos[1] - target_pos[1]) ** 2)
        collision_threshold = (white_radius + target_radius) * 1.5

        # Bekleyen çarpışma kontrolü
        if target_color == 'yellow':
            if self.yellow_collision_pending > 0:
                self.yellow_collision_pending -= 1
                # İlk konumdan ne kadar hareket etti?
                if self.yellow_initial_pos:
                    movement = math.sqrt(
                        (target_pos[0] - self.yellow_initial_pos[0]) ** 2 +
                        (target_pos[1] - self.yellow_initial_pos[1]) ** 2
                    )
                    # 1 piksel bile hareket ettiyse sayılır
                    if movement > 1.0 and not self.sari_carpti:
                        self.sari_carpti = True
                        print(f"[SARI] Hareket tespit edildi: {movement:.2f} piksel")
                        self.yellow_collision_pending = 0

        elif target_color == 'red':
            if self.red_collision_pending > 0:
                self.red_collision_pending -= 1
                if self.red_initial_pos:
                    movement = math.sqrt(
                        (target_pos[0] - self.red_initial_pos[0]) ** 2 +
                        (target_pos[1] - self.red_initial_pos[1]) ** 2
                    )
                    if movement > 1.0 and not self.kirmizi_carpti:
                        self.kirmizi_carpti = True
                        print(f"[KIRMIZI] Hareket tespit edildi: {movement:.2f} piksel")
                        self.red_collision_pending = 0

        # Yeni çarpışma kontrolü
        if distance < collision_threshold:
            if target_color == 'yellow' and not self.sari_carpti and self.yellow_collision_pending == 0:
                print(f"[SARI] Çarpışma algılandı! 10 frame izlenecek...")
                self.yellow_collision_pending = 10  # 10 frame izle
                self.yellow_initial_pos = target_pos  # Şu anki konumu kaydet

            elif target_color == 'red' and not self.kirmizi_carpti and self.red_collision_pending == 0:
                print(f"[KIRMIZI] Çarpışma algılandı! 10 frame izlenecek...")
                self.red_collision_pending = 10
                self.red_initial_pos = target_pos

        return collision_threshold


# --- ANA PROGRAM ---
def main():
    cap = cv2.VideoCapture('input.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30

    analyzer = BilardoAnalyzer()

    rally_count = 1
    in_red_screen = False
    frame_counter = 0
    non_red_frame_counter = 0

    print("Program Başladı...")
    print("-" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_counter += 1
        current_seconds = frame_counter / fps

        if current_seconds < 2.5:
            cv2.imshow('Bilardo Analiz', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        if DEBUG_MODE and current_seconds > DEBUG_LIMIT_SEC:
            print("[DEBUG] Süre doldu.")
            break

        is_red = analyzer.is_red_screen(frame)

        if is_red:
            non_red_frame_counter = 0

            if not in_red_screen:
                in_red_screen = True

                success = (analyzer.banda_sayisi >= 3 and
                           analyzer.sari_carpti and
                           analyzer.kirmizi_carpti)

                result_text = "BASARILI" if success else "BASARISIZ"
                color = (0, 255, 0) if success else (0, 0, 255)
                sari_txt = 'EVET' if analyzer.sari_carpti else 'HAYIR'
                kirmizi_txt = 'EVET' if analyzer.kirmizi_carpti else 'HAYIR'

                print(f"RALLI {rally_count} SONUCU: {result_text}")
                print(f"Bant: {analyzer.banda_sayisi} | Sari: {sari_txt} | Kirmizi: {kirmizi_txt}")
                print("-" * 50)

                summary_frame = np.zeros_like(frame)
                info = [
                    f"RALLI {rally_count} SONUCU:",
                    f"----------------",
                    f"Bant Sayisi: {analyzer.banda_sayisi}",
                    f"Sari Top Temas: {sari_txt}",
                    f"Kirmizi Top Temas: {kirmizi_txt}",
                    "",
                    "Devam etmek icin bir tusa basin..."
                ]

                y = 150
                for line in info:
                    cv2.putText(summary_frame, line, (50, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    y += 50

                cv2.putText(summary_frame, result_text, (400, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                cv2.imshow('Bilardo Analiz', summary_frame)
                cv2.waitKey(0)

                rally_count += 1
                analyzer.reset_rally()

        else:
            if in_red_screen:
                non_red_frame_counter += 1
                if non_red_frame_counter > 15:
                    in_red_screen = False
                    non_red_frame_counter = 0
            else:
                table_rect, table_mask = analyzer.detect_table_structure(frame)

                if table_rect:
                    tx, ty, tw, th = table_rect
                    cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (255, 0, 0), 2)
                    margin = 25
                    cv2.rectangle(frame, (tx + margin, ty + margin), (tx + tw - margin, ty + th - margin), (0, 255, 0),
                                  1)

                w_mask, y_mask, r_mask = analyzer.detect_balls(frame, table_mask)
                w_pos, w_rect = analyzer.get_center_and_box(w_mask)
                y_pos, y_rect = analyzer.get_center_and_box(y_mask)
                r_pos, r_rect = analyzer.get_center_and_box(r_mask)

                if w_pos:
                    x, y, w, h = w_rect
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    w_radius = min(w, h) / 2

                    if table_rect:
                        analyzer.check_cushion_collision(w_pos, table_rect)

                    if y_pos:
                        y_x, y_y, y_w, y_h = y_rect
                        cv2.rectangle(frame, (y_x, y_y), (y_x + y_w, y_y + y_h), (0, 255, 255), 2)
                        y_radius = min(y_w, y_h) / 2
                        thresh = analyzer.check_ball_collision(w_pos, w_radius, y_pos, y_radius, 'yellow')
                        cv2.circle(frame, w_pos, int(thresh), (255, 0, 255), 1)

                        # Bekleyen çarpışma göstergesi
                        if analyzer.yellow_collision_pending > 0:
                            cv2.putText(frame, f"Sari izleniyor: {analyzer.yellow_collision_pending}",
                                        (y_x, y_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    if r_pos:
                        r_x, r_y, r_w, r_h = r_rect
                        cv2.rectangle(frame, (r_x, r_y), (r_x + r_w, r_y + r_h), (0, 0, 255), 2)
                        r_radius = min(r_w, r_h) / 2
                        thresh = analyzer.check_ball_collision(w_pos, w_radius, r_pos, r_radius, 'red')
                        cv2.circle(frame, w_pos, int(thresh), (255, 0, 255), 1)

                        if analyzer.red_collision_pending > 0:
                            cv2.putText(frame, f"Kirmizi izleniyor: {analyzer.red_collision_pending}",
                                        (r_x, r_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                analyzer.prev_yellow_pos = y_pos
                analyzer.prev_red_pos = r_pos

                status = f"Ralli: {rally_count} | Bant: {analyzer.banda_sayisi} | Sari: {int(analyzer.sari_carpti)} | Kir: {int(analyzer.kirmizi_carpti)}"
                cv2.rectangle(frame, (0, 0), (650, 60), (0, 0, 0), -1)
                cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('Bilardo Analiz', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
