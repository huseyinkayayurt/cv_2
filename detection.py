import cv2
import numpy as np
from config import (
    TABLE_HSV_LOWER, TABLE_HSV_UPPER, TABLE_MARGIN, BAND_INSET,
    RED_HSV_LOWER1, RED_HSV_UPPER1, RED_HSV_LOWER2, RED_HSV_UPPER2,
    YELLOW_HSV_LOWER, YELLOW_HSV_UPPER,
    WHITE_HSV_LOWER, WHITE_HSV_UPPER,
    BALL_MIN_AREA, BALL_RADIUS_MIN, BALL_RADIUS_MAX,
    BALL_MIN_CIRCULARITY, BALL_MAX_ASPECT_RATIO
)


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


def detect_table_roi(frame, margin=TABLE_MARGIN, band_inset=BAND_INSET):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_table = np.array(TABLE_HSV_LOWER, dtype=np.uint8)
    upper_table = np.array(TABLE_HSV_UPPER, dtype=np.uint8)
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
    kernel = np.ones((3, 3), np.uint8)
    
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

    lower_red1 = np.array(RED_HSV_LOWER1, dtype=np.uint8)
    upper_red1 = np.array(RED_HSV_UPPER1, dtype=np.uint8)
    lower_red2 = np.array(RED_HSV_LOWER2, dtype=np.uint8)
    upper_red2 = np.array(RED_HSV_UPPER2, dtype=np.uint8)
    
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    rx, ry, rr = find_largest_contour(mask_red)
    if rx is not None:
        balls["red"] = {"center": (int(ix + rx), int(iy + ry)), "radius": float(rr)}

    lower_yellow = np.array(YELLOW_HSV_LOWER, dtype=np.uint8)
    upper_yellow = np.array(YELLOW_HSV_UPPER, dtype=np.uint8)
    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    yx, yy, yr = find_largest_contour(mask_y)
    if yx is not None:
        balls["yellow"] = {"center": (int(ix + yx), int(iy + yy)), "radius": float(yr)}

    lower_white = np.array(WHITE_HSV_LOWER, dtype=np.uint8)
    upper_white = np.array(WHITE_HSV_UPPER, dtype=np.uint8)
    mask_w = cv2.inRange(hsv, lower_white, upper_white)
    mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_w = cv2.morphologyEx(mask_w, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand_center = None
    cand_radius = None
    max_score = 0.0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < BALL_MIN_AREA:
            continue
        
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if not (BALL_RADIUS_MIN < radius < BALL_RADIUS_MAX):
            continue
        
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        
        circularity = 4.0 * np.pi * area / (peri * peri)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw == 0 or bh == 0:
            continue
        
        aspect = max(bw, bh) / float(min(bw, bh))
        
        if circularity < BALL_MIN_CIRCULARITY:
            continue
        if aspect > BALL_MAX_ASPECT_RATIO:
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

