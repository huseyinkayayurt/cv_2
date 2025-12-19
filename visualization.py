import cv2


def draw_frame_overlay(vis, frame_idx, roi, white_pos, event_history, rally_summary=None):

    h, w = vis.shape[:2]
    bands = roi["bands"]
    ix, iy, iw, ih = roi["inner"]

    cv2.rectangle(vis, (ix, iy), (ix + iw, iy + ih), (0, 255, 0), 1)

    cv2.line(vis, (bands["left"], bands["top"]), (bands["left"], bands["bottom"]), (255, 0, 0), 2)
    cv2.line(vis, (bands["right"], bands["top"]), (bands["right"], bands["bottom"]), (0, 0, 255), 2)
    cv2.line(vis, (bands["left"], bands["top"]), (bands["right"], bands["top"]), (0, 255, 0), 2)
    cv2.line(vis, (bands["left"], bands["bottom"]), (bands["right"], bands["bottom"]), (0, 255, 255), 2)

    if white_pos is not None:
        cx, cy, r = white_pos
        pad = int(max(12, 2 * r))
        x1 = max(0, int(cx - pad))
        y1 = max(0, int(cy - pad))
        x2 = min(w - 1, int(cx + pad))
        y2 = min(h - 1, int(cy + pad))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.putText(vis, f"frame {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, f"frame {frame_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    
    should_pause = False

    if event_history:
        box_height = min(35 + len(event_history) * 22, h - 100)
        overlay = vis.copy()
        cv2.rectangle(overlay, (w - 320, 50), (w - 10, 50 + box_height), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
        
        cv2.putText(vis, "TESPIT GECMISI:", (w - 310, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        y0 = 92
        max_display = min(len(event_history), 12)
        start_idx = max(0, len(event_history) - max_display)
        
        for i in range(start_idx, len(event_history)):
            evt = event_history[i]
            color = (255, 255, 255)
            if "RED" in evt:
                color = (0, 0, 255)
            elif "YELLOW" in evt:
                color = (0, 255, 255)
            elif "BAND" in evt:
                color = (0, 255, 0)
            cv2.putText(vis, evt, (w - 310, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y0 += 20

    if rally_summary is not None:
        overlay = vis.copy()
        cv2.rectangle(overlay, (40, 60), (w - 40, 220), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
        
        y0 = 90
        for line in rally_summary:
            cv2.putText(vis, line, (60, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y0 += 25
        
        should_pause = True
    
    return vis, should_pause

