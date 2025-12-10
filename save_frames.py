import cv2
import os
import sys

def extract_frames(video_path,output_dir="frames"):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Video açılamadı!")

    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"frame_{frame_idx:06d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)

        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    print(f"Toplam {frame_idx} frame kaydedildi. Çıkış klasörü: {output_dir}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Kullanım: python script.py video_dosyasi.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    extract_frames(video_path)
