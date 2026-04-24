import argparse
from pathlib import Path

import cv2


def extract_frames_from_video(video_path: Path, output_dir: Path, interval_sec: float, start_index: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 無法開啟影片: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"[WARN] 無效 FPS，略過: {video_path}")
        cap.release()
        return 0

    frame_step = max(1, int(round(interval_sec * fps)))
    frame_idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step == 0:
            out_name = f"{video_path.stem}_frame_{start_index + saved:06d}.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"[INFO] {video_path.name} -> {saved} 張")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="將資料夾內 mp4 批次轉成 jpg")
    parser.add_argument("--input", type=str, required=True, help="輸入影片資料夾")
    parser.add_argument("--output", type=str, required=True, help="輸出圖片資料夾")
    parser.add_argument("--interval", type=float, default=1.0, help="抽幀間隔秒數，預設 1.0")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        print(f"[WARN] 找不到 mp4: {input_dir}")
        return

    existing = len(list(output_dir.glob("*.jpg")))
    total_saved = 0
    for video in videos:
        saved = extract_frames_from_video(
            video_path=video,
            output_dir=output_dir,
            interval_sec=args.interval,
            start_index=existing + total_saved,
        )
        total_saved += saved

    print(f"[DONE] 共有 {len(videos)} 支影片，總共輸出 {total_saved} 張到 {output_dir}")


if __name__ == "__main__":
    main()