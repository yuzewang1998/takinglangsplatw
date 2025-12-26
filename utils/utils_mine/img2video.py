import cv2
import os
import argparse


def create_video(input_folder, output_folder, output_filename="output.mp4", fps=30):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    images = []
    for i in range(0, 301):  # 100 to 300 (inclusive)
        img_path = os.path.join(input_folder, f"val_{i}1.png")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Could not read {img_path}")
        else:
            print(f"Warning: {img_path} does not exist")

    if not images:
        print("No images found. Exiting.")
        return

    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in images:
        for _ in range(max(1, 30 // fps)):
            video_writer.write(img)

    video_writer.release()
    print(f"Video saved at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert sequence of images to video")
    parser.add_argument("--input_folder", type=str, default="/home/wangyz/Documents/projects/0working/langsplat-w/output/wegs_VanillaCLIPExtUncetainDM-T-100epoch_catFeature12dReconUncertainlyTMAM60Kiter_pantheon_exterior/wegs_VanillaCLIPExtUncetainDM-T-100epoch_catFeature12dReconUncertainlyTMAM60Kiter_pantheon_exterior_1/eval/ours_None/renders")
    parser.add_argument("--output_folder", type=str, default="/home/wangyz/Downloads")
    parser.add_argument("--output_filename", type=str, default="output.mp4")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the video")

    args = parser.parse_args()
    create_video(args.input_folder, args.output_folder, args.output_filename, args.fps)
