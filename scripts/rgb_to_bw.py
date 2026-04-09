import argparse
import os

import cv2


def convert_rgb_to_bw(input_path: str, output_path: str) -> None:
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if not cv2.imwrite(output_path, gray):
        raise IOError(f"Failed to write output image: {output_path}")

    print(f"Saved black-and-white image to: {output_path}")


def pick_default_input() -> str:
    candidates = [
        "data/visual_deth.png",
        "data/visual_depth.png",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an RGB depth image to black and white (grayscale)."
    )
    parser.add_argument(
        "--input",
        default=pick_default_input(),
        help="Path to the RGB input image.",
    )
    parser.add_argument(
        "--output",
        default="data/visual_deth_bw.png",
        help="Path for the black-and-white output image.",
    )
    args = parser.parse_args()

    convert_rgb_to_bw(args.input, args.output)


if __name__ == "__main__":
    main()
