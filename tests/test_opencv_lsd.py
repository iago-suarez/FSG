from pathlib import Path

import pyfsg
import cv2

IMSHOW = False


def test_opencv_lsd():
    img_path = str(Path(__file__).parent.parent / "images" / "P1080079.jpg")
    print(f"Reading image from {img_path}")

    # Load a grayscale image as uint8
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 1) Basic usage: just get Nx4 lines
    lines = pyfsg.detectLinesOpencvLSD(gray)
    print("Detected lines:", lines.shape)  # e.g. (150, 4)

    # color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # for line in lines.astype(int):
    #     cv2.line(color, (line[0], line[1]), (line[2], line[3]), (0, 255, 0))
    #
    # cv2.imshow("Result", color.copy())
    # cv2.waitKey()

    # 2) Request optional arrays
    res = pyfsg.detectLinesOpencvLSD(gray, return_width=True, return_prec=True, return_nfa=True)

    print("Detected lines shape:", res["lines"].shape)  # (N,4)
    print("Widths shape:", res["width"].shape)  # (N,)
    print("Precisions shape:", res["prec"].shape)  # (N,)
    print("NFA shape:", res["nfa"].shape)  # (N,)
