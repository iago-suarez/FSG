import cv2
import random

import numpy as np
import pyfsg


def draw_segments(img, segments, color, thickness=1, line_type=cv2.LINE_8, shift=0):
    """
    Draws a list of segments onto an image.
    segments is a NumPy array (N,4) or list of [x1,y1,x2,y2].
    """
    for seg in np.array(segments).astype(int):
        x1, y1, x2, y2 = seg
        cv2.line(img, (x1, y1), (x2, y2), color, thickness=thickness, lineType=line_type, shift=shift)


def draw_clusters(img, segments, clusters,
                  thickness=2,
                  color=(0, 0, 0),
                  draw_line_cluster=True,
                  line_type=cv2.LINE_AA,
                  shift=0):
    """
    Draws each cluster in a random color (if color == (0,0,0)) or the given color.
    If draw_line_cluster == True and a cluster has >1 segments,
    we also draw the 'fitted' line through all of them (using totalLeastSquareFitSegmentEndPts).
    """
    random_c = (color == (0, 0, 0))

    for cluster in clusters:
        # Possibly pick a random color
        if random_c:
            clr = random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)
        else:
            clr = color

        if len(cluster) > 1 and draw_line_cluster:
            # totalLeastSquareFitSegmentEndPts -> returns (a, b, c) for line eq or sometimes a cv::Vec3d
            line_eq = pyfsg.totalLeastSquareFitSegmentEndPts(segments, cluster)

            # We'll replicate the logic of projecting each endpoint onto that line,
            # tracking min & max along the x-axis of the line.
            # min_p, max_p start as corners of the image:
            max_p = [0, 0]
            min_p = [img.shape[1], img.shape[0]]

            for idx in cluster:
                x1, y1, x2, y2 = segments[idx]
                # projectPointIntoLine(line_eq, (x,y)) -> returns (px,py)
                p1 = pyfsg.projectPointIntoLine(line_eq, (x1, y1))
                p2 = pyfsg.projectPointIntoLine(line_eq, (x2, y2))

                # Update max/min by comparing the x-coordinates (like in your C++ code)
                if p1[0] > max_p[0]:
                    max_p = [p1[0], p1[1]]
                if p1[0] < min_p[0]:
                    min_p = [p1[0], p1[1]]
                if p2[0] > max_p[0]:
                    max_p = [p2[0], p2[1]]
                if p2[0] < min_p[0]:
                    min_p = [p2[0], p2[1]]

            # Finally draw that line
            cv2.line(img,
                     (int(min_p[0]), int(min_p[1])),
                     (int(max_p[0]), int(max_p[1])),
                     clr,
                     thickness=max(1, int(round(thickness / 3.0))),
                     lineType=line_type,
                     shift=shift)

        # Now draw the individual segments
        segs_this_cluster = [segments[idx] for idx in cluster]
        draw_segments(img, segs_this_cluster, clr, thickness, line_type, shift)


def main():
    # 1) Load the input image
    img = cv2.imread("images/P1080079.jpg")
    if img is None or img.size == 0:
        print("Error: Cannot read input image.")
        return

    # 2) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3) Detect line segments with LSD
    lines = pyfsg.detectLinesOpencvLSD(gray)  # Nx4 float array

    print(f"Detected {lines.shape[0]} line segments with LSD.")

    # We'll draw them in red on a copy of the original image
    img_detected = img.copy()
    draw_segments(img_detected, lines, (255, 0, 0), thickness=1)
    cv2.imwrite("Detected_line_segments.png", img_detected)

    # 4) Merge segments into clusters
    # Create a GreedyMerger for the image size
    h, w = gray.shape
    merger = pyfsg.GreedyMerger(w, h)

    merged_segments, clusters = merger.mergeSegments(lines)  # returns (Nx4 array, list-of-lists)

    # 5) Draw the clusters
    img_clusters = img.copy()
    draw_clusters(img_clusters, lines, clusters, thickness=2)
    cv2.imwrite("Segment_groups.png", img_clusters)

    # Get large lines from groups of segments
    filteredSegments, noisySegs = pyfsg.filterSegments(lines, clusters, 30.0)

    # 7) Draw the filtered (in green) vs. noisy (in red) on another copy
    img_filtered = img.copy()
    draw_segments(img_filtered, filteredSegments, (0, 255, 0), thickness=2)
    draw_segments(img_filtered, noisySegs, (255, 0, 0), thickness=2)
    cv2.imwrite("Obtained_lines.png", img_filtered)

    print("Done. Results saved to:")
    print("  Detected_line_segments.png")
    print("  Segment_groups.png")
    print("  Obtained_lines.png")

    # If you want to visualize them on screen:
    # cv2.imshow("Detected line segments", img_detected)
    # cv2.imshow("Segment groups", img_clusters)
    # cv2.imshow("Obtained lines", img_filtered)
    # cv2.waitKey(0)


if __name__ == "__main__":
    main()
