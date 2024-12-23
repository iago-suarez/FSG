from __future__ import annotations
import numpy as np
import math
import pyfsg  # Your pybind11 module with the new NumPy-based API


def test_fsg():
    merger = pyfsg.GreedyMerger(width=640, height=480)

    segs = np.array([
        [0, 0, 100, 100],
        [10, 20, 80, 90]
    ], dtype=np.float32)  # shape (0,4)

    merged, clusters = merger.mergeSegments(segs)
    print("Merged segments:", merged)
    print("Clusters:", clusters)


def test_merge_empty_segments():
    """
    Merging an empty Nx4 array => empty Nx4 array and empty clusters.
    """
    merger = pyfsg.GreedyMerger(640, 480)
    segs = np.zeros((0, 4), dtype=np.float32)  # shape (0,4)

    merged, clusters = merger.mergeSegments(segs)
    assert merged.shape == (0, 4), "Expected no merged segments."
    assert len(clusters) == 0, "Expected no clusters for empty input."


def test_merge_single_segment():
    """
    Merging one segment => single merged segment + one cluster with [0].
    """
    merger = pyfsg.GreedyMerger(640, 480)
    segs = np.array([[10, 10, 20, 20]], dtype=np.float32)  # shape (1,4)

    merged, clusters = merger.mergeSegments(segs)
    assert merged.shape == (1, 4), "Expected 1 merged segment."
    assert len(clusters) == 1, "Expected 1 cluster."
    assert len(clusters[0]) == 1, "Cluster should contain exactly one index."


def test_merge_two_collinear_segments():
    """
    Two collinear segments => single merged segment + single cluster with [0,1].
    """
    merger = pyfsg.GreedyMerger(640, 480)
    segs = np.array([
        [10, 10, 20, 20],
        [20, 20, 30, 30]
    ], dtype=np.float32)  # shape (2,4)

    merged, clusters = merger.mergeSegments(segs)
    # Typically, collinear + adjacent => merged into one line
    # Implementation detail: your code might combine them or not; this is the assumption.
    assert merged.shape[0] == 1, "Expected 1 merged segment from two collinear segments."
    assert len(clusters) == 1, "Expected 1 cluster."
    assert len(clusters[0]) == 2, "Cluster should contain both segments."


def test_merge_two_non_collinear_segments():
    """
    Two perpendicular segments => remain separate, i.e. 2 merged segments, 2 clusters.
    """
    merger = pyfsg.GreedyMerger(640, 480)
    segs = np.array([
        [10, 10, 50, 10],  # horizontal
        [20, 20, 20, 60],  # vertical
    ], dtype=np.float32)

    merged, clusters = merger.mergeSegments(segs)
    # Typically, these won't be merged (they're perpendicular).
    assert merged.shape[0] == 2, "Expected 2 merged segments for perpendicular input."
    assert len(clusters) == 2, "Expected 2 clusters."
    for c in clusters:
        assert len(c) == 1, "Each cluster should contain exactly one segment index."


def test_partial_sort_by_length():
    """
    partialSortByLength => return indices sorted by descending length.
    """
    segs = np.array([
        [0, 0, 10, 0],  # length 10
        [0, 0, 100, 0],  # length 100
        [0, 0, 50, 0],  # length 50
    ], dtype=np.float32)

    sorted_indices = pyfsg.GreedyMerger.partialSortByLength(segs, 1000, 640, 480)
    # We expect them in descending order: 1 (len=100), 2 (len=50), 0 (len=10)
    lengths = [math.hypot(segs[i, 2] - segs[i, 0], segs[i, 3] - segs[i, 1]) for i in sorted_indices]

    assert lengths == sorted(lengths, reverse=True), "Segments not in descending length order."
    # Optional exact check:
    assert sorted_indices[0] == 1, "Longest segment index should be 1."
    assert sorted_indices[1] == 2, "2nd longest segment index should be 2."
    assert sorted_indices[2] == 0, "Shortest segment index should be 0."


def test_get_orientation_histogram():
    """
    Check that near-horizontal segments go in the same bin,
    near-vertical in a separate bin, etc.
    """
    segs = np.array([
        [20, 20, 100, 21.5],  # near-horizontal
        [10, 10, 90, 15.],  # near-horizontal
        [50, 50, 49, 100],  # near-vertical
    ], dtype=np.float32)

    clusters = pyfsg.GreedyMerger().getOrientationHistogram(segs, bins=4)
    # bins=4 => each bin ~ 45 degrees.
    # Typically, the first 2 segments are near the same orientation bin,
    # the 3rd is in a different bin.
    assert len(clusters[0]) == 2
    assert (clusters[0][0] == 0) and (clusters[0][1] == 1)
    assert len(clusters[2]) == 1
    assert clusters[2][0] == 2


def test_get_tangent_line_eqs():
    """
    We pass in an array with at least one segment (1x4).
    Expect two (a,b,c) lines for the conic around this segment.
    """
    seg = np.array([[10, 10, 20, 20]], dtype=np.float32)
    line1, line2 = pyfsg.GreedyMerger.getTangentLineEqs(seg, 5.0)

    # line1, line2 => (a,b,c) each.
    assert len(line1) == 3, "Line eq must have 3 coefficients (a, b, c)."
    assert len(line2) == 3, "Line eq must have 3 coefficients (a, b, c)."

    # Quick check that they are not all zeros
    assert any(coeff != 0 for coeff in line1), "Line1 shouldn't be all zeros."
    assert any(coeff != 0 for coeff in line2), "Line2 shouldn't be all zeros."


def test_total_least_squares_and_project_ptn():
    # totalLeastSquareFitSegmentEndPts
    segs = np.array([
        [10, 10, 20, 20],
        [20, 20, 30, 30],
        [50, 0, 60, 80],
    ], dtype=np.float32)

    (a, b, c) = pyfsg.totalLeastSquareFitSegmentEndPts(segs)
    print("Line eq for all segments: a=%.4f, b=%.4f, c=%.4f" % (a, b, c))

    # getProjectionPtn
    px, py = pyfsg.projectPointIntoLine([a, b, c], [15, 15])
    print("Projection of (15,15) on line = (%.2f, %.2f)" % (px, py))


def test_filter_segments():
    # Suppose we have Nx4 segments
    originalSegs = np.array([
        [10, 10, 20, 10],
        [20, 10, 30, 10],
        [20, 20, 40, 25],
        [120, 120, 122, 125],
    ], dtype=np.float32)

    clusters = [[0, 1], [2], [3]]
    length_threshold = 15.0

    filtered, noisy = pyfsg.filterSegments(originalSegs, clusters, length_threshold)

    expected_filtered = np.array([[10, 10, 30, 10], [20, 20, 40, 25]], dtype=np.float32)
    expected_noisy = np.array([[120, 120, 122, 125]], dtype=np.float32)
    np.testing.assert_allclose(filtered, expected_filtered)
    np.testing.assert_allclose(noisy, expected_noisy)
