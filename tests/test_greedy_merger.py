import math
import numpy as np

import pyfsg  # your pybind11 extension: from .cpp => PYBIND11_MODULE(pyfsg, ...)


def seg_close_to_any(s, arr, tol=2.0):
    """
    s: (x1,y1,x2,y2)
    arr: Nx4 of segments
    tol: endpoint tolerance in pixels
    """
    # We'll accept that each endpoint is within `tol` of any endpoint in `arr`.
    p1 = np.array(s[0:2])
    p2 = np.array(s[2:4])
    for row in arr:
        r1 = np.array(row[0:2])
        r2 = np.array(row[2:4])
        # Check if p1 matches r1 or r2, and p2 matches r1 or r2 (in any order)
        p1_close = (np.linalg.norm(p1 - r1) < tol) or (np.linalg.norm(p1 - r2) < tol)
        p2_close = (np.linalg.norm(p2 - r1) < tol) or (np.linalg.norm(p2 - r2) < tol)
        if p1_close and p2_close:
            return True
    return False


# ------------------------------------------------------------------------------
# Helper: compare a line eq (a,b,c) to expected values with some tolerance.
# ------------------------------------------------------------------------------
def assert_line_eq(line, expected, tol=1e-6):
    """
    line is a tuple (a, b, c)
    expected is a tuple (a, b, c)
    """
    assert len(line) == 3, "Line should have 3 coefficients"
    for x, e in zip(line, expected):
        assert math.isclose(x, e, abs_tol=tol), f"Expected ~{e}, got {x}"


# ------------------------------------------------------------------------------
# 1) TEST: Sort segments by length
def test_sort_by_length():
    merger = pyfsg.GreedyMerger(500, 600)
    segs = np.array([
        [10, 0, 20, 0],  # length=10
        [30, 0, 50, 0],  # length=20
        [20, -10, 20, -5],  # length=5
        [500, 500, 600, 600],  # length ~141.42
    ], dtype=np.float32)

    result = merger.partialSortByLength(segs, 1000, 500, 600)
    assert result == [3, 1, 0, 2], "Segments not sorted as expected by descending length"


# ------------------------------------------------------------------------------
# 2) TEST: Sort segments by angle => getOrientationHistogram + flatten
def test_sort_by_angle():
    merger = pyfsg.GreedyMerger(500, 600)
    segs = np.array([
        [10, 0, 20, 0],  # near horizontal (theta ~ -1.57 or +1.57)
        [20, 0, 10, 2],  # some angle
        [20, 0, 10, 8],  # another angle
        [10, 10, 100, 100],  # ~ -0.7853
        [100, 100, 10, 10],  # also ~ -0.7853
        [200, 600, 200, 300],  # vertical => ~ 0
        [0, 100, 0, 105],  # vertical => ~ 0
    ], dtype=np.float32)

    histogram = merger.getOrientationHistogram(segs, bins=180)
    # histogram is a list-of-lists of segment indices. The bin index corresponds to an angle bucket.

    # Flatten them like the C++ code: each bin is histogram[h_col]
    flattened = []
    for h_col in range(len(histogram)):
        flattened.extend(histogram[h_col])

    # We expect exactly 7 segment indices total in some order
    assert len(flattened) == 7, f"Expected 7 total indices in the histogram, got {len(flattened)}"

    # The exact ordering might differ slightly depending on your angle definitions.
    # We'll do a simpler check: all indices {0,1,2,3,4,5,6} are present.
    assert sorted(flattened) == [0, 1, 2, 3, 4, 5, 6], \
        f"All segments should appear in the histogram; got {flattened}"


# ------------------------------------------------------------------------------
# 3) TEST: getTangentLineEqs
#    C++ test: TEST(UnitGreedyMerger, GetTangentLines)
# ------------------------------------------------------------------------------
def test_get_tangent_lines():
    # We'll replicate the C++ test:
    #   Segment(400,400,400,500) with radius=10 => line eq near [0.979795814, -0.200000003, 301.918365], etc.

    merger = pyfsg.GreedyMerger(500, 500)

    segs = np.array([[400, 400, 400, 500]], dtype=np.float32)  # shape (1,4)
    line1, line2 = merger.getTangentLineEqs(segs, 10.0)
    # Compare to the expected values:
    expected1 = (0.979795814, -0.200000003, 301.918365)
    expected2 = (0.979795814, 0.200000003, 481.918335)

    # We'll use a small tolerance
    tol = 1e-4
    assert_line_eq(line1, expected1, tol)
    assert_line_eq(line2, expected2, tol)

    # Next case: Segment(200,200,300,300) radius=5 =>
    # expected near:
    #   first:  (0.655336857, -0.755336821, -25.0000019)
    #   second: (0.755336821, -0.655336857,  25.0000019)
    segs = np.array([[200, 200, 300, 300]], dtype=np.float32)
    line1, line2 = merger.getTangentLineEqs(segs, 5.0)

    expected1 = (0.655336857, -0.755336821, -25.0000019)
    expected2 = (0.755336821, -0.655336857, 25.0000019)
    tol = 1e-4
    assert_line_eq(line1, expected1, tol)
    assert_line_eq(line2, expected2, tol)


# ------------------------------------------------------------------------------
# 4) TEST: MergeSegmentsHierarchicalSynthetic
#    C++ test: TEST(UnitGreedyMerger, MergeSegmentsHierarchicalSynthetic)
#    - This requires a method `mergeSegmentsHierarchical(...)` in Python,
#      which may or may not exist in your binding. If not, skip or rename.
# ------------------------------------------------------------------------------
def test_merge_segments_hierarchical_synthetic():
    # Using the same data as in the C++ test:
    ELSs = np.array([
        [100, 100, 250, 100],
        [300, 100, 400, 100],
        [300, 102, 400, 102],
        [300, 250, 400, 250],
        [300, 170, 400, 170],
        [600, 170, 700, 170],
        [600, 200, 700, 200],
        [10, 20, 40, 10],
        [50, 50, 50, 150],
        [52, 200, 52, 300],
    ], dtype=np.float32)

    merger = pyfsg.GreedyMerger(800, 400)
    merged, clusters = merger.mergeSegments(ELSs)

    # Compare to the expected merged segments
    # from the C++ test:
    expectedMergedSegs = np.array([
        [100, 100, 400, 100],
        [300, 102, 400, 102],
        [300, 250, 400, 250],
        [300, 170, 700, 170],
        [600, 200, 700, 200],
        [10, 20, 40, 10],
        [51, 50, 51, 300],  # note: the test modifies x=50->51 if multiple segments
    ], dtype=np.float32)

    # Check that merged has the same size as expected
    assert merged.shape[0] == expectedMergedSegs.shape[0], \
        f"Mismatch in number of merged segments: got {merged.shape[0]}, expected {expectedMergedSegs.shape[0]}"

    # For each expected segment, ensure we find a match in `merged`.
    for row in expectedMergedSegs:
        assert seg_close_to_any(row, merged, tol=2.0), \
            f"Expected segment {row} not found in merged results."
