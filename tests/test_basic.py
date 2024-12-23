from __future__ import annotations

import pyfsg


def test_version():
    assert pyfsg.__version__ == "0.0.1"


def test_add():
    assert pyfsg.add(1, 2) == 3


def test_sub():
    assert pyfsg.subtract(1, 2) == -1


def test_fsg():
    merger = pyfsg.GreedyMerger(width=640, height=480)

    segments = pyfsg.Segments()
    segments.append(pyfsg.Segment(0, 0, 100, 100))
    segments.append(pyfsg.Segment(10, 20, 80, 90))

    merged, clusters = merger.mergeSegments(segments)
    print("Merged segments:", merged)
    print("Clusters:", clusters)
