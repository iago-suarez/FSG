from __future__ import annotations

from ._pyfsg import (__doc__, __version__, GreedyMerger, detectLinesOpencvLSD,
                     totalLeastSquareFitSegmentEndPts, projectPointIntoLine, filterSegments)

__all__ = ["__doc__", "__version__", "GreedyMerger", "detectLinesOpencvLSD",
           "totalLeastSquareFitSegmentEndPts", "projectPointIntoLine", "filterSegments"]
