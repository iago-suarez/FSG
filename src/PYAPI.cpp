#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>        // for py::array_t
#include <opencv2/core.hpp>

#include "GreedyMerger.h"  // your library's header that defines upm::GreedyMerger

namespace py = pybind11;
using namespace upm;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace upm;

//------------------------------------------------------------------------------
// HELPER FUNCTIONS: Convert between NumPy arrays (N×4) and Segments
//------------------------------------------------------------------------------

/**
 * @brief Convert a NumPy array (N x 4, dtype=float32 or float64) -> Segments.
 * @param arr Python array, must be 2D with shape [N,4].
 * @throws std::runtime_error if shape is wrong.
 */
static Segments ndarrayToSegments(const py::array_t<float> &arr)
{
    // Request buffer info from NumPy array
    py::buffer_info buf = arr.request();

    if (buf.ndim != 2 || buf.shape[1] != 4) {
        throw std::runtime_error("Segments array must have shape (N,4)");
    }

    // Convert to Segments
    size_t n = buf.shape[0];
    float *ptr = static_cast<float*>(buf.ptr);

    Segments segments(n);
    for (size_t i = 0; i < n; ++i) {
        // each row: (x1, y1, x2, y2)
        segments[i] = cv::Vec4f(ptr[4*i + 0],
                                ptr[4*i + 1],
                                ptr[4*i + 2],
                                ptr[4*i + 3]);
    }
    return segments;
}

/**
 * @brief Convert Segments -> NumPy array (N x 4, dtype=float32).
 * @param segments The C++ vector of segments.
 * @return A pybind11 array with shape [N,4].
 */
static py::array_t<float> segmentsToNdarray(const Segments &segments)
{
    const size_t n = segments.size();
    // Create a new NumPy array of shape (n, 4)
    py::array_t<float> arr({static_cast<py::ssize_t>(n), 4l});
    py::buffer_info buf = arr.request();
    float *ptr = static_cast<float*>(buf.ptr);

    for (size_t i = 0; i < n; ++i) {
        ptr[4*i + 0] = segments[i][0];
        ptr[4*i + 1] = segments[i][1];
        ptr[4*i + 2] = segments[i][2];
        ptr[4*i + 3] = segments[i][3];
    }
    return arr;
}

//------------------------------------------------------------------------------
// PYBIND11 MODULE DEFINITION
//------------------------------------------------------------------------------

PYBIND11_MODULE(_pyfsg, m) {
    m.doc() = R"pbdoc(
        Python bindings for GreedyMerger via pybind11
        ---------------------------------------------

        .. currentmodule:: pyfsg

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    py::class_<GreedyMerger>(m, "GreedyMerger")
        // Constructor from width, height
        .def(py::init([](int width, int height) {
            return std::make_unique<GreedyMerger>(cv::Size(width, height));
        }),
        py::arg("width") = 800,
        py::arg("height") = 480,
        "Construct a GreedyMerger for an image of size (width, height).")

        // setImageSize
        .def("setImageSize",
             [](GreedyMerger &self, int width, int height) {
                 self.setImageSize(cv::Size(width, height));
             },
             py::arg("width"), py::arg("height"),
             "Change the internal image size used by the merger.")

        // mergeSegments
        //
        //  In C++: void mergeSegments(const Segments &original, Segments &merged, SegmentClusters &clusters)
        //  We'll accept a NumPy array for `original` and return two things:
        //   1) a NumPy array (N x 4) for `merged`
        //   2) a Python list of lists of int for `clusters`
        .def("mergeSegments",
             [](GreedyMerger &self, const py::array_t<float> &arr) {
                 // Convert from NumPy => Segments
                 Segments original = ndarrayToSegments(arr);

                 // Prepare outputs
                 Segments merged;
                 SegmentClusters clusters;

                 // Call the actual C++ method
                 self.mergeSegments(original, merged, clusters);

                 // Convert merged back to a NumPy array
                 py::array_t<float> mergedArr = segmentsToNdarray(merged);

                 // Return a (mergedArr, clusters) tuple
                 return py::make_tuple(mergedArr, clusters);
             },
             py::arg("segments"),
             "Merge input line segments (Nx4 array) that belong to the same line. "
             "Returns (merged_segments, segment_clusters).")

        // getOrientationHistogram (static)
        .def_static("getOrientationHistogram",
                    [](const py::array_t<float> &arr, int bins) {
                        Segments segs = ndarrayToSegments(arr);
                        auto clusters = GreedyMerger::getOrientationHistogram(segs, bins);
                        return clusters;  // automatically converted to Python list-of-lists
                    },
                    py::arg("segments"), py::arg("bins") = 90,
                    "Build an orientation histogram from an Nx4 array of segments. "
                    "Returns a list of lists of indices (SegmentClusters).")

        // partialSortByLength (static)
        .def_static("partialSortByLength",
                    [](const py::array_t<float> &arr, int bins, int width, int height) {
                        Segments segs = ndarrayToSegments(arr);
                        auto sortedIndices = GreedyMerger::partialSortByLength(
                            segs, bins, cv::Size(width, height));
                        return sortedIndices; // automatically converted to Python list of int
                    },
                    py::arg("segments"), py::arg("bins"),
                    py::arg("width"), py::arg("height"),
                    "Sort Nx4 segments by descending length. Returns list of sorted indices.")

        // getTangentLineEqs (static)
        .def_static("getTangentLineEqs",
                    [](const py::array_t<float> &arr, float radius) {
                        // Expect a single segment, i.e. shape (1,4) or something similar
                        // but we’ll just read the first row for demonstration
                        Segments segs = ndarrayToSegments(arr);
                        if (segs.empty()) {
                            throw std::runtime_error("Expected at least 1 segment in Nx4 array.");
                        }
                        // We'll just use the first one
                        auto eqs = GreedyMerger::getTangentLineEqs(segs[0], radius);
                        // eqs.first, eqs.second are cv::Vec3f => (a,b,c)
                        py::tuple line1 = py::make_tuple(eqs.first[0], eqs.first[1], eqs.first[2]);
                        py::tuple line2 = py::make_tuple(eqs.second[0], eqs.second[1], eqs.second[2]);
                        return py::make_tuple(line1, line2);
                    },
                    py::arg("segment"), py::arg("radius"),
                    "Given a single segment (1x4 array) and a radius, returns 2 lines in (a,b,c) form.");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
        m.attr("__version__") = "dev";
#endif
}
