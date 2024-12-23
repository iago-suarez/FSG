#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>        // for py::array_t
#include <opencv2/core.hpp>

#include "GreedyMerger.h"
#include "LsdOpenCV.h"

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
 * @brief Helper to convert a Python grayscale image (NumPy array) to cv::Mat(CV_8UC1).
 * @param imgArray  A 2D or 3D NumPy array of shape (H, W) or (H, W, 1).
 * @throw std::runtime_error if dtype not uint8 or shape mismatch.
 */
static cv::Mat numpyToGrayMat(const py::array_t<uint8_t> &imgArray) {
    py::buffer_info buf = imgArray.request();
    int ndims = buf.ndim;

    if (ndims < 2 || ndims > 3) {
        throw std::runtime_error("Expected 2D or 3D array for the grayscale image.");
    }
    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = (ndims == 2) ? 1 : buf.shape[2];

    if (channels != 1) {
        throw std::runtime_error("Expected single-channel (grayscale) image array.");
    }

    // Wrap in a cv::Mat without copying data (assuming row-major contiguous layout).
    cv::Mat mat(height, width, CV_8UC1, (unsigned char *) buf.ptr);
    return mat;
}

/**
 * @brief Convert a NumPy array (N x 4, dtype=float32 or float64) -> Segments.
 * @param arr Python array, must be 2D with shape [N,4].
 * @throws std::runtime_error if shape is wrong.
 */
static Segments ndarrayToSegments(const py::array_t<float> &arr) {
    // Request buffer info from NumPy array
    py::buffer_info buf = arr.request();

    if (buf.ndim != 2 || buf.shape[1] != 4) {
        throw std::runtime_error("Segments array must have shape (N,4)");
    }

    // Convert to Segments
    size_t n = buf.shape[0];
    float *ptr = static_cast<float *>(buf.ptr);

    Segments segments(n);
    for (size_t i = 0; i < n; ++i) {
        // each row: (x1, y1, x2, y2)
        segments[i] = cv::Vec4f(ptr[4 * i + 0],
                                ptr[4 * i + 1],
                                ptr[4 * i + 2],
                                ptr[4 * i + 3]);
    }
    return segments;
}

/**
 * @brief Convert Segments -> NumPy array (N x 4, dtype=float32).
 * @param segments The C++ vector of segments.
 * @return A pybind11 array with shape [N,4].
 */
static py::array_t<float> segmentsToNdarray(const Segments &segments) {
    const size_t n = segments.size();
    // Create a new NumPy array of shape (n, 4)
    py::array::ShapeContainer arr_shape{static_cast<py::ssize_t>(n), 4l};
    py::array_t<float> arr(arr_shape);
    py::buffer_info buf = arr.request();
    float *ptr = static_cast<float *>(buf.ptr);

    for (size_t i = 0; i < n; ++i) {
        ptr[4 * i + 0] = segments[i][0];
        ptr[4 * i + 1] = segments[i][1];
        ptr[4 * i + 2] = segments[i][2];
        ptr[4 * i + 3] = segments[i][3];
    }
    return arr;
}

/**
 * @brief Helpers for optional arrays (width, prec, nfa) each of shape (N,)
 */
static py::array_t<double> vectorToNdarray(const std::vector<double> &vals) {
    py::array_t<double> arr(vals.size());
    auto buf = arr.request();
    double *ptr = static_cast<double *>(buf.ptr);
    for (size_t i = 0; i < vals.size(); ++i) {
        ptr[i] = vals[i];
    }
    return arr;
}


/**
 * A free function to run LSD detection. Returns either a NumPy array (N,4)
 * or a dict with {"lines": Nx4, "width": Nx1, "prec": Nx1, "nfa": Nx1}.
 */
static py::object detectLinesOpencvLSD(
    const py::array_t<uint8_t> &imgArray,
    int refine,
    double scale,
    double sigma_scale,
    double quant,
    double ang_th,
    double log_eps,
    double density_th,
    int n_bins,
    bool return_width,
    bool return_prec,
    bool return_nfa) {
    // 1) Convert input image
    cv::Mat gray = numpyToGrayMat(imgArray);

    // 2) Construct an LSD detector with the given params
    LsdOpenCV lsd(refine, scale, sigma_scale, quant, ang_th, log_eps, density_th, n_bins);

    // 3) Prepare storage for lines and optionally the widths, prec, nfa
    Segments lines; // std::vector<cv::Vec4f>
    std::vector<double> widths, precs, nfas;

    // 4) We can call LsdOpenCV::detect(...) in 2 ways:
    //    - either the overload with only lines
    //    - or the overload with lines + width + prec + nfa
    if (!return_width && !return_prec && !return_nfa) {
        // simpler call
        lsd.detect(gray, lines);
    } else {
        // we want the extra arrays
        cv::Mat wMat, pMat, nMat; // Output arrays
        lsd.detect(gray, lines, wMat, pMat, nMat);

        if (return_width && !wMat.empty()) {
            widths.resize(wMat.total());
            wMat.copyTo(cv::Mat(widths));
        }
        if (return_prec && !pMat.empty()) {
            precs.resize(pMat.total());
            pMat.copyTo(cv::Mat(precs));
        }
        if (return_nfa && !nMat.empty()) {
            nfas.resize(nMat.total());
            nMat.copyTo(cv::Mat(nfas));
        }
    }

    // 5) Convert lines => Nx4 array
    py::array_t<float> linesArr = segmentsToNdarray(lines);

    // 6) Return
    if (!return_width && !return_prec && !return_nfa) {
        // Just return the Nx4 lines array
        return py::cast<py::object>(linesArr);
    }

    // Otherwise, return a dict
    py::dict result;
    result["lines"] = linesArr;

    if (return_width) {
        result["width"] = vectorToNdarray(widths);
    }
    if (return_prec) {
        result["prec"] = vectorToNdarray(precs);
    }
    if (return_nfa) {
        result["nfa"] = vectorToNdarray(nfas);
    }

    return result;
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
                            return clusters; // automatically converted to Python list-of-lists
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

    // -------------------------------------------------------------------------
    // 4) Define the FREE FUNCTION for LSD detection
    // -------------------------------------------------------------------------
    m.def("detectLinesOpencvLSD", &detectLinesOpencvLSD,
          py::arg("image"),
          py::arg("refine") = 1,
          py::arg("scale") = 0.8,
          py::arg("sigma_scale") = 0.6,
          py::arg("quant") = 2.0,
          py::arg("ang_th") = 22.5,
          py::arg("log_eps") = 0.0,
          py::arg("density_th") = 0.7,
          py::arg("n_bins") = 1024,
          py::arg("return_width") = false,
          py::arg("return_prec") = false,
          py::arg("return_nfa") = false,
          R"doc(
Run LSD detection in a single call.
Parameters:
    image        - grayscale uint8 array, shape (H,W) or (H,W,1)
    refine       - LSD refine mode (0=NONE, 1=STD, 2=ADV)
    scale        - LSD scale factor
    sigma_scale  - LSD sigma scale
    quant        - LSD quant
    ang_th       - LSD angle threshold in degrees
    log_eps      - LSD detection threshold
    density_th   - LSD minimal density
    n_bins       - LSD number of bins
    return_width - whether to return line widths
    return_prec  - whether to return line precisions
    return_nfa   - whether to return line NFA

Return:
    If return_width/prec/nfa = False => Nx4 float array of line segments (x1,y1,x2,y2).
    Else => dict with {
        'lines': Nx4 array,
        'width': Nx array (if return_width=True),
        'prec': Nx array  (if return_prec=True),
        'nfa': Nx array   (if return_nfa=True)
    }
)doc"
    );
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
        m.attr("__version__") = "dev";
#endif
}
