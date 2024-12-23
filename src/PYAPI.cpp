#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <opencv2/core.hpp>
#include "GreedyMerger.h"  // your library's header

namespace py = pybind11;
using namespace upm;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    cv::Vec2f v_i(i, 0);
    cv::Vec2f v_j(j, 0);
    cv::Vec2f sum = v_i + v_j;
    std::cout << "Sum: " << sum << std::endl;
    return sum(0);
}


// A small helper struct to hold the 4 floats from a "Segment".
struct PySegment {
    float x1, y1, x2, y2;

    PySegment() : x1(0.f), y1(0.f), x2(0.f), y2(0.f) {
    }

    PySegment(float x1_, float y1_, float x2_, float y2_)
        : x1(x1_), y1(y1_), x2(x2_), y2(y2_) {
    }

    explicit PySegment(const Segment &seg) {
        x1 = seg[0];
        y1 = seg[1];
        x2 = seg[2];
        y2 = seg[3];
    }

    [[nodiscard]] Segment toSegment() const {
        Segment s;
        s[0] = x1;
        s[1] = y1;
        s[2] = x2;
        s[3] = y2;
        return s;
    }
};

using PySegments = std::vector<PySegment>;

PYBIND11_MAKE_OPAQUE(std::vector<PySegment>);

PYBIND11_MAKE_OPAQUE(std::vector<std::vector<unsigned int>>);

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

    py::class_<PySegment>(m, "Segment")
            .def(py::init<>())
            .def(py::init<float, float, float, float>(),
                 py::arg("x1"), py::arg("y1"),
                 py::arg("x2"), py::arg("y2"))
            .def_readwrite("x1", &PySegment::x1)
            .def_readwrite("y1", &PySegment::y1)
            .def_readwrite("x2", &PySegment::x2)
            .def_readwrite("y2", &PySegment::y2);

    // Wrap vector of PySegment as "Segments"
    py::bind_vector<std::vector<PySegment> >(m, "Segments");

    // Wrap vector<vector<unsigned int>> as "SegmentClusters"
    py::bind_vector<std::vector<std::vector<unsigned int> > >(m, "SegmentClusters");

    py::class_<GreedyMerger>(m, "GreedyMerger")
            .def(py::init([](int width, int height) {
                     return std::make_unique<GreedyMerger>(cv::Size(width, height));
                 }),
                 py::arg("width") = 800, py::arg("height") = 480)
            .def("setImageSize",
                 [](GreedyMerger &self, int width, int height) {
                     self.setImageSize(cv::Size(width, height));
                 },
                 py::arg("width"), py::arg("height"))
            .def("mergeSegments",
                 [](GreedyMerger &self, const std::vector<PySegment> &pySegs) {
                     Segments in;
                     in.reserve(pySegs.size());
                     for (auto &ps: pySegs) in.push_back(ps.toSegment());

                     Segments out;
                     SegmentClusters clusters;
                     self.mergeSegments(in, out, clusters);

                     // Convert output segments back to Python-friendly structure
                     std::vector<PySegment> pyOut;
                     pyOut.reserve(out.size());
                     for (auto &seg: out) {
                         pyOut.emplace_back(seg);
                     }
                     // Return them as (mergedSegments, clusters)
                     return std::make_pair(pyOut, clusters);
                 },
                 py::arg("segments"))
            .def_static("getOrientationHistogram",
                        [](const std::vector<PySegment> &pySegs, int bins) {
                            Segments in;
                            in.reserve(pySegs.size());
                            for (auto &ps: pySegs) in.push_back(ps.toSegment());
                            return GreedyMerger::getOrientationHistogram(in, bins);
                        },
                        py::arg("segments"), py::arg("bins") = 90)
            .def_static("partialSortByLength",
                        [](const std::vector<PySegment> &pySegs,
                           int bins, int width, int height) {
                            Segments in;
                            in.reserve(pySegs.size());
                            for (auto &ps: pySegs) in.push_back(ps.toSegment());
                            return GreedyMerger::partialSortByLength(
                                in, bins, cv::Size(width, height)
                            );
                        },
                        py::arg("segments"), py::arg("bins"),
                        py::arg("width"), py::arg("height"))
            .def_static("getTangentLineEqs",
                        [](const PySegment &pySeg, float radius) {
                            auto eq = GreedyMerger::getTangentLineEqs(pySeg.toSegment(), radius);
                            py::tuple l1 = py::make_tuple(eq.first[0], eq.first[1], eq.first[2]);
                            py::tuple l2 = py::make_tuple(eq.second[0], eq.second[1], eq.second[2]);
                            return py::make_tuple(l1, l2);
                        },
                        py::arg("segment"), py::arg("radius"));

    m.def("add", &add, R"pbdoc(
            Add two numbers

            Some other explanation about the add function.
        )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
            Subtract two numbers

            Some other explanation about the subtract function.
        )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
        m.attr("__version__") = "dev";
#endif
}
