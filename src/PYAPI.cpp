#include <pybind11/pybind11.h>
#include <opencv2/opencv.hpp>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    cv::Vec2f v_i(i, 0);
    cv::Vec2f v_j(j, 0);
    cv::Vec2f sum = v_i + v_j;
    std::cout << "Sum: " << sum << std::endl;
    return sum(0);
}

namespace py = pybind11;

PYBIND11_MODULE(_pyfsg, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: pyfsg

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

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
