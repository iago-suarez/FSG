#ifndef FSG_LSDOPENCV_H_
#define FSG_LSDOPENCV_H_

#include <opencv2/opencv.hpp>
#include "Utils.h"

namespace upm {
class LsdOpenCV {
 public:
  /**
   * Create a LineSegmentDetectorImpl object. Specifying scale, number of subdivisions for the image,
   * should the lines be refined and other constants as follows:
   *
   * @param _refine       How should the lines found be refined?
   *                      LSD_REFINE_NONE - No refinement applied.
   *                      LSD_REFINE_STD  - Standard refinement is applied. E.g. breaking arches into
   *                      smaller line approximations.
   *                      LSD_REFINE_ADV  - Advanced refinement. Number of false alarms is calculated,
   *                                    lines are refined through increase of precision, decrement in size, etc.
   * @param scale        The scale of the image that will be used to find the lines. Range (0..1].
   * @param sigma_scale  Sigma for Gaussian filter is computed as sigma = _sigma_scale/_scale.
   * @param quant        Bound to the quantization error on the gradient norm.
   * @param ang_th       Gradient angle tolerance in degrees.
   * @param log_eps      Detection threshold: -log10(NFA) > _log_eps
   * @param density_th   Minimal density of aligned region points in rectangle.
   * @param n_bins       Number of bins in pseudo-ordering of gradient modulus.
   */
  explicit LsdOpenCV(int refine = cv::LSD_REFINE_STD,
                     double scale = 0.8,
                     double sigma_scale = 0.6,
                     double quant = 2.0,
                     double ang_th = 22.5,
                     double log_eps = 0,
                     double density_th = 0.7,
                     int n_bins = 1024);

  /**
   * Detect lines in the input image.
   *
   * @param _image    A grayscale(CV_8UC1) input image.
   *                  If only a roi needs to be selected, use
   *                  lsd_ptr->detect(image(roi), ..., lines);
   *                  lines += Scalar(roi.x, roi.y, roi.x, roi.y);
   * @param _lines    Return: A vector of Vec4i or Vec4f elements specifying the beginning and ending point of a line.
   *                          Where Vec4i/Vec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end.
   *                          Returned lines are strictly oriented depending on the gradient.
   * @param width     Return: Vector of widths of the regions, where the lines are found. E.g. Width of line.
   * @param prec      Return: Vector of precisions with which the lines are found.
   * @param nfa       Return: Vector containing number of false alarms in the line region, with precision of 10%.
   *                          The bigger the value, logarithmically better the detection.
   *                              * -1 corresponds to 10 mean false alarms
   *                              * 0 corresponds to 1 mean false alarm
   *                              * 1 corresponds to 0.1 mean false alarms
   *                          This vector will be calculated _only_ when the objects type is REFINE_ADV
   */
  void detect(cv::InputArray _image,
              cv::OutputArray _lines,
              cv::OutputArray width = cv::noArray(),
              cv::OutputArray prec = cv::noArray(),
              cv::OutputArray nfa = cv::noArray());

 protected:
  cv::Mat image;
  cv::Mat scaled_image;
  cv::Mat_<double> angles;     // in rads
  cv::Mat_<double> modgrad;
  cv::Mat_<uchar> used;

  int img_width;
  int img_height;
  double LOG_NT;

  bool w_needed;
  bool p_needed;
  bool n_needed;

  const double SCALE;
  const int doRefine;
  const double SIGMA_SCALE;
  const double QUANT;
  const double ANG_TH;
  const double LOG_EPS;
  const double DENSITY_TH;
  const int N_BINS;

  struct RegionPoint {
    int x;
    int y;
    uchar *used;
    double angle;
    double modgrad;
  };

  struct coorlist {
    cv::Point2i p;
    struct coorlist *next;
  };

  std::vector<coorlist> list;

  struct rect {
    double x1, y1, x2, y2;    // first and second point of the line segment
    double width;             // rectangle width
    double x, y;              // center of the rectangle
    double theta;             // angle
    double dx, dy;             // (dx,dy) is vector oriented as the line segment
    double prec;              // tolerance angle
    double p;                 // probability of a point with angle within 'prec'
  };

  // LsdOpenCV &operator=(const LsdOpenCV &);  // to quiet MSVC

  /**
   * Detect lines in the whole input image.
   *
   * @param lines         Return: A vector of Vec4f elements specifying the beginning and ending point of a line.
   *                              Where Vec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end.
   *                              Returned lines are strictly oriented depending on the gradient.
   * @param widths        Return: Vector of widths of the regions, where the lines are found. E.g. Width of line.
   * @param precisions    Return: Vector of precisions with which the lines are found.
   * @param nfas          Return: Vector containing number of false alarms in the line region, with precision of 10%.
   *                              The bigger the value, logarithmically better the detection.
   *                                  * -1 corresponds to 10 mean false alarms
   *                                  * 0 corresponds to 1 mean false alarm
   *                                  * 1 corresponds to 0.1 mean false alarms
   */
  virtual void
  detectSegments(Segments &lines,  // NOLINT
                 std::vector<double> &widths,  // NOLINT
                 std::vector<double> &precisions,  // NOLINT
                 std::vector<double> &nfas);  // NOLINT

  /**
   * Finds the angles and the gradients of the image. Generates a list of pseudo ordered points.
   *
   * @param threshold The minimum value of the angle that is considered defined, otherwise NOTDEF
   * @param n_bins    The number of bins with which gradients are ordered by, using bucket sort.
   * @param list      Return: Vector of coordinate points that are pseudo ordered by magnitude.
   *                  Pixels would be ordered by norm value, up to a precision given by max_grad/n_bins.
   */
  virtual void
  computeGradAndCandidates(const double &threshold, const unsigned int &n_bins);  // NOLINT

  /**
   * Grow a region starting from point s with a defined precision,
   * returning the containing points size and the angle of the gradients.
   *
   * @param s         Starting point for the region.
   * @param reg       Return: Vector of points, that are part of the region
   * @param reg_angle Return: The mean angle of the region.
   * @param prec      The precision by which each region angle should be aligned to the mean.
   */
  virtual void
  regionGrow(const cv::Point2i &s,
             std::vector<RegionPoint> &reg,  // NOLINT
             double &reg_angle,  // NOLINT
             const double &prec);

  /**
   * Finds the bounding rotated rectangle of a region.
   *
   * @param reg       The region of points, from which the rectangle to be constructed from.
   * @param reg_angle The mean angle of the region.
   * @param prec      The precision by which points were found.
   * @param p         Probability of a point with angle within 'prec'.
   * @param rec       Return: The generated rectangle.
   */
  void region2rect(const std::vector<RegionPoint> &reg,
                   double reg_angle,
                   double prec,
                   double p,
                   rect &rec) const;  // NOLINT

  /**
   * Compute region's angle as the principal inertia axis of the region.
   * @return          Regions angle.
   */
  double
  getTheta(const std::vector<RegionPoint> &reg,
           const double &x,
           const double &y,
           const double &reg_angle,
           const double &prec) const;

  /**
   * An estimation of the angle tolerance is performed by the standard deviation of the angle at points
   * near the region's starting point. Then, a new region is grown starting from the same point, but using the
   * estimated angle tolerance. If this fails to produce a rectangle with the right density of region points,
   * 'reduceRegionRadius' is called to try to satisfy this condition.
   */
  bool
  refine(std::vector<RegionPoint> &reg,  // NOLINT
         double reg_angle,
         double prec,
         double p,
         rect &rec,  // NOLINT
         double density_th);

  /**
   * Reduce the region size, by elimination the points far from the starting point, until that leads to
   * rectangle with the right density of region points or to discard the region if too small.
   */
  bool reduceRegionRadius(std::vector<RegionPoint> &reg,  // NOLINT
                          double reg_angle,
                          double prec,
                          double p,
                          rect &rec,  // NOLINT
                          double density,
                          double density_th);

  /**
   * Try some rectangles variations to improve NFA value. Only if the rectangle is not meaningful (i.e., log_nfa <= log_eps).
   * @return      The new NFA value.
   */
  double rectImprove(rect &rec) const;  // NOLINT

  /**
   * Calculates the number of correctly aligned points within the rectangle.
   * @return      The new NFA value.
   */
  double rectNfa(const rect &rec) const;

  /**
   * Computes the NFA values based on the total number of points, points that agree.
   * n, k, p are the binomial parameters.
   * @return      The new NFA value.
   */
  double nfa(const int &n, const int &k, const double &p) const;

  /**
   * Is the point at place 'address' aligned to angle theta, up to precision 'prec'?
   * @return      Whether the point is aligned.
   */
  bool isAligned(int x, int y, const double &theta, const double &prec) const;
};

}  // namespace upm
#endif  // FSG_LSDOPENCV_H_
