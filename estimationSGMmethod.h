#ifndef ESTIMATIONSGMMETHOD
#define ESTIMATIONSGMMETHOD

#include <iostream>
#include <vector>
#include <array>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "usedStructnFunctions.h"
#include "smallSupportFunctions.h"
#include "onGridOperations.h"
#include "matchingCostFunctions.h"

void getEstimationSGMbased(const cv::Mat& lookupTable, const cv::Mat& xydMatrix, const cv::Mat& XYZMatrix, const cv::Mat& badPoints, const cv::Mat& left_image, const cv::Mat& right_image, const cv::Mat& camera_matrix_left, const cv::Mat& camera_matrix_right, const std::vector<TotalDerivatives>& derivative_vector, const int& sampl_factor, const float& baseline, const float& doffs, cv::Mat& estimation_disparity, cv::Mat& estimation_X, cv::Mat& estimation_Y, cv::Mat& estimation_Z);


// try to develop a method that calculates the best matching using SGM method
// the method will work pixel-wise for both edges and no edges case in order to have consistency with the matrix
//void normalSGMEstimation(const cv::Mat& lookupTable, const cv::Mat& xyd_data, const cv::Mat& XYZ_data, const cv::Mat& left_image, const cv::Mat& right_image, const int& gap, cv::Mat& disparity_matrix);

#endif // !ESTIMATIONSGMMETHOD
