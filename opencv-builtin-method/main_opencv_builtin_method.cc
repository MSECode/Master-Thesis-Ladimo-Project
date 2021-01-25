#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

std::string DATASETPATHLADIMO = std::string(getenv("DATASET_LADIMO"));
std::string DATASETPATHMIDDLEBURY = std::string(getenv("DATASET_MIDDLEBURY"));
std::string BASEPATH = std::string(getenv("SGM_BASE_DIR"));
std::string TESTPATH = std::string(getenv("OPENCV_SGM_TEST_PATH"));
std::string FILESPATH = DATASETPATHMIDDLEBURY + "Couch-perfect/";

int main(int argc, char** argv) {

    // BUILT-IN SGM METHOD OPENCV 
    cv::Mat left_test_image = cv::imread(FILESPATH + "im0.png", cv::IMREAD_GRAYSCALE);
    cv::Mat right_test_image = cv::imread(FILESPATH + "im1.png", cv::IMREAD_GRAYSCALE);

    //left_test_image.convertTo(left_8bit, CV_8U, 255.0 / 1023.0);
    //right_test_image.convertTo(right_8bit, CV_8U, 255.0 / 1023.0);


    cv::namedWindow("8 bit left input image", cv::WINDOW_NORMAL);
    cv::imshow("8 bit left input image", left_test_image);
    cv::namedWindow("8 bit right input image", cv::WINDOW_NORMAL);
    cv::imshow("8 bit right input image", right_test_image);

    cv::waitKey(0);
    cv::destroyAllWindows();


    cv::Mat output_baseline_disparity = cv::Mat::zeros(left_test_image.size(), CV_16S);
    cv::Ptr<cv::StereoSGBM> stereo_sgm_test = cv::StereoSGBM::create(0, 16, 5);
    int number_of_disparities = 0;
    int window_size = 11;
    //number_of_disparities = number_of_disparities > 0 ? number_of_disparities : ((left_8bit.cols / 8) + 15) & -16;
    number_of_disparities = 320;
    int sgm_window_size = window_size > 0 ? window_size : 1;
    stereo_sgm_test->setMinDisparity(64);
    stereo_sgm_test->setPreFilterCap(21);
    stereo_sgm_test->setBlockSize(sgm_window_size);
    stereo_sgm_test->setP1(8 * sgm_window_size * sgm_window_size);
    stereo_sgm_test->setP2(32 * sgm_window_size * sgm_window_size);
    stereo_sgm_test->setNumDisparities(number_of_disparities);
    stereo_sgm_test->setUniquenessRatio(10);
    stereo_sgm_test->setSpeckleWindowSize(100);
    stereo_sgm_test->setSpeckleRange(2);
    stereo_sgm_test->setDisp12MaxDiff(1);
    stereo_sgm_test->setMode(cv::StereoSGBM::MODE_HH4);

    stereo_sgm_test->compute(left_test_image, right_test_image, output_baseline_disparity);

    cv::Mat disp_8bit_format;
    output_baseline_disparity.convertTo(disp_8bit_format, CV_8U, 255 / (number_of_disparities * 16.));

    cv::namedWindow("disparity output", cv::WINDOW_NORMAL);
    cv::imshow("disparity output", disp_8bit_format);
    cv::waitKey(0);
    cv::destroyAllWindows();

    //cv::imwrite(FILESPATH + "opencv-built-in-sgbm/output_disparity_8bit_couch_01.png", disp_8bit_format);

    return EXIT_SUCCESS;
}