#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdlib.h>
#include <math.h>

#include "ioFunctions.h"
#include "onGridOperations.h"
#include "onGridOPerations_tester.h"
#include "getWrongPoints.h"
#include "matchingCostFunctions.h"
#include "smallSupportFunctions.h"
#include "testFunctions.h"
#include "estimationSGMmethod.h"

int main(int argc, char** argv) {

    // IMAGE LOADING      

    // Read the left and right images 
    std::string datasetPath = std::string(getenv("DATASET_MIDDLEBURY"));
    cv::Mat im_left = cv::imread(datasetPath + "Motorcycle-perfect/im0.png", cv::IMREAD_GRAYSCALE);
    cv::Mat im_right = cv::imread(datasetPath + "Motorcycle-perfect/im1.png", cv::IMREAD_GRAYSCALE);

    // Check fro failure
    if (im_left.empty() || im_right.empty())
    {
        std::cout << "Could not open or find one or both images" << std::endl;
        std::cin.get();
        return -1;
    }

    std::string windowLeft = "Left Image";
    cv::namedWindow(windowLeft, cv::WINDOW_NORMAL);
    std::string windowRight = "Right Image";
    cv::namedWindow(windowRight, cv::WINDOW_NORMAL);
    imshow(windowLeft, im_left);
    imshow(windowRight, im_right);

    // Load disparity images from binary file
    std::string basePath = std::string(getenv("SGM_BASE_DIR"));

    std::vector <float> leftDispArray = io::readBinary(basePath + "disparity_left.bin");
    std::vector <float> rightDispArray = io::readBinary(basePath + "disparity_right.bin");

    int height = im_left.rows;
    int width = im_left.cols;
    cv::Mat leftDispMat = cv::Mat(width, height, CV_32FC1, leftDispArray.data()).t();
    cv::Mat rightDispMat = cv::Mat(width, height, CV_32FC1, rightDispArray.data()).t();

    double min, max;
    cv::minMaxLoc(leftDispMat, &min, &max);
    cv::Mat normalizedMat_l = leftDispMat / float(max);
    cv::Mat normalizedMat_r = rightDispMat / float(max);

    std::string windowDisp_l = "Left Disp";
    std::string windowDisp_r = "Right Disp";

    cv::namedWindow(windowDisp_l, cv::WINDOW_NORMAL);
    cv::namedWindow(windowDisp_r, cv::WINDOW_NORMAL);
    imshow(windowDisp_l, normalizedMat_l);
    imshow(windowDisp_r, normalizedMat_r);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // PARAMETERS 
    // Define parameters that we are going to use

    // Camera parameters --> specific for the imported image --> remember to change if import different image from the Dataset
    std::array<std::array<float, 3>, 3> cam0 = { { {3979.911, 0, 1244.772}, {0, 3979.911, 1019.507}, {0, 0, 1} } };
    std::array<std::array<float, 3>, 3> cam1 = { { {3979.911, 0, 1369.115}, {0, 3979.911, 1019.507}, {0, 0, 1} } };
    float doffs = 124.343;
    float baseline = 193.001;
    float focal = cam0[0][0];
    float cx = cam0[0][2];
    float cy = cam0[1][2];

    cv::Mat camera_matrix_0 = cv::Mat(3, 3, CV_32F, cam0.data());
    cv::Mat camera_matrix_1 = cv::Mat(3, 3, CV_32F, cam1.data());

    // Fictititous Ladimo grid parameters
    int gap = 25;
    int xPoints = (height - gap) / gap;
    int yPoints = (width - gap) / gap;

    // Convert images from grayscale to floating point numbers (easy to use them later)
    im_left.convertTo(im_left, CV_32F);
    im_right.convertTo(im_right, CV_32F);

    // Image limits necessary for the estimation algorithm with simple raw difference for best estimation
    std::vector<int> im_limits = { im_left.cols, im_left.rows };

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // FICTITIOUS LADIMO GRID

    cv::Mat ladimoGridPoints_l = cv::Mat(xPoints * yPoints, 3, CV_32F);
    cv::Mat ladimoGridPoints_r = cv::Mat(xPoints * yPoints, 3, CV_32F);

    float c = 0;    // index for iterating over all the grid points as float
    int n = 0;      // index for iterating over all the grid points as integer
    int k_y = 0;    // index for the indexes of the y grid points
    int h_y = gap;  // index for the pixel y coords of the grid points


    for (float i = float(gap); i < float(height) - float(gap) + 1; i = float(i + gap))
    {
        int k_x = 0;     // index for the indexes of the x grid points
        int h_x = gap;   // index for the pixel x coords of the grid points
        for (float j = float(gap); j < float(width) - float(gap); j = float(j + gap))
        {
            ladimoGridPoints_l.at<float>(c, 0) = j;                                                                                       // x value of the left grid point
            ladimoGridPoints_l.at<float>(c, 1) = i;                                                                                       // y value of the left grid point
            ladimoGridPoints_l.at<float>(c, 2) = leftDispMat.at<float>(i, j);                                                             // disparity value of the left grid point

            ladimoGridPoints_r.at<float>(c, 0) = ladimoGridPoints_l.at<float>(c, 0) - leftDispMat.at<float>(i, j);                        // x value of the right grid point
            ladimoGridPoints_r.at<float>(c, 1) = i;                                                                                       // y value of the right grid point
            if (ladimoGridPoints_r.at<float>(c, 0) > 0)
            {
                ladimoGridPoints_r.at<float>(c, 2) = rightDispMat.at<float>(i, ladimoGridPoints_r.at<float>(c, 0));                       // disparity value of the right grid point
            }
            else
            {
                ladimoGridPoints_r.at<float>(c, 2) = -1.f;
            }
            ++k_x;
            h_x += gap;
            ++n;
            ++c;
        }
        ++k_y;
        h_y += gap;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // CONVERTING THE POINT TO SPACE POSITION 

    cv::Mat ladimoSpacePoints_l = cv::Mat(xPoints * yPoints, 3, CV_32F);
    cv::Mat ladimoSpacePoints_r = cv::Mat(xPoints * yPoints, 3, CV_32F);

    for (size_t i = 0; i < ladimoGridPoints_l.rows; i++)
    {
        ladimoSpacePoints_l.at<float>(i, 2) = (baseline * camera_matrix_0.at<float>(0, 0)) / (ladimoGridPoints_l.at<float>(i, 2) + doffs);
        ladimoSpacePoints_l.at<float>(i, 0) = (ladimoGridPoints_l.at<float>(i, 0) - camera_matrix_0.at<float>(0, 2)) * ladimoSpacePoints_l.at<float>(i, 2) / camera_matrix_0.at<float>(0, 0);
        ladimoSpacePoints_l.at<float>(i, 1) = (ladimoGridPoints_l.at<float>(i, 1) - camera_matrix_0.at<float>(1, 2)) * ladimoSpacePoints_l.at<float>(i, 2) / camera_matrix_0.at<float>(1, 1);

        ladimoSpacePoints_r.at<float>(i, 2) = (baseline * camera_matrix_1.at<float>(0, 0)) / (ladimoGridPoints_r.at<float>(i, 2) + doffs);
        ladimoSpacePoints_r.at<float>(i, 0) = (ladimoGridPoints_r.at<float>(i, 0) - camera_matrix_1.at<float>(0, 2)) * ladimoSpacePoints_r.at<float>(i, 2) / camera_matrix_1.at<float>(0, 0) + baseline;
        ladimoSpacePoints_r.at<float>(i, 1) = (ladimoGridPoints_r.at<float>(i, 1) - camera_matrix_1.at<float>(1, 2)) * ladimoSpacePoints_r.at<float>(i, 2) / camera_matrix_1.at<float>(1, 1);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   // LOOKUP TABLE
   // It contains the indexes to the specific row of the LadimoGridPoints for each one of the points in the grid
   // Used to recover XYZ coordinates of the grid points faster

    // Get the lookup table

    LookupTable lookup_table;
    lookup_table.numPoint_x = xPoints;
    lookup_table.numPoint_y = yPoints;

    cv::Mat lookupTable = lookup_table.lookupMatrix(ladimoGridPoints_l.rows);

    // Get the bad points (very big z-value)
    GetWrongPoints get_wrong_points_l;
    GetWrongPoints get_wrong_points_r;

    cv::Mat badDepths_l = get_wrong_points_l.getWrongPoints(ladimoSpacePoints_l);
    cv::Mat badDepths_r = get_wrong_points_r.getWrongPoints(ladimoSpacePoints_r);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   // CALCULATE THE DERIVATIVES OVER THE GRID POINTS

    cv::Mat fullMatrix;
    fullMatrix = getFullMatrix(baseline, focal, doffs, ladimoSpacePoints_l);

    // Derivative vector with XYZ + disparity information
    std::vector<TotalDerivatives> totVector;
    calculateDerivatives(fullMatrix, lookupTable, totVector);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // DISPARITY ESTIMATION

    // Get the estimations inside 4 corner point of the lookuptable
    int sampl_factor = 4;                                             // number of samples for each subrow of the small window
    int cols = (sampl_factor + 1) * (sampl_factor + 1);
    int rows = (lookupTable.rows - 1) * (lookupTable.cols - 1);
    cv::Mat estimated_disparity_matrix_raw_l = cv::Mat(rows, cols, CV_32F);
    cv::Mat estimated_Z = cv::Mat(rows, cols, CV_32F);


    // ESTIMATION FOR ONLY THE DISPARITY
    // Not overloaded function --> it  returns only the disparity values for the estimated points using absolute diffrence as cost for best estimation
    getEstimations_testing(fullMatrix, lookupTable, im_left, im_right, totVector, camera_matrix_0, badDepths_l, estimated_disparity_matrix_raw_l, estimated_Z);

    // Create a disparity image using the disparities from the grid of points and the estimated ones 
    // and check the reliability of the estimations
    int totrows = (lookupTable.rows - 1) * sampl_factor + 1;
    int totcols = (lookupTable.cols - 1) * sampl_factor + 1;
    cv::Mat guessedDisparityImage_from_raw = cv::Mat(totrows, totcols, CV_32F);
    cv::Mat guessed_Z = cv::Mat(totrows, totcols, CV_32F);


    getGuessedDisparity_testing(fullMatrix, lookupTable, estimated_disparity_matrix_raw_l, estimated_Z, sampl_factor, guessedDisparityImage_from_raw, guessed_Z);

    // Normalize the disparity values so that they are between 0 and 1 so that it is possible to visualize them nicely
    double min_raw, max_raw;
    cv::minMaxLoc(guessedDisparityImage_from_raw, &min_raw, &max_raw);
    cv::Mat normalizedDispMatRaw_l = guessedDisparityImage_from_raw / float(max_raw);

    std::string windowGuessRaw_l = "Guessed Left Disp Raw";

    cv::namedWindow(windowGuessRaw_l, cv::WINDOW_NORMAL);;
    imshow(windowGuessRaw_l, normalizedDispMatRaw_l);

    cv::waitKey(0);
    cv::destroyAllWindows();

    // TEST SOME PARTS OF THE IMAGE TO SEE HOW THE ALGORITHMS WORK
    /*
    int row_pixel = 1100;
    int col_pixel = 2450;

    int look_up_r = row_pixel / gap;
    int look_up_c = col_pixel / gap;

    // Visualize the patch
    cv::Mat rgb_left_image = cv::imread(datasetPath + "Motorcycle-perfect/im0.png");
    cv::Rect patch = cv::Rect(col_pixel, row_pixel, gap, gap);
    cv::rectangle(rgb_left_image, patch, cv::Scalar(0, 0, 0), 2);

    cv::namedWindow("image_with_patch", cv::WINDOW_NORMAL);
    imshow("image_with_patch", rgb_left_image);

    cv::imwrite(basePath + "Images_folder/image_with_patch.png", rgb_left_image);

    cv::namedWindow("visualize patch", cv::WINDOW_NORMAL);
    imshow("visualize patch", rgb_left_image(patch));

    cv::waitKey(0);
    cv::destroyAllWindows();

    estimatedDisparityTest(lookupTable, fullMatrix, totVector, look_up_r, look_up_c, sampl_factor);
   

    guessedDisparityTest(row_pixel, col_pixel, leftDispMat, guessedDisparityImage_from_raw, sampl_factor, gap);
    
    int row_d = gap * (lookupTable.rows) + 1;
    int col_d = gap * (lookupTable.cols) + 1;
    cv::Mat disparity_matrix = cv::Mat(row_d, col_d, CV_32F, cv::Scalar(0));

    
    normalSGMEstimation(lookupTable, ladimoGridPoints_l, fullMatrix, im_left, im_right, gap, disparity_matrix);
    
    double min_d, max_d;
    cv::minMaxLoc(disparity_matrix, &min_d, &max_d);
    cv::Mat normalized_disp = disparity_matrix / float(max_d);

    cv::namedWindow("SGM result", cv::WINDOW_NORMAL);
    imshow("SGM result", normalized_disp);
    
    cv::waitKey(0);
    cv::destroyAllWindows();
    

    std::string test_path = basePath + "test_files/";

    sgmTest(lookupTable, ladimoGridPoints_l, fullMatrix, im_left, im_right, gap, look_up_r, look_up_c, test_path);
     
    
    int kernel_size = 3;
    
    cv::Mat destination_mat_median = cv::Mat::zeros(disparity_matrix.size(), disparity_matrix.type());
    cv::Mat destination_mat_bilateral = cv::Mat::zeros(disparity_matrix.size(), disparity_matrix.type());
    
    /// Applying Median blur
    cv::medianBlur(disparity_matrix, destination_mat_median, kernel_size);
    
    double min_mb, max_mb;
    cv::minMaxLoc(destination_mat_median, &min_mb, &max_mb);
    cv::Mat destination_mat_median_n = destination_mat_median / float(max_mb);

    /// Applying Bilateral Filter
    cv::bilateralFilter(disparity_matrix, destination_mat_bilateral, kernel_size, kernel_size * 2, kernel_size / 2);
    double min_bl, max_bl;
    cv::minMaxLoc(disparity_matrix, &min_bl, &max_bl);
    cv::Mat destination_mat_bilateral_n = destination_mat_bilateral / float(max_bl);


    cv::namedWindow("median filter", cv::WINDOW_NORMAL);
    imshow("median filter", destination_mat_median_n);

    cv::namedWindow("bilateral filter", cv::WINDOW_NORMAL);
    imshow("bilateral filter", destination_mat_bilateral_n);

    cv::waitKey(0);
    cv::destroyAllWindows();
    */

    std::string txtFilePath = std::string(getenv("PATH_TXT_FILE_LADIMO_PRG"));
    io::writeMatrixToFile(normalizedDispMatRaw_l, txtFilePath + "disparity_normal.txt");

    int row_pixel = 136;
    int col_pixel = 348;

    // Visualize the patch
    cv::Rect patch = cv::Rect(col_pixel, row_pixel, 15, 15);
    cv::rectangle(normalizedDispMatRaw_l, patch, cv::Scalar(0, 0, 0), 1);

    cv::namedWindow("image_with_patch", cv::WINDOW_NORMAL);
    imshow("image_with_patch", normalizedDispMatRaw_l);


    cv::namedWindow("visualize patch", cv::WINDOW_NORMAL);
    imshow("visualize patch", normalizedDispMatRaw_l(patch));

    cv::waitKey(0);
    cv::destroyAllWindows();

    std::string test_path = basePath + "test_files/";
    io::writeMatrixToFile(normalizedDispMatRaw_l(patch), test_path + "disparity_test_patch.txt");
    return EXIT_SUCCESS;
}