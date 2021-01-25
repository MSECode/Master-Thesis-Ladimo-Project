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

#include "ioFunctions.h"
#include "usedStructnFunctions.h"
#include "smallSupportFunctions.h"
#include "preGridOPerations.h"
#include "onGridOPerations.h"
#include "getWrongPoints.h"
#include "matchingCostFunctions.h"
#include "estimationSGMmethod.h"


// Global variables (Path related to data and storage, which are going to be used all over the code)
std::string DATASETPATH = std::string(getenv("DATASET_MIDDLEBURY"));         // Datasets folder
std::string BASEPATH = std::string(getenv("SGM_BASE_DIR"));                  // Genaral directory of the project
std::string TXTFILEPATH = std::string(getenv("PATH_TXT_FILE_LADIMO_PRG"));   // Path where to save txt files

// Calibration file of the images that we are going to use
std::string CALIBRATIONFILE = DATASETPATH + "Motorcycle-perfect/calib.txt";

// Ground truth files
std::string GT_0 = BASEPATH + "Ground_truth_images/disparity_left_motorcycle.bin";
std::string GT_1 = BASEPATH + "Ground_truth_images/disparity_right_motorcycle.bin";

int main(int argc, char** argv) {

    // IMPORT CALIBRATION DATA FROM TXT FILE 
    std::vector<std::string> calibration_data;
    io::readCalibrationData(CALIBRATIONFILE, calibration_data);
    
    // Structure with all calibration data
    MatrixnCalibration matrix_n_calibration;

    // Parameters setting
    char char2delete = ';';
    matrix_n_calibration.setLeftRightCalibrationMat(calibration_data[0], calibration_data[1], char2delete);
    matrix_n_calibration.setDoffs(calibration_data[2]);
    matrix_n_calibration.setBaseline(calibration_data[3]);
    matrix_n_calibration.setWidth(calibration_data[4]);
    matrix_n_calibration.setHeight(calibration_data[5]);

    // Parameters getting
    cv::Mat im_left = matrix_n_calibration.getLeftImage(DATASETPATH + "Motorcycle-perfect/im0.png");
    cv::Mat im_right = matrix_n_calibration.getRightImage(DATASETPATH + "Motorcycle-perfect/im1.png");
    cv::Mat left_camera_mat = matrix_n_calibration.getLeftCamMatrix();
    cv::Mat right_camera_mat = matrix_n_calibration.getRightCamMatrix();
    float doffs = matrix_n_calibration.getDoffs();
    float baseline = matrix_n_calibration.getBaseline();
    int image_width = matrix_n_calibration.getWidth();
    int image_height = matrix_n_calibration.getHeight();

    // Load disparity images from binary file
    matrix_n_calibration.setLeftRightGroundTruth(GT_0, GT_1);
    cv::Mat leftDispMat = matrix_n_calibration.getLeftGT();
    cv::Mat rightDispMat = matrix_n_calibration.getRightGT();

    double min, max;
    cv::minMaxLoc(leftDispMat, &min, &max);
    cv::Mat normalizedMat_l = leftDispMat / float(max);
    cv::Mat normalizedMat_r = rightDispMat / float(max);

    std::vector<cv::Mat> image_vector = { im_left , im_right , normalizedMat_l , normalizedMat_r };
    std::string image_windows[4] = { "Image Left", "Image Right", "Ground Truth Left", "Ground Truth Right" };
    for (size_t i = 0; i < 4; i++)
    {
        cv::namedWindow(image_windows[i], cv::WINDOW_NORMAL);;
        imshow(image_windows[i], image_vector[i]);
    }
    
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // Image conversion to float
    im_left.convertTo(im_left, CV_32F);
    im_right.convertTo(im_right, CV_32F);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // FICTITIOUS LADIMO GRID
    // Fictititous Ladimo grid parameters
    GridParameters grid_parameters;
    grid_parameters.setGap(32);
    grid_parameters.setSamples(4);
    grid_parameters.setGridPoints(image_width, image_height);

    int gridGap = grid_parameters.getGap();
    int patchSamples = grid_parameters.getSamples();
    int pointsOnWidth = grid_parameters.getWidthPoints();
    int pointsOnHeight = grid_parameters.getHeightPoints();

    // Grid formulation
    LadimoGridData ladimo_grid_data;
    ladimo_grid_data.setPixelData(gridGap, leftDispMat, rightDispMat, pointsOnWidth, pointsOnHeight);
    ladimo_grid_data.setSpaceData(baseline, doffs, left_camera_mat, right_camera_mat, pointsOnWidth, pointsOnHeight);

    cv::Mat GridDataLeft_pixel = ladimo_grid_data.getPixelDataLeft();
    cv::Mat GridDataRight_pixel = ladimo_grid_data.getPixelDataRight();
    cv::Mat GridDataLeft_space = ladimo_grid_data.getSpaceDataLeft();
    cv::Mat GridDataRight_space = ladimo_grid_data.getSpaceDataRight();
    

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   // LOOKUP TABLE
   // It contains the indexes to the specific row of the LadimoGridPoints for each one of the points in the grid
   // Used to recover XYZ coordinates of the grid points faster
    LookupTable lookup_table;
    lookup_table.setLookupMatrix(GridDataLeft_pixel.rows, pointsOnWidth, pointsOnHeight);
    cv::Mat lookupMatrix = lookup_table.getLookupMatrix();
    
    io::writeMatrixToFile(lookupMatrix, TXTFILEPATH + "lookup_table.txt");

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // CALCULATE THE NORMALS FOR EACH GRID POINT
    // Get the u,v,w values of the normals (x,y,z magnitudes) and save them in a matrix
    // Normals considering the left grid points 
    NormalCalculator normal_calculator_left;
    normal_calculator_left.setNormalMatrix(GridDataLeft_space, lookupMatrix);
    cv::Mat normals_matrix_left = normal_calculator_left.getNormalMatrix();

    // Normlas considering the right grid points
    NormalCalculator normal_calculator_right;
    normal_calculator_right.setNormalMatrix(GridDataRight_space, lookupMatrix);
    cv::Mat normal_matrix_right = normal_calculator_right.getNormalMatrix();

    // Get the bad points (very big z-value)
    GetWrongPoints get_wrong_points_l;
    GetWrongPoints get_wrong_points_r;

    cv::Mat badDepths_l = get_wrong_points_l.getWrongPoints(GridDataLeft_space);
    cv::Mat badDepths_r = get_wrong_points_r.getWrongPoints(GridDataRight_space);

     //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cv::Mat fullMatrix;
    fullMatrix = getFullMatrix(baseline, left_camera_mat.at<float>(0, 0), doffs, GridDataLeft_space);
    io::writeMatrixToFile(fullMatrix, TXTFILEPATH + "full_matrix.txt");

    // CALCULATE THE DERIVATIVES OVER THE GRID POINTS
    DerivativeClaculator derivative_calculator;
    // Derivative vector with XYZ + disparity information
    std::vector<TotalDerivatives> totVector;
    derivative_calculator.setConsistencyParams(left_camera_mat, gridGap, 120.f);
    derivative_calculator.setDerivativesVector(fullMatrix, lookupMatrix);
    totVector = derivative_calculator.getDerivativesVector();

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // DISPARITY ESTIMATION
    float edge_threshold = 10.f;
    DisparityEstimations disparity_estimations;
    disparity_estimations.setEstimationParameters(left_camera_mat, right_camera_mat, patchSamples, edge_threshold, baseline, doffs);
    
    // ESTIMATION FOR ONLY THE DISPARITY
    disparity_estimations.setEstimatedValues(fullMatrix, lookupMatrix, im_left, im_right, totVector, badDepths_l);
    disparity_estimations.setGuessedValues(fullMatrix, lookupMatrix);
    // Create a disparity image using the disparities from the grid of points and the estimated ones 
    // and check the reliability of the estimations
    cv::Mat guessed_DisparityImage = disparity_estimations.getGuessedDisparity();
    cv::Mat guessed_Z = disparity_estimations.getGuessedZ();
    cv::Mat guessed_X = disparity_estimations.getGuessedX();
    cv::Mat guessed_Y = disparity_estimations.getGuessedY();

    // Normalize the disparity values so that they are between 0 and 1 so that it is possible to visualize them nicely
    double min_raw, max_raw;
    cv::minMaxLoc(guessed_DisparityImage, &min_raw, &max_raw);
    cv::Mat normalizedDispMatRaw_l = guessed_DisparityImage / float(max_raw);

    std::string windowGuessRaw_l = "Guessed Left Disp Raw";

    cv::namedWindow(windowGuessRaw_l, cv::WINDOW_NORMAL);;
    imshow(windowGuessRaw_l, normalizedDispMatRaw_l);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // New algorithm
    /*
    cv::Mat estimated_disparity_matrix_l_new = cv::Mat(rows, cols, CV_32F);
    cv::Mat estimated_X_matrix_new = cv::Mat(rows, cols, CV_32F);
    cv::Mat estimated_Y_matrix_new = cv::Mat(rows, cols, CV_32F);
    cv::Mat estimated_Z_matrix_new = cv::Mat(rows, cols, CV_32F);

    getEstimationSGMbased(lookupMatrix, GridDataLeft_space, fullMatrix, badDepths_l, im_left, im_right, left_camera_mat, right_camera_mat, totVector, sampl_factor, baseline, doffs, estimated_disparity_matrix_l_new, estimated_X_matrix_new, estimated_Y_matrix_new, estimated_Z_matrix_new);


    // Export the disparity values as 2D image
    cv::Mat guessedDisparity_l_new = cv::Mat(totrows, totcols, CV_32F);
    cv::Mat guessedZ_l_new = cv::Mat(totrows, totcols, CV_32F);
    getGuessedDisparity(fullMatrix, lookupMatrix, estimated_disparity_matrix_l_new, estimated_Z_matrix_new, sampl_factor, guessedDisparity_l_new, guessedZ_l_new);

    // Normalize the disparity values so that they are between 0 and 1 so that it is possible to visualize them nicely
    double min_disp_new, max_disp_new;
    cv::minMaxLoc(guessedDisparity_l_new, &min_disp_new, &max_disp_new);
    cv::Mat normalizedDispMat_l_new = guessedDisparity_l_new / float(max_disp_new);

    std::string windowGuess_l_new = "Guessed Left Disp Last";

    cv::namedWindow(windowGuess_l_new, cv::WINDOW_NORMAL);;
    imshow(windowGuess_l_new, normalizedDispMat_l_new);
    

    */

    cv::waitKey(0);
    cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
