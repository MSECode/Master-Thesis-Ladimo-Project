#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <array>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

#include "io_functions.h"
#include "common_structures.h"
#include "image_processing_function.h"
#include "support_structures_simulated_grid.h"
#include "estimation_functions_simulated_grid.h"


// Global variables (Path related to data and storage, which are going to be used all over the code)
std::string DATASETPATH = std::string(getenv("DATASET_MIDDLEBURY"));         // Datasets folder
std::string BASEPATH = std::string(getenv("SGM_BASE_DIR"));                  // Genaral directory of the project
std::string GROUNDTRUTHPATH = BASEPATH + "Ground_truth_images/";
std::string FILESPATH = DATASETPATH + "Motorcycle-perfect/";

int main(int argc, char** argv) {

    // Min and Max values that will be frequently used for image pixel normalization
    // define them at the beginning and then update them when needed
    double min, max;
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // IMPORT CALIBRATION DATA FROM TXT FILE 
    std::vector<std::string> calibration_data;
    io::readMiddleburyCalibrationData(FILESPATH + "calib.txt", calibration_data);

    // Define calibration parameters
    CalibrationAndImagesDataset calibration_images_data;
    char delete_character = ';';

    // Parameters setting
    calibration_images_data.setLeftRightCalibrationMatrices(calibration_data[0], calibration_data[1], delete_character);
    calibration_images_data.setDoffs(calibration_data[2]);
    calibration_images_data.setBaseline(calibration_data[3]);
    calibration_images_data.setWidth(calibration_data[4]);
    calibration_images_data.setHeight(calibration_data[5]);
    calibration_images_data.setLeftRightGroundTruth(GROUNDTRUTHPATH + "disparity_left_motorcycle.bin", GROUNDTRUTHPATH + "disparity_right_motorcycle.bin");
    
    // Parameters getting
    double baseline = calibration_images_data.getBaseline();
    double doffs = calibration_images_data.getDoffs();
    int image_width = calibration_images_data.getWidth();
    int image_height = calibration_images_data.getHeight();
    cv::Mat camera_matrix_left = calibration_images_data.getLeftCamMatrix();
    cv::Mat camera_matrix_right = calibration_images_data.getRightCamMatrix();
    cv::Mat image_left = calibration_images_data.getLeftImage(FILESPATH + "im0.png");
    cv::Mat image_right = calibration_images_data.getRightImage(FILESPATH + "im1.png");
    cv::Mat ground_truth_left = calibration_images_data.getLeftGT();
    cv::Mat ground_truth_right = calibration_images_data.getRightGT();

    // Check correctness the calibration data and the camera matrices
    /*
    std::cout << "Left camera matrix: " << "\n" << camera_matrix_left << "\n" <<
        "Right camera matrix: " << "\n" << camera_matrix_right << "\n" <<
        "Baseline: " << baseline << "\n" <<
        "Disparity offset: " << doffs << "\n" <<
        "Image width: " << image_width << "\n" <<
        "Image height: " << image_height << std::endl;
    */

    std::vector<cv::Mat> image_vector = { image_left , image_right };
    std::string image_windows[2] = { "Image Left", "Image Right" };

    ground_truth_left.convertTo(ground_truth_left, CV_32F);
    ground_truth_right.convertTo(ground_truth_right, CV_32F);

    std::vector<cv::Mat> ground_truth_vector = { ground_truth_left , ground_truth_right };
    std::string ground_truth_image_windows[2] = { "Ground Truth Image Left", "Ground Truth Image Right" };

    for (size_t i = 0; i < image_vector.size(); i++)
    {
        cv::namedWindow(image_windows[i], cv::WINDOW_NORMAL);
        cv::imshow(image_windows[i], image_vector[i]);
        cv::minMaxLoc(ground_truth_vector[i], &min, &max);
        cv::Mat normalized_gt_mat = ground_truth_vector[i] / float(max);
        cv::namedWindow(ground_truth_image_windows[i], cv::WINDOW_NORMAL);
        imshow(ground_truth_image_windows[i], normalized_gt_mat);
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // IMAGE PREPROCESSING
    std::vector<cv::Mat> median_filtered_images_vector(image_vector.size());
    std::vector<cv::Mat> bilateral_filtered_images_vector(image_vector.size());
    ImageProcessingStandard image_pre_processing;
    int kernel_size = 5;
    double gaussian_sigma_X = 0.0;
    double gaussian_sigma_Y = 0.0;
    int diameter_bilateral = 5;
    double sigma_bilateral = 0.0;
    int median_filter_index = (int)ImageProcessingStandard::FilteringTechnique::median_filter;
    int bilateral_filter_index = (int)ImageProcessingStandard::FilteringTechnique::bilateral_filter;

    image_pre_processing.setKernelSize(kernel_size);
    image_pre_processing.setBilateralParams(diameter_bilateral, sigma_bilateral);

    std::string median_image_windows[2] = { "Image Left Median", "Image Right Median" };
    std::string bilateral_image_windows[2] = { "Ground Truth Image Left Bilateral", "Ground Truth Image Right Bilateral" };

    for (size_t i = 0; i < image_vector.size(); i++)
    {
        image_pre_processing.setProcessedImageStandard(image_vector[i], median_filter_index);
        median_filtered_images_vector[i] = image_pre_processing.getProcessedImageStandard(median_filter_index);

        cv::namedWindow(median_image_windows[i], cv::WINDOW_NORMAL);
        cv::imshow(median_image_windows[i], median_filtered_images_vector[i]);
        
        image_pre_processing.setProcessedImageStandard(ground_truth_vector[i], bilateral_filter_index);
        bilateral_filtered_images_vector[i] = image_pre_processing.getProcessedImageStandard(bilateral_filter_index);
        cv::minMaxLoc(bilateral_filtered_images_vector[i], &min, &max);
        cv::Mat normalized_bilateral_gt_mat = ground_truth_vector[i] / float(max);
        cv::namedWindow(bilateral_image_windows[i], cv::WINDOW_NORMAL);
        cv::imshow(bilateral_image_windows[i], normalized_bilateral_gt_mat);
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    /*
    cv::imwrite(FILESPATH + "median_filtered_left.png", median_filtered_images_vector[0]);
    cv::imwrite(FILESPATH + "median_filtered_right.png", median_filtered_images_vector[1]);
    cv::imwrite(FILESPATH + "bilateral_filtered_gt_left.png", bilateral_filtered_images_vector[0]);
    cv::imwrite(FILESPATH + "bilateral_filtered_gt_right.png", bilateral_filtered_images_vector[1]);
    */

    cv::Mat median_left = median_filtered_images_vector[0];
    cv::Mat median_right = median_filtered_images_vector[1];
    cv::Mat bilateral_gt_left = bilateral_filtered_images_vector[0];
    cv::Mat bilateral_gt_right = bilateral_filtered_images_vector[1];
    bilateral_gt_left.convertTo(bilateral_gt_left, CV_64F);
    bilateral_gt_right.convertTo(bilateral_gt_right, CV_64F);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SIMULATED LADIMO GRID
    // Parameters initialization
    SimulatedGridParameters simulated_grid_parameters;
    int simulated_micro_pixel_gap = 16;
    int simulated_4eye_pixel_gap = 24;
    int sample_per_patch_row = 4;
    simulated_grid_parameters.setGap(simulated_4eye_pixel_gap);
    simulated_grid_parameters.setSamples(sample_per_patch_row);
    simulated_grid_parameters.setGridPoints(image_width, image_height);

    int grid_gap = simulated_grid_parameters.getGap();
    int patch_samples = simulated_grid_parameters.getSamples();
    int point_on_width = simulated_grid_parameters.getWidthPoints();
    int points_on_height = simulated_grid_parameters.getHeightPoints();
   
    // Grid formulation
    SimulatedGridData simulated_grid_data;
    simulated_grid_data.setPixelData(grid_gap, bilateral_gt_left, bilateral_gt_right, point_on_width, points_on_height);
    simulated_grid_data.setSpaceData(baseline, doffs, camera_matrix_left, camera_matrix_right, point_on_width, points_on_height);
    cv::Mat grid_data_pixel_left = simulated_grid_data.getPixelDataLeft();
    cv::Mat grid_data_pixel_right = simulated_grid_data.getPixelDataRight();
    cv::Mat grid_data_space_left = simulated_grid_data.getSpaceDataLeft();
    cv::Mat grid_data_space_right = simulated_grid_data.getSpaceDataRight();

    cv::Mat complete_matrix;
    cv::hconcat(grid_data_space_left, grid_data_pixel_left, complete_matrix);
    io::writeMatrixToFile(complete_matrix, FILESPATH + "full_matrix_new.txt");

    cv::Mat left_image_plus_points = image_left;
    cv::Mat right_image_plus_points = image_right;
    cv::Point circle_center_left(0, 0);
    int radius = 2;

    for (size_t i = 0; i < grid_data_pixel_left.rows; i++)
    {
        circle_center_left = cv::Point(grid_data_pixel_left.at<double>(i, 0), grid_data_pixel_left.at<double>(i, 1));
        cv::circle(left_image_plus_points, circle_center_left, radius, cv::Scalar(0, 0, 255), 1, 8);
    }
    
    cv::namedWindow("Left image with simulated grid", cv::WINDOW_NORMAL);
    cv::imshow("Left image with simulated grid", left_image_plus_points);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cv::imwrite(FILESPATH + "left-image-plus-simulated-grid.png", left_image_plus_points);

    /*
    io::writeMatrixToFile(GridDataLeft_pixel, FILESPATH + "left_point_pixel.txt");
    io::writeMatrixToFile(GridDataRight_pixel, FILESPATH + "right_point_pixel.txt");

    io::writeMatrixToFile(GridDataLeft_space, FILESPATH + "left_point_space.txt");
    io::writeMatrixToFile(GridDataRight_space, FILESPATH + "right_point_space.txt");
    */

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // LOOKUP TABLE
    LookupTable lookup_table;
    lookup_table.setLookupMatrix(grid_data_pixel_left.rows, point_on_width, points_on_height);
    cv::Mat lookup_matrix = lookup_table.getLookupMatrix();

    io::writeMatrixToFile(lookup_matrix, FILESPATH + "lookup_table.txt");

    // Get the bad points (very big z-value)
    GetWrongPoints get_wrong_point_left;
    GetWrongPoints get_wrong_point_right;

    cv::Mat bad_depths_left = get_wrong_point_left.getWrongPoints(grid_data_space_left);
    cv::Mat bad_depths_right = get_wrong_point_right.getWrongPoints(grid_data_space_right);

    io::writeMatrixToFile(bad_depths_left, FILESPATH + "bad_depths_left.txt");
    //io::writeMatrixToFile(badDepths_r, FILESPATH + "bad_depths_right.txt");

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // CALCULATE THE DERIVATIVES OVER THE GRID POINTS
    LocalDerivativeClaculator local_derivative_calculator;
    // Derivative vector with XYZ + disparity information
    std::vector<TotalDerivatives> total_derivatives;
    double depth_threshold = 120.0;
    local_derivative_calculator.setConsistencyParams(camera_matrix_left, grid_gap, depth_threshold);
    local_derivative_calculator.setDerivativesVector(complete_matrix, lookup_matrix);
    total_derivatives = local_derivative_calculator.getDerivativesVector();

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // DISPARITY ESTIMATION
    double edge_threshold = 5.0;
    SimulatedGridEstimations simulated_grid_estimations;
    simulated_grid_estimations.setEstimationParameters(camera_matrix_left, 
        camera_matrix_right, 
        patch_samples, 
        edge_threshold, 
        baseline, 
        doffs);

    // ESTIMATION FOR ONLY THE DISPARITY
    simulated_grid_estimations.setEstimatedValues(complete_matrix, lookup_matrix, median_left, median_right, total_derivatives, bad_depths_left);
    simulated_grid_estimations.setGuessedValues(complete_matrix, lookup_matrix);
    // Create a disparity image using the disparities from the grid of points and the estimated ones 
    // and check the reliability of the estimations
    cv::Mat guessed_disparity = simulated_grid_estimations.getGuessedDisparity();
    cv::Mat guessed_Z = simulated_grid_estimations.getGuessedZ();
    cv::Mat guessed_X = simulated_grid_estimations.getGuessedX();
    cv::Mat guessed_Y = simulated_grid_estimations.getGuessedY();

    // Normalize the disparity values so that they are between 0 and 1 so that it is possible to visualize them nicely
    guessed_disparity.convertTo(guessed_disparity, CV_32F);
    cv::minMaxLoc(guessed_disparity, &min, &max);
    cv::Mat normalized_disparity_raw = guessed_disparity / float(max);

    cv::namedWindow("Guessed Left Disp Raw", cv::WINDOW_NORMAL);
    cv::imshow("Guessed Left Disp Raw", normalized_disparity_raw);

    int resh_rows = guessed_disparity.rows * guessed_disparity.cols;
    cv::Mat guessed_disp_reshaped = normalized_disparity_raw.reshape(1, resh_rows);
    cv::Mat guessed_X_reshaped = guessed_X.reshape(1, resh_rows);
    cv::Mat guessed_Y_reshaped = guessed_Y.reshape(1, resh_rows);
    cv::Mat guessed_Z_reshaped = guessed_Z.reshape(1, resh_rows);
    cv::Mat merged_mat;
    cv::hconcat(guessed_X_reshaped, guessed_Y_reshaped, merged_mat);
    cv::hconcat(merged_mat, guessed_Z_reshaped, merged_mat);

    io::writeMatrixToFile(merged_mat, FILESPATH + "XYZ_guessed_before_post_proc.txt");

    std::vector<cv::Mat> estimated_image_vector = { guessed_disparity, guessed_X, guessed_Y, guessed_Z };

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // POST PROCESSING
    ImageProcessingStandard image_post_processing;
    image_post_processing.setKernelSize(kernel_size);
    image_post_processing.setBilateralParams(diameter_bilateral, sigma_bilateral);

    image_post_processing.setProcessedImageStandard(normalized_disparity_raw, median_filter_index);
    cv::Mat median_filtered_disparity = image_post_processing.getProcessedImageStandard(median_filter_index);
    image_post_processing.setProcessedImageStandard(normalized_disparity_raw, bilateral_filter_index);
    cv::Mat bilteral_filtered_disparity = image_post_processing.getProcessedImageStandard(bilateral_filter_index);

    cv::namedWindow("Median Filtered Disparity", cv::WINDOW_NORMAL);
    cv::imshow("Median Filtered Disparity", median_filtered_disparity);
    cv::namedWindow("Bilateral Filtered Disparity", cv::WINDOW_NORMAL);
    cv::imshow("Bilateral Filtered Disparity", bilteral_filtered_disparity);
   

    cv::waitKey(0);
    cv::destroyAllWindows();

	return EXIT_SUCCESS;
}