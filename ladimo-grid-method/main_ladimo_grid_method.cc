#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include "io_functions.h"
#include "common_structures.h"
#include "image_processing_function.h"
#include "support_functions_ladimo_grid_method.h"
#include "support_structures_ladimo_grid_method.h"
#include "estimation_functions_ladimo_grid_method.h"

int main(int argc, char** argv) {
	// Global variables (Path related to data and storage, which are going to be used all over the code)
	std::string DATASETPATH = std::string(getenv("DATASET_LADIMO"));           // Datasets folder
	std::string BASEPATH = std::string(getenv("SGM_BASE_DIR"));                // Genaral directory of the project
	std::string FILESPATH = DATASETPATH + "stereo_matching_set_2b/";           // Calibration file of the images that we are going to use


    // Min and Max values that will be frequently used for image pixel normalization
    // define them at the beginning and then update them when needed
    double min, max;
    double alpha_img_conv = 0.0;

    // IMPORT CALIBRATION DATA FROM TXT FILE
    std::vector<std::string> calibration_data;
    io::readLadimoCalibration(FILESPATH + "README.txt", calibration_data);

    // Define calibration parameters
    LadimoCalibrationandMatrices ladimo_matrices_and_parameters;
    char sep_character = ',';

    // Parameters setting
    ladimo_matrices_and_parameters.setBaseline(calibration_data.back());
    ladimo_matrices_and_parameters.setImageDimension(calibration_data[0], sep_character);
    ladimo_matrices_and_parameters.setLeftRightCameraMatrices(calibration_data[2], 
        calibration_data[2 + 3], 
        calibration_data[1], 
        calibration_data[1 + 3], 
        sep_character);
    ladimo_matrices_and_parameters.setLeftRightTransformDeviceToStereo(FILESPATH + "rotation_left_matrix.yml", 
        FILESPATH + "rotation_right_matrix.yml",
        "Left Rotation Matrix",
        "Right Rotation Matrix");

    // Parameters getting
    double baseline = ladimo_matrices_and_parameters.getBaseline();
    double doffs = 0.0;
    int image_width = ladimo_matrices_and_parameters.getImageWidth();
    int image_height = ladimo_matrices_and_parameters.getImageHeight();
    cv::Mat camera_matrix_left = ladimo_matrices_and_parameters.getLeftCameraMatrix();
    cv::Mat camera_matrix_right = ladimo_matrices_and_parameters.getRightCameraMatrix();
    cv::Mat rot_dist_undist_left = ladimo_matrices_and_parameters.getLeftTransfDeviceToStereo();
    cv::Mat rot_dist_undist_right = ladimo_matrices_and_parameters.getRightTransfDeviceToStereo();

    // Check the values
    /*
    std::cout << "baseline: " << baseline << "\n" <<
        "doffs: " << doffs << "\n" <<
        "image width: " << image_width << "\n" <<
        "image height: " << image_height << "\n" <<
        "camera matrix left: " << camera_matrix_left << "\n" <<
        "camera matrix right: " << camera_matrix_right << "\n" <<
        "rotation matrix left from dist to undist: " << rot_dist_undist_left << "\n" <<
        "rotation matrix right from dist to undist: " << rot_dist_undist_right << std::endl;
    */

    cv::Mat image_left_16_rgb = ladimo_matrices_and_parameters.getImageLeft(FILESPATH + "stereo_rectified_left_16bit.png");
    cv::Mat image_right_16_rgb = ladimo_matrices_and_parameters.getImageRight(FILESPATH + "stereo_rectified_right_16bit.png");


    cv::Mat image_left_8_rgb, image_right_8_rgb;
    alpha_img_conv = 1. / (pow(2.0, 8.0) - 1);
    image_left_16_rgb.convertTo(image_left_8_rgb, CV_8UC3, alpha_img_conv);
    image_right_16_rgb.convertTo(image_right_8_rgb, CV_8UC3, alpha_img_conv);
    std::vector<cv::Mat> image_vector = { image_left_8_rgb, image_right_8_rgb };
    std::string image_windows[2] = { "Image Left", "Image Right" };

    for (size_t i = 0; i < image_vector.size(); i++)
    {
        cv::namedWindow(image_windows[i], cv::WINDOW_NORMAL);
        cv::imshow(image_windows[i], image_vector[i]);
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    // IMPORT INITIAL POINT CLOUD DATA
    cv::Mat grid_raw_data;
    grid_raw_data = io::readCSV(FILESPATH + "input_grid_data.csv", ',');
    cv::Mat grid_data_to_use = getInitialGridData(grid_raw_data, rot_dist_undist_left, camera_matrix_left, baseline);

    // Check if transformation works correctly
    /*
    int random_row = 300;
    std::cout << "Grid position: " << grid_data_to_use.row(random_row).colRange(0, 2) << "\n" <<
        "Initial 3D position distorted: " << grid_raw_data.row(random_row).colRange(2, 5) << "\n" <<
        "Transformed 3D position rectified: " << grid_data_to_use.row(random_row).colRange(2, 5) << "\n" <<
        "Projected pixel positions: " << grid_data_to_use.row(random_row).colRange(5, 7) << "\n" <<
        "------------------------------------" << std::endl;
    */

    // Declare what you need
    cv::FileStorage grid_point_data_file(FILESPATH + "grid_matrix.yml", cv::FileStorage::WRITE);
    // Write to file!
    grid_point_data_file << "Observations" << grid_data_to_use;
    io::writeMatrixToFile(grid_data_to_use, FILESPATH + "grid_data_new.txt");

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // LADIMO REAL GRID AND NEIGHBOURS SELECTION
    LadimoGridAccelerationStructure ladimo_grid_acc_structure;
    // Set Parameters
    ladimo_grid_acc_structure.setDataParameters(grid_data_to_use, 0.12f, 0.07f);
    ladimo_grid_acc_structure.setObservationData(grid_data_to_use);

    // Get Parameters
    std::vector<GridPositions> grid_positions = ladimo_grid_acc_structure.getGridPositions();
    // Check if copying positions works fine
    /*
    for (auto& grid_pos : grid_positions)
    {
        std::cout << "x grid position: \n" << grid_pos.x << "\n" <<
            "y grid position: \n" << grid_pos.y << std::endl;
    }
    */
    ladimo_grid_acc_structure.setLaserGrid(grid_positions);
    std::vector<ObservationData> grid_observations = ladimo_grid_acc_structure.getObservationVector();
    std::vector<GridSquare> grid_squares = ladimo_grid_acc_structure.getGridSquares();
    std::vector<CompleteDerivatives> internal_derivatives = ladimo_grid_acc_structure.getInternalDerivatives();
    std::vector<CompleteDerivatives> external_derivatives = ladimo_grid_acc_structure.getExternalDerivatives();
    /*
    for (auto& qsrvec : gridSquares)
    {
        std::cout << "TL: " << qsrvec.top_left_index << " | " <<
            "TR: " << qsrvec.top_right_index << " | " <<
            "BL: " << qsrvec.bottom_left_index << " | " <<
            "BR: " << qsrvec.bottom_right_index << std::endl;
    }
    */
    /*
    std::cout << "number of points :" << observationsVector.size() << std::endl;
    std::cout << "number of squared grids: " << gridSquares.size() << std::endl;

    int chosen_row = 760;
    std::cout << "chosen_row: " << chosen_row << std::endl;
    std::cout << " chosen point: " << observationsVector[chosen_row].x_px <<  std::endl;
    std::cout << " top derivative: " << externalCompleteDerivatives[chosen_row].top_deriv.x_px <<
        " right derivative: " << internalCompleteDerivatives[chosen_row].right_deriv.x_px <<
        " bottom derivative: " << internalCompleteDerivatives[chosen_row].bottom_deriv.x_px <<
        " left derivative: " << externalCompleteDerivatives[chosen_row].left_deriv.x_px << std::endl;
    */
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // IMAGE PREPROCESSING
    std::vector<cv::Mat> median_filtered_images_vector(image_vector.size());
    std::vector<cv::Mat> bilateral_filtered_images_vector(image_vector.size());
    ImageProcessingStandard standard_image_processing;
    int kernel_size = 5;
    double gaussian_sigma_X = 0.0;
    double gaussian_sigma_Y = 0.0;
    int diameter_bilateral = 5;
    double sigma_bilateral = 0.0;
    int median_filter_index = (int)ImageProcessingStandard::FilteringTechnique::median_filter;
    int bilateral_filter_index = (int)ImageProcessingStandard::FilteringTechnique::bilateral_filter;

    standard_image_processing.setKernelSize(kernel_size);
    standard_image_processing.setGaussianBlurParams(gaussian_sigma_X, gaussian_sigma_Y);
    standard_image_processing.setBilateralParams(diameter_bilateral, sigma_bilateral);

    std::string median_image_windows[2] = { "Image Left Median", "Image Right Median" };
    std::string bilateral_image_windows[2] = { "Image Left Bilateral", "Image Right Bilateral" };

    for (size_t i = 0; i < image_vector.size(); i++)
    {
        standard_image_processing.setProcessedImageStandard(image_vector[i], median_filter_index);
        median_filtered_images_vector[i] = standard_image_processing.getProcessedImageStandard(median_filter_index);
        cv::namedWindow(median_image_windows[i], cv::WINDOW_NORMAL);
        cv::imshow(median_image_windows[i], median_filtered_images_vector[i]);
    }
    for (size_t i = 0; i < image_vector.size(); i++)
    {
        standard_image_processing.setProcessedImageStandard(image_vector[i], bilateral_filter_index);
        bilateral_filtered_images_vector[i] = standard_image_processing.getProcessedImageStandard(bilateral_filter_index);
        cv::namedWindow(bilateral_image_windows[i], cv::WINDOW_NORMAL);
        cv::imshow(bilateral_image_windows[i], bilateral_filtered_images_vector[i]);
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    cv::imwrite(FILESPATH + "median_filtered_left.png", median_filtered_images_vector[0]);
    cv::imwrite(FILESPATH + "median_filtered_right.png", median_filtered_images_vector[1]);

    cv::imwrite(FILESPATH + "bilateral_filtered_left.png", bilateral_filtered_images_vector[0]);
    cv::imwrite(FILESPATH + "bilateral_filtered_right.png", bilateral_filtered_images_vector[1]);


    cv::Mat medianLeft = median_filtered_images_vector[0];
    cv::Mat medianRight = median_filtered_images_vector[1];

    cv::Mat bilateralLeft = bilateral_filtered_images_vector[0];
    cv::Mat bilateralRight = bilateral_filtered_images_vector[1];

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // REAL LADIMO GRID ESTIMATIONS WITH THE NEW FUNCTIONS IN TESTFUNCTIONS
    cv::Mat median_left_gray(medianLeft.size(), CV_8U);
    cv::Mat median_right_gray(medianLeft.size(), CV_8U);
    cv::cvtColor(medianLeft, median_left_gray, cv::COLOR_RGB2GRAY);
    cv::cvtColor(medianRight, median_right_gray, cv::COLOR_RGB2GRAY);
    LadimoGridEstimations ladimo_grid_estimation;
    ladimo_grid_estimation.setEstimationParameters(camera_matrix_left, camera_matrix_right, baseline, 8);
    ladimo_grid_estimation.setNecessaryVectors(grid_squares, grid_observations, internal_derivatives, external_derivatives);
    ladimo_grid_estimation.initilizeEstimationMatrices();
    ladimo_grid_estimation.setEstimations(medianLeft, medianRight);
    ladimo_grid_estimation.setGuesses();

    cv::Mat newEstimated_disp = ladimo_grid_estimation.getEstimations_disp();
    cv::Mat newEstimated_X = ladimo_grid_estimation.getEstimations_X();
    cv::Mat newEstimated_Y = ladimo_grid_estimation.getEstimations_Y();
    cv::Mat newEstimated_Z = ladimo_grid_estimation.getEstimations_Z();

    cv::Mat newEstimated_x_px;
    cv::Mat newEstimated_y_px;
    cv::divide(newEstimated_X, newEstimated_Z, newEstimated_x_px);
    cv::multiply(newEstimated_x_px, camera_matrix_left.at<double>(0, 0), newEstimated_x_px);
    cv::add(newEstimated_x_px, camera_matrix_left.at<double>(0, 2), newEstimated_x_px);

    cv::divide(newEstimated_Y, newEstimated_Z, newEstimated_y_px);
    cv::multiply(newEstimated_y_px, camera_matrix_left.at<double>(1, 1), newEstimated_y_px);
    cv::add(newEstimated_y_px, camera_matrix_left.at<double>(1, 2), newEstimated_y_px);

    cv::Mat Newestimated_XYZ;
    cv::hconcat(newEstimated_X, newEstimated_Y, Newestimated_XYZ);
    cv::hconcat(Newestimated_XYZ, newEstimated_Z, Newestimated_XYZ);
    io::writeMatrixToFile(Newestimated_XYZ, FILESPATH + "complete_cases_estimated_XYZ_8_samples_new_01.txt");

    cv::Mat Newestimated_xyd;
    cv::hconcat(newEstimated_x_px, newEstimated_y_px, Newestimated_xyd);
    cv::hconcat(Newestimated_xyd, newEstimated_disp, Newestimated_xyd);
    io::writeMatrixToFile(Newestimated_xyd, FILESPATH + "complete_cases_estimated_xyd_8_samples_new_01.txt");

    // No edge
    cv::Mat newEstimated_disp_no_edge = ladimo_grid_estimation.getEstimations_disp_no_edge();
    cv::Mat newEstimated_X_no_edge = ladimo_grid_estimation.getEstimations_X_no_edge();
    cv::Mat newEstimated_Y_no_edge = ladimo_grid_estimation.getEstimations_Y_no_edge();
    cv::Mat newEstimated_Z_no_edge = ladimo_grid_estimation.getEstimations_Z_no_edge();

    cv::Mat Newestimated_XYZ_no_edge;
    cv::hconcat(newEstimated_X_no_edge, newEstimated_Y_no_edge, Newestimated_XYZ_no_edge);
    cv::hconcat(Newestimated_XYZ_no_edge, newEstimated_Z_no_edge, Newestimated_XYZ_no_edge);
    io::writeMatrixToFile(Newestimated_XYZ_no_edge, FILESPATH + "no_edge_case_estimated_XYZ_8_samples_new_01.txt");

    // Strong Edge
    cv::Mat newEstimated_disp_strong_edge = ladimo_grid_estimation.getEstimations_disp_strong_edge();
    cv::Mat newEstimated_X_strong_edge = ladimo_grid_estimation.getEstimations_X_strong_edge();
    cv::Mat newEstimated_Y_strong_edge = ladimo_grid_estimation.getEstimations_Y_strong_edge();
    cv::Mat newEstimated_Z_strong_edge = ladimo_grid_estimation.getEstimations_Z_strong_edge();

    cv::Mat Newestimated_XYZ_strong_edge;
    cv::hconcat(newEstimated_X_strong_edge, newEstimated_Y_strong_edge, Newestimated_XYZ_strong_edge);
    cv::hconcat(Newestimated_XYZ_strong_edge, newEstimated_Z_strong_edge, Newestimated_XYZ_strong_edge);
    io::writeMatrixToFile(Newestimated_XYZ_strong_edge, FILESPATH + "strong_edge_case_estimated_XYZ_8_samples_new_01.txt");

    // Soft Edge
    cv::Mat newEstimated_disp_soft_edge = ladimo_grid_estimation.getEstimations_disp_soft_edge();
    cv::Mat newEstimated_X_soft_edge = ladimo_grid_estimation.getEstimations_X_soft_edge();
    cv::Mat newEstimated_Y_soft_edge = ladimo_grid_estimation.getEstimations_Y_soft_edge();
    cv::Mat newEstimated_Z_soft_edge = ladimo_grid_estimation.getEstimations_Z_soft_edge();
    cv::Mat Newestimated_XYZ_soft_edge;
    cv::hconcat(newEstimated_X_soft_edge, newEstimated_Y_soft_edge, Newestimated_XYZ_soft_edge);
    cv::hconcat(Newestimated_XYZ_soft_edge, newEstimated_Z_soft_edge, Newestimated_XYZ_soft_edge);
    io::writeMatrixToFile(Newestimated_XYZ_soft_edge, FILESPATH + "soft_edge_case_estimated_XYZ_8_samples_new_01.txt");


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // POST PROCESSING
    
    ImageProcessingMorphology post_processing_ladimo;
    int morph_size = 5;
    int morph_shape = (int)cv::MORPH_RECT;
    post_processing_ladimo.setProcessedParametersMorph(morph_size, morph_shape);
    int closing_indx = (int)ImageProcessingMorphology::MorphologicalFilters::morph_close;
    int opening_indx = (int)ImageProcessingMorphology::MorphologicalFilters::morph_open;
    int mixed_index = (int)ImageProcessingMorphology::MorphologicalFilters::morph_mixed;
    int dilation_index = (int)ImageProcessingMorphology::MorphologicalFilters::morph_dilation;

    //std::vector<cv::Mat> estimatedImages_real_grid = { newEstimated_X, newEstimated_Y, newEstimated_Z };
    
    post_processing_ladimo.setProcessedImageMorph(newEstimated_Z, opening_indx);
    cv::Mat guessed_Z_real_post = post_processing_ladimo.getProcessedImageMorph(opening_indx);

    cv::Mat merged_mat_XYZ_post;
    cv::hconcat(newEstimated_X, newEstimated_Y, merged_mat_XYZ_post);
    cv::hconcat(merged_mat_XYZ_post, guessed_Z_real_post, merged_mat_XYZ_post);
    
    io::writeMatrixToFile(merged_mat_XYZ_post, FILESPATH + "XYZ_guessed_opening_real_grid_new_02.txt");
    

    // Median blur for post processing
    cv::Mat median_blurred_est_Z = cv::Mat(newEstimated_X.size(), CV_32F);

    //std::vector<cv::Mat> estimated_image_median_blurred = { median_blurred_est_X, median_blurred_est_Y, median_blurred_est_Z };
    int median_filt_kernel_size = 3;
    newEstimated_Z.convertTo(newEstimated_Z, CV_32F);
    cv::medianBlur(newEstimated_Z, median_blurred_est_Z, median_filt_kernel_size);
    

    cv::Mat merged_mat_XYZ_median_blurred;
    newEstimated_X.convertTo(newEstimated_X, CV_32F);
    newEstimated_Y.convertTo(newEstimated_Y, CV_32F);
    cv::hconcat(newEstimated_X, newEstimated_Y, merged_mat_XYZ_median_blurred);
    cv::hconcat(merged_mat_XYZ_median_blurred, median_blurred_est_Z, merged_mat_XYZ_median_blurred);

    io::writeMatrixToFile(merged_mat_XYZ_median_blurred, FILESPATH + "XYZ_guessed_median_blurred_real_grid_new_02.txt");

    return EXIT_SUCCESS;
}