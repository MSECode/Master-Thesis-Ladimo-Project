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

#include "ioFunctions.h"
#include "preWorkingLadimoImages.h"
#include "usedStructnFunctions.h"
#include "smallSupportFunctions.h"
#include "getWrongPoints.h"
#include "matchingCostFunctions.h"
#include "preGridOPerations.h"
#include "onGridOperations.h"
#include "postProcessingFunctions.h"
#include "realLadimoGridEstimations.h"
#include "testFunctions.h"

// Global variables (Path related to data and storage, which are going to be used all over the code)
std::string DATASETPATH = std::string(getenv("DATASET_LADIMO"));           // Datasets folder
std::string BASEPATH = std::string(getenv("SGM_BASE_DIR"));                // Genaral directory of the project
std::string FILESPATH = DATASETPATH + "stereo_matching_set_2a/";           // Calibration file of the images that we are going to use

int main(int argc, char **argv)
{
    // Min and Max values that will be frequently used for image pixel normalization
    // define them at the beginning and then update them when needed
    double min, max;
    double alpha_img_conv = 0.0;

    // IMPORT CALIBRATION DATA FROM TXT FILE
    std::vector<std::string> calibration_data;
    io::readLadimoCalibration(FILESPATH + "README.txt", calibration_data);

    // Define calibration parameters
    LadimoMatricesImporting ladimo_matrices_importing;
    char sep_character = ',';

    // Parameters setting
    ladimo_matrices_importing.setBaseline(calibration_data.back());
    ladimo_matrices_importing.setImageDimensions(calibration_data[0], sep_character);
    ladimo_matrices_importing.setFocalLength(calibration_data[1], calibration_data[1 + 3], sep_character);
    ladimo_matrices_importing.setPrincipalPoints(calibration_data[2], calibration_data[2 + 3], sep_character);
    ladimo_matrices_importing.setCameraMatrixLeft();
    ladimo_matrices_importing.setCameraMatrixRight();
    ladimo_matrices_importing.setRotationMatrixDeviceToStereoLeft(FILESPATH + "rotation_left_matrix.yml", "Left Rotation Matrix");
    ladimo_matrices_importing.setRotationMatrixDeviceToStereoRight(FILESPATH + "rotation_right_matrix.yml", "Right Rotation Matrix");

    // Parameters getting
    double baseline = ladimo_matrices_importing.getBaseline();
    double doffs = 0.0;
    int image_width = ladimo_matrices_importing.getImageWidth();
    int image_height = ladimo_matrices_importing.getImageHeight();
    cv::Mat camera_matrix_left = ladimo_matrices_importing.getCameraMatrixLeft();
    cv::Mat camera_matrix_right = ladimo_matrices_importing.getCameraMatrixRight();
    cv::Mat rot_dist_undist_left = ladimo_matrices_importing.getRotationMatrixDeviceToStereoLeft();
    cv::Mat rot_dist_undist_right = ladimo_matrices_importing.getRotationMatrixDeviceToStereoRight();
    
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

    cv::Mat image_left_16_rgb = ladimo_matrices_importing.getImageLeft(FILESPATH + "stereo_rectified_left_16bit.png");
    cv::Mat image_right_16_rgb = ladimo_matrices_importing.getImageRight(FILESPATH + "stereo_rectified_right_16bit.png");

    
    cv::Mat image_left_8_rgb, image_right_8_rgb;
    alpha_img_conv = 1. / (pow(2.0, 8.0) - 1);
    image_left_16_rgb.convertTo(image_left_8_rgb, CV_8UC3, alpha_img_conv);
    image_right_16_rgb.convertTo(image_right_8_rgb, CV_8UC3, alpha_img_conv);
    std::vector<cv::Mat> image_vector = { image_left_8_rgb, image_right_8_rgb };
    std::string image_windows[2] = { "Image Left", "Image Right" };

    for (size_t i = 0; i < image_vector.size(); i++)
    {
        cv::namedWindow(image_windows[i], cv::WINDOW_NORMAL);
        imshow(image_windows[i], image_vector[i]);
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    // IMPORT INITIAL POINT CLOUD DATA
    cv::Mat gridPointsData;
    gridPointsData = io::readCSV(FILESPATH + "input_grid_data.csv", ',');
    cv::Mat grid_data_to_use = getInitialGridData(gridPointsData, rot_dist_undist_left, camera_matrix_left, baseline);

    // Declare what you need
    cv::FileStorage grid_point_data_file(FILESPATH + "grid_matrix.yml", cv::FileStorage::WRITE);
    // Write to file!
    grid_point_data_file << "Observations" << grid_data_to_use;
    io::writeMatrixToFile(grid_data_to_use, FILESPATH + "grid_data.txt");
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // LADIMO REAL GRID AND NEIGHBOURS SELECTION
    RealLadimoGridAccStructure real_ladimo_grid_acc_structure;
    // Set Parameters
    real_ladimo_grid_acc_structure.setDataParameters(gridPointsData, 0.12f, 0.07f);
    real_ladimo_grid_acc_structure.setObservationData(gridPointsData);

    // Get Parameters
    std::vector<GridPositions> grid_positions = real_ladimo_grid_acc_structure.getGridPositions();
    real_ladimo_grid_acc_structure.setLaserGrid(grid_positions);
    std::vector<ObservationData> observationsVector = real_ladimo_grid_acc_structure.getObservationVector();
    std::vector<GridSquare> gridSquares = real_ladimo_grid_acc_structure.getGridSquares();
    std::vector<cv::Mat> SGMcostCubes = real_ladimo_grid_acc_structure.getSGMCostCubesVector();
    std::vector<CompleteValuesDerivatives> internalCompleteDerivatives = real_ladimo_grid_acc_structure.getCompleteInternalDerivatives();
    std::vector<CompleteValuesDerivatives> externalCompleteDerivatives = real_ladimo_grid_acc_structure.getCompleteExternalDerivatives();
    std::vector<TotalDerivatives> derivativesVector = real_ladimo_grid_acc_structure.getDerivativesVector();
    std::vector<TotalDerivatives> derivativesVectorNew = real_ladimo_grid_acc_structure.getDerivativesVectorNew();
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
    Preprocessing pre_processing;
    int kernel_size = 5;
    double gaussian_sigma_X = 0.f;
    double gaussian_sigma_Y = 0.f;
    int diameter_bilateral = 5;
    double sigma_bilateral = 0.f;
    int median_filter_index = (int)Preprocessing::FilteringTechnique::median_filter;
    int bilateral_filter_index = (int)Preprocessing::FilteringTechnique::bilateral_filter;

    pre_processing.setPreprocessingParameters(kernel_size, gaussian_sigma_X, gaussian_sigma_Y, diameter_bilateral, sigma_bilateral);

    std::string median_image_windows[2] = {"Image Left Median", "Image Right Median"};
    std::string bilateral_image_windows[2] = {"Image Left Bilateral", "Image Right Bilateral"};

    for (size_t i = 0; i < image_vector.size(); i++)
    {
        pre_processing.setPreprocessedImage(image_vector[i], median_filter_index);
        median_filtered_images_vector[i] = pre_processing.getPreprocessedImage(median_filter_index);
        cv::namedWindow(median_image_windows[i], cv::WINDOW_NORMAL);
        cv::imshow(median_image_windows[i], median_filtered_images_vector[i]);
    }
    for (size_t i = 0; i < image_vector.size(); i++)
    {
        pre_processing.setPreprocessedImage(image_vector[i], bilateral_filter_index);
        bilateral_filtered_images_vector[i] = pre_processing.getPreprocessedImage(bilateral_filter_index);
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

    RealGridOperationsTesting real_grid_estimations_testing;
    real_grid_estimations_testing.setEstimationParameters(camera_matrix_left, camera_matrix_right, baseline, 8);
    real_grid_estimations_testing.setNecessaryVectors(gridSquares, observationsVector, internalCompleteDerivatives, externalCompleteDerivatives);
    real_grid_estimations_testing.initilizeEstimationMatrices();
    real_grid_estimations_testing.setEstimations(medianLeft, medianRight);
    real_grid_estimations_testing.setGuesses();

    cv::Mat newEstimated_disp = real_grid_estimations_testing.getEstimations_disp();
    cv::Mat newEstimated_X = real_grid_estimations_testing.getEstimations_X();
    cv::Mat newEstimated_Y = real_grid_estimations_testing.getEstimations_Y();
    cv::Mat newEstimated_Z = real_grid_estimations_testing.getEstimations_Z();

    cv::Mat newEstimated_x_px;
    cv::Mat newEstimated_y_px;
    cv::divide(newEstimated_X, newEstimated_Z, newEstimated_x_px);
    cv::multiply(newEstimated_x_px, camera_matrix_left.at<float>(0, 0), newEstimated_x_px);
    cv::add(newEstimated_x_px, camera_matrix_left.at<float>(0, 2), newEstimated_x_px);

    cv::divide(newEstimated_Y, newEstimated_Z, newEstimated_y_px);
    cv::multiply(newEstimated_y_px, camera_matrix_left.at<float>(1, 1), newEstimated_y_px);
    cv::add(newEstimated_y_px, camera_matrix_left.at<float>(1, 2), newEstimated_y_px);

    cv::Mat Newestimated_XYZ;
    cv::hconcat(newEstimated_X, newEstimated_Y, Newestimated_XYZ);
    cv::hconcat(Newestimated_XYZ, newEstimated_Z, Newestimated_XYZ);
    io::writeMatrixToFile(Newestimated_XYZ, FILESPATH + "complete_cases_estimated_XYZ_8_samples_05.txt");

    cv::Mat Newestimated_xyd;
    cv::hconcat(newEstimated_x_px, newEstimated_y_px, Newestimated_xyd);
    cv::hconcat(Newestimated_xyd, newEstimated_disp, Newestimated_xyd);
    io::writeMatrixToFile(Newestimated_xyd, FILESPATH + "complete_cases_estimated_xyd_8_samples_05.txt");

    
    cv::Mat estimated_rgb_values(Newestimated_XYZ.rows, 3, Newestimated_XYZ.type(), cv::Scalar::all(0));
    getEstimatedImageRGB(Newestimated_xyd, image_left_8_rgb, estimated_rgb_values);

    cv::Mat colored_point_cloud;
    cv::hconcat(Newestimated_XYZ, estimated_rgb_values, colored_point_cloud);
    io::writeMatrixToFile(colored_point_cloud, FILESPATH + "colored_estimated_point_cloud.txt");
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // POST PROCESSING
    PostProcessingLadimo post_processing_ladimo;
    int morph_size = 5;
    int morph_shape = (int)cv::MORPH_RECT;
    post_processing_ladimo.setPostProcParameters(morph_size, morph_shape);
    int closing_indx = (int)PostProcessingLadimo::MorphologicalFilters::morph_close;
    int opening_indx = (int)PostProcessingLadimo::MorphologicalFilters::morph_open;
    int mixed_index = (int)PostProcessingLadimo::MorphologicalFilters::morph_mixed;
    int dilation_index = (int)PostProcessingLadimo::MorphologicalFilters::morph_dilation;

    std::vector<cv::Mat> estimatedImages_real_grid = {newEstimated_X, newEstimated_Y, newEstimated_Z};

    for (size_t i = 0; i < estimatedImages_real_grid.size(); i++)
    {
        post_processing_ladimo.setPostProcOperation(estimatedImages_real_grid[i], opening_indx);
        estimatedImages_real_grid[i] = post_processing_ladimo.getProcessedImage(opening_indx);
    }

    int resheped_rows = newEstimated_X.rows * newEstimated_X.cols;
    cv::Mat guessed_X_real_post = estimatedImages_real_grid[0].reshape(1, resheped_rows);
    cv::Mat guessed_Y_real_post = estimatedImages_real_grid[1].reshape(1, resheped_rows);
    cv::Mat guessed_Z_real_post = estimatedImages_real_grid[2].reshape(1, resheped_rows);
    cv::Mat merged_mat_XYZ_post;
    cv::hconcat(guessed_X_real_post, guessed_Y_real_post, merged_mat_XYZ_post);
    cv::hconcat(merged_mat_XYZ_post, guessed_Z_real_post, merged_mat_XYZ_post);

    //io::writeMatrixToFile(merged_mat_XYZ_post, FILESPATH + "XYZ_guessed_opening_real_grid_01.txt");

    /*
    // Median blur for post processing
    cv::Mat median_blurred_est_X = cv::Mat(newEstimated_X.size(), newEstimated_X.type(), cv::Scalar::all(0));
    cv::Mat median_blurred_est_Y = cv::Mat(newEstimated_X.size(), newEstimated_X.type(), cv::Scalar::all(0));
    cv::Mat median_blurred_est_Z = cv::Mat(newEstimated_X.size(), newEstimated_X.type(), cv::Scalar::all(0));

    std::vector<cv::Mat> estimated_image_median_blurred = { median_blurred_est_X, median_blurred_est_Y, median_blurred_est_Z };
    int median_filt_kernel_size = 3;
    for (size_t i = 0; i < estimatedImages_real_grid.size(); i++)
    {
        
        cv::medianBlur(estimatedImages_real_grid[i], estimated_image_median_blurred[i], median_filt_kernel_size);
    }

    cv::Mat merged_mat_XYZ_median_blurred;
    cv::hconcat(median_blurred_est_X, median_blurred_est_Y, merged_mat_XYZ_median_blurred);
    cv::hconcat(merged_mat_XYZ_median_blurred, median_blurred_est_Z, merged_mat_XYZ_median_blurred);

    io::writeMatrixToFile(merged_mat_XYZ_median_blurred, FILESPATH + "XYZ_guessed_median_blurred_real_grid_02.txt");
    */

    // No edge
    cv::Mat newEstimated_disp_no_edge = real_grid_estimations_testing.getEstimations_disp_no_edge();
    cv::Mat newEstimated_X_no_edge = real_grid_estimations_testing.getEstimations_X_no_edge();
    cv::Mat newEstimated_Y_no_edge = real_grid_estimations_testing.getEstimations_Y_no_edge();
    cv::Mat newEstimated_Z_no_edge = real_grid_estimations_testing.getEstimations_Z_no_edge();

    cv::Mat Newestimated_XYZ_no_edge;
    cv::hconcat(newEstimated_X_no_edge, newEstimated_Y_no_edge, Newestimated_XYZ_no_edge);
    cv::hconcat(Newestimated_XYZ_no_edge, newEstimated_Z_no_edge, Newestimated_XYZ_no_edge);
    io::writeMatrixToFile(Newestimated_XYZ_no_edge, FILESPATH + "no_edge_case_estimated_XYZ_8_samples_06.txt");

    // Strong Edge
    cv::Mat newEstimated_disp_strong_edge = real_grid_estimations_testing.getEstimations_disp_strong_edge();
    cv::Mat newEstimated_X_strong_edge = real_grid_estimations_testing.getEstimations_X_strong_edge();
    cv::Mat newEstimated_Y_strong_edge = real_grid_estimations_testing.getEstimations_Y_strong_edge();
    cv::Mat newEstimated_Z_strong_edge = real_grid_estimations_testing.getEstimations_Z_strong_edge();

    cv::Mat median_blurred_est_X = cv::Mat(newEstimated_X_strong_edge.size(), newEstimated_X_strong_edge.type(), cv::Scalar::all(0));
    cv::Mat median_blurred_est_Y = cv::Mat(newEstimated_X_strong_edge.size(), newEstimated_X_strong_edge.type(), cv::Scalar::all(0));
    cv::Mat median_blurred_est_Z = cv::Mat(newEstimated_X_strong_edge.size(), newEstimated_X_strong_edge.type(), cv::Scalar::all(0));

    std::vector<cv::Mat> estimatedImages_real_grid_strong_edges = {newEstimated_X_strong_edge, newEstimated_Y_strong_edge, newEstimated_Z_strong_edge};
    std::vector<cv::Mat> estimated_image_median_blurred = {median_blurred_est_X, median_blurred_est_Y, median_blurred_est_Z};
    int median_filt_kernel_size = 3;

    for (size_t i = 0; i < estimatedImages_real_grid_strong_edges.size(); i++)
    {
        cv::medianBlur(estimatedImages_real_grid_strong_edges[i], estimated_image_median_blurred[i], median_filt_kernel_size);
    }

    resheped_rows = median_blurred_est_X.rows * median_blurred_est_X.cols;
    guessed_X_real_post = estimated_image_median_blurred[0].reshape(1, resheped_rows);
    guessed_Y_real_post = estimated_image_median_blurred[1].reshape(1, resheped_rows);
    guessed_Z_real_post = estimated_image_median_blurred[2].reshape(1, resheped_rows);
    cv::Mat merged_mat_XYZ_post_strong_edges;
    cv::hconcat(guessed_X_real_post, guessed_Y_real_post, merged_mat_XYZ_post_strong_edges);
    cv::hconcat(merged_mat_XYZ_post_strong_edges, guessed_Z_real_post, merged_mat_XYZ_post_strong_edges);

    io::writeMatrixToFile(merged_mat_XYZ_post_strong_edges, FILESPATH + "XYZ_guessed_median_grid_strong_edges_02.txt");

    // Soft Edge
    cv::Mat newEstimated_disp_soft_edge = real_grid_estimations_testing.getEstimations_disp_soft_edge();
    cv::Mat newEstimated_X_soft_edge = real_grid_estimations_testing.getEstimations_X_soft_edge();
    cv::Mat newEstimated_Y_soft_edge = real_grid_estimations_testing.getEstimations_Y_soft_edge();
    cv::Mat newEstimated_Z_soft_edge = real_grid_estimations_testing.getEstimations_Z_soft_edge();

    std::vector<cv::Mat> estimatedImages_real_grid_soft_edges = {newEstimated_X_soft_edge, newEstimated_Y_soft_edge, newEstimated_Z_soft_edge};

    for (size_t i = 0; i < estimatedImages_real_grid_soft_edges.size(); i++)
    {
        cv::medianBlur(estimatedImages_real_grid[i], estimated_image_median_blurred[i], median_filt_kernel_size);
    }

    resheped_rows = median_blurred_est_X.rows * median_blurred_est_X.cols;
    guessed_X_real_post = estimated_image_median_blurred[0].reshape(1, resheped_rows);
    guessed_Y_real_post = estimated_image_median_blurred[1].reshape(1, resheped_rows);
    guessed_Z_real_post = estimated_image_median_blurred[2].reshape(1, resheped_rows);
    cv::Mat merged_mat_XYZ_post_soft_edges;
    cv::hconcat(guessed_X_real_post, guessed_Y_real_post, merged_mat_XYZ_post_soft_edges);
    cv::hconcat(merged_mat_XYZ_post_soft_edges, guessed_Z_real_post, merged_mat_XYZ_post_soft_edges);

    io::writeMatrixToFile(merged_mat_XYZ_post_soft_edges, FILESPATH + "XYZ_guessed_median_real_grid_soft_edges_02.txt");

    return EXIT_SUCCESS;
}
