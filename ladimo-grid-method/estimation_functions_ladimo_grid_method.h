#pragma once

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

#include "common_structures.h"
#include "support_structures_ladimo_grid_method.h"
#include "support_functions_ladimo_grid_method.h"

class LadimoGridEstimations
{
public:
	// setter functions
	void setEstimationParameters(const cv::Mat& camera_matrix_left_, 
		const cv::Mat& camera_matrix_right_, 
		const double& baseline_, 
		const int& sampling_factor_);
	void setNecessaryVectors(const std::vector<GridSquare>& grid_squares_, 
		const std::vector<ObservationData>& grid_observations_, 
		const std::vector<CompleteDerivatives>& internal_derivatives_, 
		const std::vector<CompleteDerivatives>& external_derivatives_);
	void initilizeEstimationMatrices();
	void setEstimations(const cv::Mat& image_left, const cv::Mat& image_right);
	void setGuesses();

	// Getter
	cv::Mat getEstimations_disp() {
		return guessed_Disparity;
	}
	cv::Mat getEstimations_X() {
		return guessed_X;
	}
	cv::Mat getEstimations_Y() {
		return guessed_Y;
	}
	cv::Mat getEstimations_Z() {
		return guessed_Z;
	}
	cv::Mat getEstimations_disp_no_edge() {
		return guessed_Disparity_no_edge;
	}
	cv::Mat getEstimations_X_no_edge() {
		return guessed_X_no_edge;
	}
	cv::Mat getEstimations_Y_no_edge() {
		return guessed_Y_no_edge;
	}
	cv::Mat getEstimations_Z_no_edge() {
		return guessed_Z_no_edge;
	}

	cv::Mat getEstimations_disp_strong_edge() {
		return guessed_Disparity_strong_edge;
	}
	cv::Mat getEstimations_X_strong_edge() {
		return guessed_X_strong_edge;
	}
	cv::Mat getEstimations_Y_strong_edge() {
		return guessed_Y_strong_edge;
	}
	cv::Mat getEstimations_Z_strong_edge() {
		return guessed_Z_strong_edge;
	}
	cv::Mat getEstimations_disp_soft_edge() {
		return guessed_Disparity_soft_edge;
	}
	cv::Mat getEstimations_X_soft_edge() {
		return guessed_X_soft_edge;
	}
	cv::Mat getEstimations_Y_soft_edge() {
		return guessed_Y_soft_edge;
	}
	cv::Mat getEstimations_Z_soft_edge() {
		return guessed_Z_soft_edge;
	}

private:
	// Vectors to use
	std::vector<GridSquare> grid_squares{};
	std::vector<ObservationData> grid_observations{};
	std::vector<CompleteDerivatives> internal_derivatives{};
	std::vector<CompleteDerivatives> external_derivatives{};

	// Estimation computation related vectors
	std::vector<double> raw_differences{ 0.0, 0.0, 0.0, 0.0 };
	ObservationData final_estimastion_no_edge_internal_deriv{};

	double no_edge_disp = 0.0;
	double no_edge_Z = 0.0;


	// Internal Parameters
	cv::Mat left_image_limits{};
	cv::Mat right_image_limits{};
	int sampling_factor{ 4 };
	double step{ 0.25 };
	double baseline{ 74.0 };
	int window_size = 7;
	cv::Mat camera_matrix_left{};
	cv::Mat camera_matrix_right{};

	double P0{ 0.0 };
	double P1{ 0.0 };
	double P2{ 0.0 };
	double P3{ 0.0 };
	double P4{ 0.0 };

	// Final Complete Functions
	// Final estimation matrices
	cv::Mat estimated_Disparity;
	cv::Mat estimated_X;
	cv::Mat estimated_Y;
	cv::Mat estimated_Z;

	// Final guessed matrices
	cv::Mat guessed_Disparity;
	cv::Mat guessed_X;
	cv::Mat guessed_Y;
	cv::Mat guessed_Z;

	// Final estimation matrices
	cv::Mat estimated_Disparity_no_edge;
	cv::Mat estimated_X_no_edge;
	cv::Mat estimated_Y_no_edge;
	cv::Mat estimated_Z_no_edge;

	cv::Mat estimated_Disparity_strong_edge;
	cv::Mat estimated_X_strong_edge;
	cv::Mat estimated_Y_strong_edge;
	cv::Mat estimated_Z_strong_edge;

	cv::Mat estimated_Disparity_soft_edge;
	cv::Mat estimated_X_soft_edge;
	cv::Mat estimated_Y_soft_edge;
	cv::Mat estimated_Z_soft_edge;

	// Final guessed matrices
	cv::Mat guessed_Disparity_no_edge;
	cv::Mat guessed_X_no_edge;
	cv::Mat guessed_Y_no_edge;
	cv::Mat guessed_Z_no_edge;

	cv::Mat guessed_Disparity_strong_edge;
	cv::Mat guessed_X_strong_edge;
	cv::Mat guessed_Y_strong_edge;
	cv::Mat guessed_Z_strong_edge;

	cv::Mat guessed_Disparity_soft_edge;
	cv::Mat guessed_X_soft_edge;
	cv::Mat guessed_Y_soft_edge;
	cv::Mat guessed_Z_soft_edge;

	// Internal method

	bool findStrongEdges(const int& TL, const int& TR, 
		const int& BL, const int& BR, 
		EdgeShape& edge_shape);
	bool findSoftEdges(const int& TL, const int& TR, 
		const int& BL, const int& BR, 
		EdgeShape& edge_shape);

	void bilinearInterpolation(const int& TL, const int& TR, 
		const int& BL, const int& BR, 
		cv::Vec4d& sub_square_estimation);
	void noEdgesEstimations(const int& TL, const int& TR, 
		const int& BL, const int& BR, 
		const double& indx_r, const double& indx_c, 
		std::vector<ObservationData>& no_edge_est_vec);
	void strongEdgesEstimations(const int& TL, const int& TR, 
		const int& BL, const int& BR, 
		const double& indx_r, const double& indx_c, 
		const cv::Mat& im_left, const cv::Mat& im_right, 
		std::vector<ObservationData>& strong_edge_est_vec);
	void softEdgesEstimations(const int& TL, const int& TR, 
		const int& BL, const int& BR,
		const double& indx_r, const double& indx_c, 
		const cv::Mat& im_left, const cv::Mat& im_right, 
		std::vector<ObservationData>& soft_edge_est_vec);
};