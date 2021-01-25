#pragma once

#include <opencv2/core/mat.hpp>

#include <support_functions_simulated_grid.h>

class SimulatedGridEstimations
{
public:
	// Setter
	void setEstimationParameters(const cv::Mat& camera_matrix_left_, 
		const cv::Mat& camera_matrix_right_, 
		const int& sampling_factor_, 
		const double& disparity_edge_threshold_, 
		const double& baseline_, 
		const double& doffs_);
	void setEstimatedValues(const cv::Mat& complete_data_matrix, 
		const cv::Mat& lookup_matrix, 
		const cv::Mat& image_left, 
		const cv::Mat& image_right, 
		std::vector<TotalDerivatives>& derivative_vector, 
		const cv::Mat& bad_points);
	void setGuessedValues(const cv::Mat& complete_data_matrix, const cv::Mat& lookup_matrix);

	// Getter
	cv::Mat getGuessedDisparity();
	cv::Mat getGuessedZ();
	cv::Mat getGuessedX();
	cv::Mat getGuessedY();

private:
	// Parameters 
	cv::Mat camera_matrix_left;
	cv::Mat camera_matrix_right;
	int sampling_factor = 0;
	double step = 0.0;
	double baseline = 0.0;
	double doffs = 0.0;
	double disparity_edge_threshold = 0.0;

	double p_0 = 0.0;
	double p_1 = 0.0;
	double p_2 = 0.0;

	// Internal vectors and matrices
	std::vector<ObservationData> estimations_vector;       // vector for storing the estimations for each subsamples
	std::vector<cv::Point2i> int_pixel_pos_left;           // vector for storing estimation's pixel position in the left image plane
	std::vector<cv::Point2i> int_pixel_pos_right;          // vector for storing estimation's pixel position in the right image plane
	std::vector<double> corr_pixel_diff;                   // vector for storing the cost (absolute difference between left and right pixel intensity)
	cv::Mat matrix_limits_l;                               // class istantiation for left image limits
	cv::Mat matrix_limits_r;						       // class istantiation for right image limits

	// Final estimation matrices
	cv::Mat estimated_disparity;
	cv::Mat estimated_X;
	cv::Mat estimated_Y;
	cv::Mat estimated_Z;

	// Final guessed matrices
	cv::Mat guessed_disparity;
	cv::Mat guessed_X;
	cv::Mat guessed_Y;
	cv::Mat guessed_Z;

	void noEdgesBestEstimation(const int& TL, const int& TR,
		const int& BL, const int& BR,
		const double& indx_r, const double& indx_c,
		const cv::Mat& left_image,
		const cv::Mat& right_image,
		const cv::Mat& complete_data_matrix,
		const cv::Mat& lookup_table,
		const cv::Mat& bad_points,
		std::vector<TotalDerivatives>& derivative_vector);

	void yesEdgesBestEstimation(const int& TL, const int& TR, 
		const int& BL, const int& BR, 
		const double& indx_r, const double& indx_c, 
		const cv::Mat& left_image,
		const cv::Mat& right_image,
		const cv::Mat& complete_data_matrix, 
		const cv::Mat& lookup_table, 
		std::vector<TotalDerivatives>& derivative_vector);

};