#include "estimation_functions_simulated_grid.h"
#include "support_structures_simulated_grid.h"
#include "common_structures.h"

void SimulatedGridEstimations::setEstimationParameters(const cv::Mat& camera_matrix_left_, 
	const cv::Mat& camera_matrix_right_, 
	const int& sampling_factor_, 
	const double& disparity_edge_threshold_, 
	const double& baseline_, 
	const double& doffs_)
{
	camera_matrix_left = camera_matrix_left_;
	camera_matrix_right = camera_matrix_right_;
	sampling_factor = sampling_factor_;
	disparity_edge_threshold = disparity_edge_threshold_;
	step = double(1.0 / sampling_factor_);
	baseline = baseline_;
	doffs = doffs_;

	CostPenalties cost_penalties;
	p_0 = cost_penalties.P_0;
	p_1 = cost_penalties.P_1;
	p_2 = cost_penalties.P_2;

}

void SimulatedGridEstimations::setEstimatedValues(const cv::Mat& complete_data_matrix, 
	const cv::Mat& lookup_matrix, 
	const cv::Mat& image_left, 
	const cv::Mat& image_right, 
	std::vector<TotalDerivatives>& derivative_vector, 
	const cv::Mat& bad_points)
{
	// Estimation matrix and vector initialization
	int cols = (sampling_factor + 1) * (sampling_factor + 1);
	int rows = (lookup_matrix.rows - 1) * (lookup_matrix.cols - 1);
	cv::Size matrix_size = cv::Size(cols, rows);
	estimated_disparity = cv::Mat(matrix_size, CV_64F, cv::Scalar::all(-1.0));
	estimated_X = cv::Mat(matrix_size, CV_64F, cv::Scalar::all(-1.0));
	estimated_Y = cv::Mat(matrix_size, CV_64F, cv::Scalar::all(-1.0));
	estimated_Z = cv::Mat(matrix_size, CV_64F, cv::Scalar::all(-1.0));

	matrix_limits_l = defineMatrixLimits(0, image_left.rows, 0, image_left.cols).t();
	matrix_limits_r = defineMatrixLimits(0, image_right.rows, 0, image_right.cols).t();

	// Corner indexes 
	int TL{ 0 };
	int TR{ 0 };
	int BL{ 0 };
	int BR{ 0 };

	// Indexes for final matrices
	int d = 0;
	int r = 0;

	int best_value_index = 0;
	double best_value = 0.0;

	for (int i = 0; i < lookup_matrix.rows - 1; i++)
	{
		for (int j = 0; j < lookup_matrix.cols - 1; j++)
		{
			// Corners of the subwindow
			TL = lookup_matrix.at<int>(i, j);
			TR = lookup_matrix.at<int>(i, j + 1);
			BL = lookup_matrix.at<int>(i + 1, j);
			BR = lookup_matrix.at<int>(i + 1, j + 1);

			// Check if there's a vertical or an horizontal edge between the corner points
			// This is done in order to adapt the following strategy for the subsampling
			d = 0;
			EdgeDirection ed = EdgeDirection::undefined;
			isEdgeSimulatedGrid(TL, TR, BL, BR, complete_data_matrix, ed, disparity_edge_threshold);
			if (isEdgeSimulatedGrid(TL, TR, BL, BR, complete_data_matrix, ed, disparity_edge_threshold) == false)
			{
				// There is no edge 
				// Threfore make the estimations over all the small window 
				for (double ii = 0.0; ii <= 1.0; ii += step)
				{
					for (double jj = 0.0; jj <= 1.0; jj += step)
					{
						estimations_vector.clear();
						int_pixel_pos_left.clear();
						int_pixel_pos_right.clear();
						corr_pixel_diff.clear();
						noEdgesBestEstimation(TL, TR, BL, BR, ii, jj, 
							image_left,
							image_right,
							complete_data_matrix, 
							lookup_matrix,
							bad_points,
							derivative_vector);

						// Evaluate the best estimation
						if (corr_pixel_diff.size() != 0)
						{
							best_value_index = std::min_element(corr_pixel_diff.begin(), corr_pixel_diff.end()) - corr_pixel_diff.begin();
							best_value = estimations_vector[best_value_index].disp;
							estimated_disparity.at<double>(r, d) = best_value;
							estimated_X.at<double>(r, d) = estimations_vector[best_value_index].X_mt;
							estimated_Y.at<double>(r, d) = estimations_vector[best_value_index].Y_mt;
							estimated_Z.at<double>(r, d) = baseline * camera_matrix_left.at<double>(0, 0) / (estimated_disparity.at<double>(r, d) + doffs);
						}
						d++;
					}
				}
			}
			else
			{
				estimations_vector.clear();
				corr_pixel_diff.clear();
				int_pixel_pos_left.clear();
				int_pixel_pos_right.clear();

				estimations_vector.resize(4);
				corr_pixel_diff.resize(4);
				int_pixel_pos_left.resize(4);
				int_pixel_pos_right.resize(4);

				// There is an edge
				// With the derivative it is possible to identify if vertical or horizontal
				if (ed == EdgeDirection::vertical)
				{
					// Vertical edge
					// Penalty if estimation different from closer corner point considering the edge
					for (double ii = 0.0; ii <= 1.0; ii += step)
					{
						for (double jj = 0.0; jj <= 1.0; jj += step)
						{
							yesEdgesBestEstimation(TL, TR, BL, BR, ii, jj,
								image_left,
								image_right,
								complete_data_matrix,
								lookup_matrix,
								derivative_vector);
							
							corr_pixel_diff[0] = corr_pixel_diff[0] + jj * p_2 + ii * p_1;
							corr_pixel_diff[1] = corr_pixel_diff[1] + (1.0 - jj) * p_2 + ii * p_1;
							corr_pixel_diff[2] = corr_pixel_diff[2] + jj * p_2 + (1.0 - ii) * p_1;
							corr_pixel_diff[3] = corr_pixel_diff[3] + (1.0 - jj) * p_2 + (1.0 - ii) * p_1;
							// Evaluate the best estimation
							best_value_index = std::min_element(corr_pixel_diff.begin(), corr_pixel_diff.end()) - corr_pixel_diff.begin();
							best_value = estimations_vector[best_value_index].disp;
							estimated_disparity.at<double>(r, d) = best_value;
							estimated_X.at<double>(r, d) = estimations_vector[best_value_index].X_mt;
							estimated_Y.at<double>(r, d) = estimations_vector[best_value_index].Y_mt;
							estimated_Z.at<double>(r, d) = baseline * camera_matrix_left.at<double>(0, 0) / (estimated_disparity.at<double>(r, d) + doffs);
							d++;
						}
					}
				}
				else if (ed == EdgeDirection::horizontal)
				{
					// Horizontal edge
					// Penalty if estimation different from closer corner point considering the edge
					for (double ii = 0.0; ii <= 1.0; ii += step)
					{
						for (double jj = 0.0; jj <= 1.0; jj += step)
						{
							yesEdgesBestEstimation(TL, TR, BL, BR, ii, jj,
								image_left,
								image_right,
								complete_data_matrix,
								lookup_matrix,
								derivative_vector);

							corr_pixel_diff[0] = corr_pixel_diff[0] + jj * p_1 + ii * p_2;
							corr_pixel_diff[1] = corr_pixel_diff[1] + (1.0 - jj) * p_1 + ii * p_2;
							corr_pixel_diff[2] = corr_pixel_diff[2] + jj * p_1 + (1.0 - ii) * p_2;
							corr_pixel_diff[3] = corr_pixel_diff[3] + (1.0 - jj) * p_1 + (1.0 - ii) * p_2;

							// Evaluate the best estimation
							best_value_index = std::min_element(corr_pixel_diff.begin(), corr_pixel_diff.end()) - corr_pixel_diff.begin();
							best_value = estimations_vector[best_value_index].disp;
							estimated_disparity.at<double>(r, d) = best_value;
							estimated_X.at<double>(r, d) = estimations_vector[best_value_index].X_mt;
							estimated_Y.at<double>(r, d) = estimations_vector[best_value_index].Y_mt;
							estimated_Z.at<double>(r, d) = baseline * camera_matrix_left.at<double>(0, 0) / (estimated_disparity.at<double>(r, d) + doffs);
							d++;
						}
					}
				}
				else if (ed == EdgeDirection::undefined)
				{
					// Undefined edge
					// Penalty if estimation different from closer corner point considering the edge
					for (double ii = 0.0; ii <= 1.0; ii += step)
					{
						for (double jj = 0.0; jj <= 1.0; jj += step)
						{
							yesEdgesBestEstimation(TL, TR, BL, BR, ii, jj,
								image_left,
								image_right,
								complete_data_matrix,
								lookup_matrix,
								derivative_vector);

							corr_pixel_diff[0] = corr_pixel_diff[0] + jj * p_0 + ii * p_0;
							corr_pixel_diff[1] = corr_pixel_diff[1] + (1.0 - jj) * p_0 + ii * p_0;
							corr_pixel_diff[2] = corr_pixel_diff[2] + jj * p_0 + (1.0 - ii) * p_0;
							corr_pixel_diff[3] = corr_pixel_diff[3] + (1.0 - jj) * p_0 + (1.0 - ii) * p_0;


							// Evaluate the best estimation
							best_value_index = std::min_element(corr_pixel_diff.begin(), corr_pixel_diff.end()) - corr_pixel_diff.begin();
							best_value = estimations_vector[best_value_index].disp;
							estimated_disparity.at<double>(r, d) = best_value;
							estimated_X.at<double>(r, d) = estimations_vector[best_value_index].X_mt;
							estimated_Y.at<double>(r, d) = estimations_vector[best_value_index].Y_mt ;
							estimated_Z.at<double>(r, d) = baseline * camera_matrix_left.at<double>(0, 0) / (estimated_disparity.at<double>(r, d) + doffs);
							d++;
						}
					}
				}
			}
			r++;
		}
	}
}

void SimulatedGridEstimations::setGuessedValues(const cv::Mat& complete_data_matrix, const cv::Mat& lookup_matrix)
{
	// Matrix initialization
	int totrows = (lookup_matrix.rows - 1) * sampling_factor + 1;
	int totcols = (lookup_matrix.cols - 1) * sampling_factor + 1;
	cv::Size matrix_size = cv::Size(totcols, totrows);
	guessed_disparity = cv::Mat(matrix_size, CV_64F, cv::Scalar::all(0));
	guessed_X = cv::Mat(matrix_size, CV_64F, cv::Scalar::all(-1.0));
	guessed_Y = cv::Mat(matrix_size, CV_64F, cv::Scalar::all(-1.0));
	guessed_Z = cv::Mat(matrix_size, CV_64F, cv::Scalar::all(DBL_MAX));

	int m = 0;
	int n = 0;
	int h = 0;
	int k = 0;

	int indx = 0;

	for (int i = 0; i < guessed_disparity.rows - sampling_factor; i += sampling_factor)
	{
		n = 0;
		for (int j = 0; j < guessed_disparity.cols - sampling_factor; j += sampling_factor)
		{
			k = 0;
			for (int ii = i; ii <= i + sampling_factor; ii++)
			{
				for (int jj = j; jj <= j + sampling_factor; jj++)
				{
					if (estimated_disparity.at<double>(h, k) != -1.0)
					{
						guessed_disparity.at<double>(ii, jj) = estimated_disparity.at<double>(h, k);
						guessed_Z.at<double>(ii, jj) = estimated_Z.at<double>(h, k);
						guessed_X.at<double>(ii, jj) = estimated_X.at<double>(h, k);
						guessed_Y.at<double>(ii, jj) = estimated_Y.at<double>(h, k);
					}
					k++;
				}
				indx = lookup_matrix.at<int>(m, n);
				guessed_disparity.at<double>(i, j) = complete_data_matrix.at<double>(indx, 5);
				guessed_Z.at<double>(i, j) = complete_data_matrix.at<double>(indx, 2);
				guessed_X.at<double>(i, j) = complete_data_matrix.at<double>(indx, 0);
				guessed_Y.at<double>(i, j) = complete_data_matrix.at<double>(indx, 1);
			}
			n++;
			h++;
		}
		m++;
	}
}

cv::Mat SimulatedGridEstimations::getGuessedDisparity()
{
	return guessed_disparity;
}

cv::Mat SimulatedGridEstimations::getGuessedZ()
{
	return guessed_Z;
}

cv::Mat SimulatedGridEstimations::getGuessedX()
{
	return guessed_X;
}

cv::Mat SimulatedGridEstimations::getGuessedY()
{
	return guessed_Y;
}

void SimulatedGridEstimations::noEdgesBestEstimation(const int& TL, const int& TR, 
	const int& BL, const int& BR, 
	const double& indx_r, const double& indx_c, 
	const cv::Mat& left_image, 
	const cv::Mat& right_image,
	const cv::Mat& complete_data_matrix, 
	const cv::Mat& lookup_table, 
	const cv::Mat& bad_points,
	std::vector<TotalDerivatives>& derivative_vector)
{
	if (bad_points.at<int>(TL, 0) == 0) {
		ObservationData est_1 = lookupIndexToObservation(TL, complete_data_matrix) +
			derivative_vector[TL].der_left * indx_c +
			derivative_vector[TL].der_top * indx_r;
		estimations_vector.push_back(est_1);
		int_pixel_pos_left.push_back(fromObsToPoint2i(est_1, camera_matrix_left));
		int_pixel_pos_right.push_back(int_pixel_pos_left.back() - cv::Point2i(static_cast<int>(est_1.disp), 0));
	}
	if (bad_points.at<int>(TR, 0) == 0)
	{
		ObservationData est_2 = lookupIndexToObservation(TR, complete_data_matrix) +
			derivative_vector[TR].der_right * (1.0 - indx_c) +
			derivative_vector[TR].der_top * indx_r;
		estimations_vector.push_back(est_2);
		int_pixel_pos_left.push_back(fromObsToPoint2i(est_2, camera_matrix_left));
		int_pixel_pos_right.push_back(int_pixel_pos_left.back() - cv::Point2i(static_cast<int>(est_2.disp), 0));
	}
	if (bad_points.at<int>(BL, 0) == 0)
	{
		ObservationData est_3 = lookupIndexToObservation(BL, complete_data_matrix) +
			derivative_vector[BL].der_left * indx_c +
			derivative_vector[BL].der_bot * (1.0 - indx_r);
		estimations_vector.push_back(est_3);
		int_pixel_pos_left.push_back(fromObsToPoint2i(est_3, camera_matrix_left));
		int_pixel_pos_right.push_back(int_pixel_pos_left.back() - cv::Point2i(static_cast<int>(est_3.disp), 0));
	}
	if (bad_points.at<int>(BR, 0) == 0)
	{
		ObservationData est_4 = lookupIndexToObservation(BR, complete_data_matrix) +
			derivative_vector[BR].der_right * (1.0 - indx_c) +
			derivative_vector[BR].der_bot * (1.0 - indx_r);
		estimations_vector.push_back(est_4);
		int_pixel_pos_left.push_back(fromObsToPoint2i(est_4, camera_matrix_left));
		int_pixel_pos_right.push_back(int_pixel_pos_left.back() - cv::Point2i(static_cast<int>(est_4.disp), 0));
	}
	
	for (int i = 0; i < int_pixel_pos_left.size(); i++)
	{
		if (isPixelInsideImage(int_pixel_pos_left[i], int_pixel_pos_right[i], matrix_limits_l, matrix_limits_r) == true)
		{
			corr_pixel_diff.push_back(abs(left_image.at<uchar>(int_pixel_pos_left[i].y, int_pixel_pos_left[i].x) -
				right_image.at<uchar>(int_pixel_pos_right[i].y, int_pixel_pos_right[i].x)));
		}
		else
		{
			corr_pixel_diff.push_back(DBL_MAX);
		}
	}
}

void SimulatedGridEstimations::yesEdgesBestEstimation(const int& TL, const int& TR, 
	const int& BL, const int& BR, 
	const double& indx_r, const double& indx_c, 
	const cv::Mat& left_image, 
	const cv::Mat& right_image, 
	const cv::Mat& complete_data_matrix, 
	const cv::Mat& lookup_table, 
	std::vector<TotalDerivatives>& derivative_vector)
{
	std::fill(corr_pixel_diff.begin(), corr_pixel_diff.end(), DBL_MAX);
	estimations_vector[0] = lookupIndexToObservation(TL, complete_data_matrix) +
		derivative_vector[TL].der_left * indx_c +
		derivative_vector[TL].der_top * indx_r;
	estimations_vector[1] = lookupIndexToObservation(TR, complete_data_matrix) +
		derivative_vector[TR].der_right * (1.0 - indx_c) +
		derivative_vector[TR].der_top * indx_r;
	estimations_vector[2] = lookupIndexToObservation(BL, complete_data_matrix) +
		derivative_vector[BL].der_left * indx_c +
		derivative_vector[BL].der_bot * (1.0 - indx_r);
	estimations_vector[3] = lookupIndexToObservation(BR, complete_data_matrix) +
		derivative_vector[BR].der_right * (1.0 - indx_c) +
		derivative_vector[BR].der_bot * (1.0 - indx_r);

	for (int i = 0; i < estimations_vector.size(); i++)
	{
		int_pixel_pos_left[i] = fromObsToPoint2i(estimations_vector[i], camera_matrix_left);
		int_pixel_pos_right[i] = int_pixel_pos_left[i] - cv::Point2i(static_cast<int>(estimations_vector[i].disp), 0);
	}
	
	if (isPixelInsideImage(int_pixel_pos_left[0], int_pixel_pos_right[0], matrix_limits_l, matrix_limits_r) == true) {
		// make row comparison evaluating the absolute difference between pixel's intensity values
		corr_pixel_diff[0] = abs(left_image.at<uchar>(int_pixel_pos_left[0].y, int_pixel_pos_left[0].x) -
			right_image.at<uchar>(int_pixel_pos_right[0].y, int_pixel_pos_right[0].x));
	}
	if (isPixelInsideImage(int_pixel_pos_left[1], int_pixel_pos_right[1], matrix_limits_l, matrix_limits_r) == true)
	{
		corr_pixel_diff[1] = abs(left_image.at<uchar>(int_pixel_pos_left[1].y, int_pixel_pos_left[1].x) -
			right_image.at<uchar>(int_pixel_pos_right[1].y, int_pixel_pos_right[1].x));
	}
	if (isPixelInsideImage(int_pixel_pos_left[2], int_pixel_pos_right[2], matrix_limits_l, matrix_limits_r) == true)
	{
		corr_pixel_diff[2] = abs(left_image.at<uchar>(int_pixel_pos_left[2].y, int_pixel_pos_left[2].x) -
			right_image.at<uchar>(int_pixel_pos_right[2].y, int_pixel_pos_right[2].x));
	}
	if (isPixelInsideImage(int_pixel_pos_left[3], int_pixel_pos_right[3], matrix_limits_l, matrix_limits_r) == true)
	{
		corr_pixel_diff[3] = abs(left_image.at<uchar>(int_pixel_pos_left[3].y, int_pixel_pos_left[3].x) -
			right_image.at<uchar>(int_pixel_pos_right[3].y, int_pixel_pos_right[3].x));
	}
}
