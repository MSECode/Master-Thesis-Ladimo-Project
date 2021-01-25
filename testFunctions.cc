#include "testFunctions.h"

/*
void estimatedDisparityTest(const cv::Mat& lookup_table, const cv::Mat& XYZd_data, const std::vector<TotalDerivatives>& derivative_vector, const int& index_row, const int& index_col, const int sampl_fact)
{
	float step = float(1.f / sampl_fact);
	cv::Mat estimation_matrix_1 = cv::Mat(sampl_fact + 1, sampl_fact + 1, CV_32F);
	cv::Mat estimation_matrix_2 = cv::Mat(sampl_fact + 1, sampl_fact + 1, CV_32F);
	cv::Mat estimation_matrix_3 = cv::Mat(sampl_fact + 1, sampl_fact + 1, CV_32F);
	cv::Mat estimation_matrix_4 = cv::Mat(sampl_fact + 1, sampl_fact + 1, CV_32F);

	const int TL = lookup_table.at<int>(index_row, index_col);
	const int TR = lookup_table.at<int>(index_row, index_col + 1);
	const int BL = lookup_table.at<int>(index_row + 1, index_col);
	const int BR = lookup_table.at<int>(index_row + 1, index_col + 1);

	float ii = 0;

	for (float i = 0; i < estimation_matrix_1.rows; i++)
	{
		float jj = 0;
		for (float j = 0; j < estimation_matrix_1.cols; j++)
		{
			cv::Vec4f est_1 = fromBigMat(TL, XYZd_data) + jj * derivative_vector[TL].der_left + ii * derivative_vector[TL].der_top;
			cv::Vec4f est_2 = fromBigMat(TR, XYZd_data) + jj * derivative_vector[TR].der_right + ii * derivative_vector[TR].der_top;
			cv::Vec4f est_3 = fromBigMat(BL, XYZd_data) + jj * derivative_vector[BL].der_left + ii * derivative_vector[BL].der_bot;
			cv::Vec4f est_4 = fromBigMat(BR, XYZd_data) + jj * derivative_vector[BR].der_right + ii * derivative_vector[BR].der_bot;
			
			estimation_matrix_1.at<float>(i, j) = est_1[3];
			estimation_matrix_2.at<float>(i, j) = est_2[3];
			estimation_matrix_3.at<float>(i, j) = est_3[3];
			estimation_matrix_4.at<float>(i, j) = est_4[3];
			
			jj += step;
		}
		ii += step;
	}

	std::cout << "TL estimations" << std::endl;
	std::cout << estimation_matrix_1 << std::endl;

	std::cout << "TR estimations" << std::endl;
	std::cout << estimation_matrix_2 << std::endl;

	std::cout << "BL estimations" << std::endl;
	std::cout << estimation_matrix_3 << std::endl;

	std::cout << "BR estimations" << std::endl;
	std::cout << estimation_matrix_4 << std::endl;

}

void guessedDisparityTest(int image_row, int image_col, const cv::Mat& ground_truth, const cv::Mat& guessed_disparity, int& sampl_factor, int& gap)
{
	int x_pos = (image_col / gap) * sampl_factor;
	int y_pos = (image_row / gap) * sampl_factor;
	
	cv::Rect estimations_patch = cv::Rect(x_pos, y_pos, sampl_factor + 1, sampl_factor + 1);
	cv::Rect real_patch = cv::Rect(image_col, image_row, gap, gap);

	cv::Mat roi_estimation = guessed_disparity(estimations_patch);
	cv::Mat roi_ground_truth = ground_truth(real_patch);
	cv::resize(roi_ground_truth, roi_ground_truth, roi_ground_truth.size()/5);


	std::cout << " Estimated disparities" << std::endl;
	std::cout << roi_estimation << std::endl;

	std::cout << " Exact disparities" << std::endl;
	std::cout << roi_ground_truth << std::endl;
}


void getMinimumCost(const cv::Mat& cost_cube, const std::vector<float>& disparity_levels, cv::Mat& input_disparity)
{
	int channels = cost_cube.channels();
	std::vector<int> temp_z_elements_vec(channels);
	for (int i = 0; i < cost_cube.rows; i++)
	{
		for (int j = 0; j < cost_cube.cols; j++)
		{
			for (int ch = 0; ch < channels; ch++)
			{
				temp_z_elements_vec[ch] = cost_cube.ptr<float>(i)[channels * j + ch];
			}

			int disparity_index = std::min_element(temp_z_elements_vec.begin(), temp_z_elements_vec.end()) - temp_z_elements_vec.begin();
			input_disparity.at<float>(i, j) = disparity_levels[disparity_index];
			//std::cout << input_disparity.at<float>(i, j) << std::endl;
		}
	}

}




void sgmTest(const cv::Mat& lookupTable, const cv::Mat& xyd_data, const cv::Mat& XYZ_data, const cv::Mat& left_image, const cv::Mat& right_image, const int& gap, const int& lookup_r, const int& lookup_c, std::string& path_test_matrices)
{
	std::vector<cv::Vec2i> pixel_positions_left_vector(4);
	std::vector<float> disparity_levels_vector(4);
	cv::Mat patch_cost_cube = cv::Mat(gap + 1, gap + 1, CV_32FC(4));
	EstimationParameters estimation_parameters;
	int window_size = estimation_parameters.setWindowSize();
	float edge_threshold = estimation_parameters.setDispThreshold();

	// Corners of the window
	const int TL = lookupTable.at<int>(lookup_r, lookup_c);
	const int TR = lookupTable.at<int>(lookup_r, lookup_c + 1);
	const int BL = lookupTable.at<int>(lookup_r + 1, lookup_c);
	const int BR = lookupTable.at<int>(lookup_r + 1, lookup_c + 1);

	pixel_positions_left_vector = GetLeftPixelPositions(TL, TR, BL, BR, xyd_data, pixel_positions_left_vector);
	disparity_levels_vector = GetDisparityLevels(TL, TR, BL, BR, xyd_data, disparity_levels_vector);
	patch_cost_cube = getSGMCostCube(pixel_positions_left_vector, disparity_levels_vector, left_image, right_image, window_size, gap, patch_cost_cube);

	
	std::cout << " | " << " TL disp " << disparity_levels_vector[0] << " | " <<
		" TR disp " << disparity_levels_vector[1] << " | " <<
		" BL disp " << disparity_levels_vector[2] << " | " <<
		" BR disp " << disparity_levels_vector[3] << " | " << std::endl;

	cv::Mat ch1, ch2, ch3, ch4;
	std::vector<cv::Mat> channels(patch_cost_cube.channels());

	cv::Mat disparity_output = cv::Mat(patch_cost_cube.size(), CV_32F, cv::Scalar(0));


	EdgeDirection edg = EdgeDirection::undefined;

	if (findEdge_deriv(TL, TR, BL, BR, XYZ_data, edg, edge_threshold) == false)
	{
		cv::split(patch_cost_cube, channels);
		ch1 = channels[0];
		ch2 = channels[1];
		ch3 = channels[2];
		ch4 = channels[3];

		io::writeMatrixToFile(ch1, path_test_matrices + "channel 1_N.txt");
		io::writeMatrixToFile(ch2, path_test_matrices + "channel 2_N.txt");
		io::writeMatrixToFile(ch3, path_test_matrices + "channel 3_N.txt");
		io::writeMatrixToFile(ch4, path_test_matrices + "channel 4_N.txt");

		
		getMinimumCost(patch_cost_cube, disparity_levels_vector, disparity_output);

		io::writeMatrixToFile(disparity_output, path_test_matrices + "disparity_N.txt");

	}
	else
	{
		float P1 = 5;
		float P2 = 40;

		if (edg == EdgeDirection::vertical)
		{
			cv::Mat direction_cost_cube_0 = cv::Mat(patch_cost_cube.size(), patch_cost_cube.type(), cv::Scalar(0));
			cv::Mat direction_cost_cube_2 = cv::Mat(patch_cost_cube.size(), patch_cost_cube.type(), cv::Scalar(0));
			cv::Mat total_cost_cube = cv::Mat(patch_cost_cube.size(), patch_cost_cube.type(), cv::Scalar(0));

			// Direction 0 (FROM LEFT)
			// Calculate the cost for the left borders pixel
			aggregationCostDirection0(patch_cost_cube, direction_cost_cube_0, P1, P2);

			// Direction 2 (FROM RIGHT)
			// Calculate the cost for the right borders pixel
			aggregationCostDirection2(patch_cost_cube, direction_cost_cube_2, P1, P2);

			total_cost_cube = direction_cost_cube_0 + direction_cost_cube_2;

			cv::split(total_cost_cube, channels);

			ch1 = channels[0];
			ch2 = channels[1];
			ch3 = channels[2];
			ch4 = channels[3];

			io::writeMatrixToFile(ch1, path_test_matrices + "channel 1_V.txt");
			io::writeMatrixToFile(ch2, path_test_matrices + "channel 2_V.txt");
			io::writeMatrixToFile(ch3, path_test_matrices + "channel 3_V.txt");
			io::writeMatrixToFile(ch4, path_test_matrices + "channel 4_V.txt");

			getMinimumCost(total_cost_cube, disparity_levels_vector, disparity_output);
			io::writeMatrixToFile(disparity_output, path_test_matrices + "disparity_V.txt");

		}
		else if (edg == EdgeDirection::horizontal)
		{
			cv::Mat direction_cost_cube_1 = cv::Mat(patch_cost_cube.size(), patch_cost_cube.type(), cv::Scalar(0));
			cv::Mat direction_cost_cube_3 = cv::Mat(patch_cost_cube.size(), patch_cost_cube.type(), cv::Scalar(0));
			cv::Mat total_cost_cube = cv::Mat(patch_cost_cube.size(), patch_cost_cube.type(), cv::Scalar(0));

			// Direction 1 (FROM TOP)
			// Calculate the cost for the top borders pixel
			aggregationCostDirection1(patch_cost_cube, direction_cost_cube_1, P1, P2);

			// Direction 3 (FROM BOT)
			// Calculate the cost for the bottom borders pixel
			aggregationCostDirection3(patch_cost_cube, direction_cost_cube_3, P1, P2);

			total_cost_cube = direction_cost_cube_1 + direction_cost_cube_3;

			cv::split(total_cost_cube, channels);
			
			ch1 = channels[0];
			ch2 = channels[1];
			ch3 = channels[2];
			ch4 = channels[3];

			io::writeMatrixToFile(ch1, path_test_matrices + "channel 1_H.txt");
			io::writeMatrixToFile(ch2, path_test_matrices + "channel 2_H.txt");
			io::writeMatrixToFile(ch3, path_test_matrices + "channel 3_H.txt");
			io::writeMatrixToFile(ch4, path_test_matrices + "channel 4_H.txt");

			getMinimumCost(total_cost_cube, disparity_levels_vector, disparity_output);

			io::writeMatrixToFile(disparity_output, path_test_matrices + "disparity_H.txt");

		}
		else if (edg == EdgeDirection::undefined)
		{
			cv::Mat direction_cost_cube_0 = cv::Mat(patch_cost_cube.size(), patch_cost_cube.type(), cv::Scalar(0));
			cv::Mat direction_cost_cube_2 = cv::Mat(patch_cost_cube.size(), patch_cost_cube.type(), cv::Scalar(0));
			cv::Mat direction_cost_cube_1 = cv::Mat(patch_cost_cube.size(), patch_cost_cube.type(), cv::Scalar(0));
			cv::Mat direction_cost_cube_3 = cv::Mat(patch_cost_cube.size(), patch_cost_cube.type(), cv::Scalar(0));
			cv::Mat total_cost_cube = cv::Mat(patch_cost_cube.size(), patch_cost_cube.type(), cv::Scalar(0));

			// Direction 0 (FROM LEFT)
			// Calculate the cost for the left borders pixel
			aggregationCostDirection0(patch_cost_cube, direction_cost_cube_0, P1, P2);

			// Direction 1 (FROM TOP)
			// Calculate the cost for the top borders pixel
			aggregationCostDirection1(patch_cost_cube, direction_cost_cube_1, P1, P2);

			// Direction 2 (FROM RIGHT)
			// Calculate the cost for the right borders pixel
			aggregationCostDirection2(patch_cost_cube, direction_cost_cube_2, P1, P2);

			// Direction 3 (FROM BOT)
			// Calculate the cost for the bottom borders pixel
			aggregationCostDirection3(patch_cost_cube, direction_cost_cube_3, P1, P2);

			total_cost_cube = direction_cost_cube_0 + direction_cost_cube_1 + direction_cost_cube_2 + direction_cost_cube_3;

			cv::split(total_cost_cube, channels);

			ch1 = channels[0];
			ch2 = channels[1];
			ch3 = channels[2];
			ch4 = channels[3];

			io::writeMatrixToFile(ch1, path_test_matrices + "channel 1_U.txt");
			io::writeMatrixToFile(ch2, path_test_matrices + "channel 2_U.txt");
			io::writeMatrixToFile(ch3, path_test_matrices + "channel 3_U.txt");
			io::writeMatrixToFile(ch4, path_test_matrices + "channel 4_U.txt");

			
			getMinimumCost(total_cost_cube, disparity_levels_vector, disparity_output);

			io::writeMatrixToFile(disparity_output, path_test_matrices + "disparity_U.txt");
		}
	}
}
*/

void RealGridOperationsTesting::setEstimationParameters(const cv::Mat& camera_matrix_left_, const cv::Mat& camera_matrix_right_, const float& baseline_, const int& sampling_factor_)
{
	camera_matrix_left = camera_matrix_left_;
	camera_matrix_right = camera_matrix_right_;
	sampling_factor = sampling_factor_;
	step = static_cast<float>(1.f / sampling_factor_);
	baseline = baseline_;

	CostPenalties cost_penalties;
	P0 = cost_penalties.P_0;
	P1 = cost_penalties.P_1;
	P2 = cost_penalties.P_2;
	P3 = cost_penalties.P_3;
	P4 = cost_penalties.P_4;

	P1_sgm = cost_penalties.P1_sgm;
	P2_sgm = cost_penalties.P2_sgm;
}

void RealGridOperationsTesting::setNecessaryVectors(const std::vector<GridSquare>& gridSquaresVector_, const std::vector<ObservationData>& observationsVector_, const std::vector<CompleteValuesDerivatives>& internalCompleteDerivativesVector_, const std::vector<CompleteValuesDerivatives>& esternalCompleteDerivativesVector_)
{
	gridSquaresVector = gridSquaresVector_;
	observationsVector = observationsVector_;
	internalCompleteDerivativesVector = internalCompleteDerivativesVector_;
	esternalCompleteDerivativesVector = esternalCompleteDerivativesVector_;
}

void RealGridOperationsTesting::initilizeEstimationMatrices()
{
	int cols_est = (sampling_factor + 1) * (sampling_factor + 1);
	int rows_est = gridSquaresVector.size();
	cv::Size matrix_size_est = cv::Size(cols_est, rows_est);
	estimated_Disparity = cv::Mat(matrix_size_est, CV_32F, cv::Scalar::all(0));
	estimated_X = cv::Mat(matrix_size_est, CV_32F, cv::Scalar::all(0));
	estimated_Y = cv::Mat(matrix_size_est, CV_32F, cv::Scalar::all(0));
	estimated_Z = cv::Mat(matrix_size_est, CV_32F, cv::Scalar::all(0));

	estimated_Disparity_no_edge = cv::Mat(matrix_size_est, CV_32F, cv::Scalar::all(0));
	estimated_X_no_edge = cv::Mat(matrix_size_est, CV_32F, cv::Scalar::all(0));
	estimated_Y_no_edge = cv::Mat(matrix_size_est, CV_32F, cv::Scalar::all(0));
	estimated_Z_no_edge = cv::Mat(matrix_size_est, CV_32F, cv::Scalar::all(0));

	estimated_Disparity_strong_edge = cv::Mat(matrix_size_est, CV_32F, cv::Scalar::all(0));
	estimated_X_strong_edge = cv::Mat(matrix_size_est, CV_32F, cv::Scalar::all(0));
	estimated_Y_strong_edge = cv::Mat(matrix_size_est, CV_32F, cv::Scalar::all(0));
	estimated_Z_strong_edge = cv::Mat(matrix_size_est, CV_32F, cv::Scalar::all(0));

	estimated_Disparity_soft_edge = cv::Mat(matrix_size_est, CV_32F, cv::Scalar::all(0));
	estimated_X_soft_edge = cv::Mat(matrix_size_est, CV_32F, cv::Scalar::all(0));
	estimated_Y_soft_edge = cv::Mat(matrix_size_est, CV_32F, cv::Scalar::all(0));
	estimated_Z_soft_edge = cv::Mat(matrix_size_est, CV_32F, cv::Scalar::all(0));

	int rows_guess = estimated_Disparity_no_edge.rows * estimated_Disparity_no_edge.cols;
	cv::Size matrix_size_guess = cv::Size(1, rows_guess);
	guessed_Disparity = cv::Mat(matrix_size_guess, CV_32F, cv::Scalar::all(0));
	guessed_X = cv::Mat(matrix_size_guess, CV_32F, cv::Scalar::all(0));
	guessed_Y = cv::Mat(matrix_size_guess, CV_32F, cv::Scalar::all(0));
	guessed_Z = cv::Mat(matrix_size_guess, CV_32F, cv::Scalar::all(0));

	guessed_Disparity_no_edge = cv::Mat(matrix_size_guess, CV_32F, cv::Scalar::all(0));
	guessed_X_no_edge = cv::Mat(matrix_size_guess, CV_32F, cv::Scalar::all(0));
	guessed_Y_no_edge = cv::Mat(matrix_size_guess, CV_32F, cv::Scalar::all(0));
	guessed_Z_no_edge = cv::Mat(matrix_size_guess, CV_32F, cv::Scalar::all(0));

	guessed_Disparity_strong_edge = cv::Mat(matrix_size_guess, CV_32F, cv::Scalar::all(0));
	guessed_X_strong_edge = cv::Mat(matrix_size_guess, CV_32F, cv::Scalar::all(0));
	guessed_Y_strong_edge = cv::Mat(matrix_size_guess, CV_32F, cv::Scalar::all(0));
	guessed_Z_strong_edge = cv::Mat(matrix_size_guess, CV_32F, cv::Scalar::all(0));

	guessed_Disparity_soft_edge = cv::Mat(matrix_size_guess, CV_32F, cv::Scalar::all(0));
	guessed_X_soft_edge = cv::Mat(matrix_size_guess, CV_32F, cv::Scalar::all(0));
	guessed_Y_soft_edge = cv::Mat(matrix_size_guess, CV_32F, cv::Scalar::all(0));
	guessed_Z_soft_edge = cv::Mat(matrix_size_guess, CV_32F, cv::Scalar::all(0));

}

void RealGridOperationsTesting::setEstimations(const cv::Mat& image_left, const cv::Mat& image_right)
{
	/*
	int cols = (sampling_factor + 1) * (sampling_factor + 1);
	int rows = gridSquaresVector.size();
	cv::Size matrix_size = cv::Size(cols, rows);
	estimated_Disparity = cv::Mat(matrix_size, CV_32F, cv::Scalar::all(0));
	estimated_X = cv::Mat(matrix_size, CV_32F, cv::Scalar::all(0));
	estimated_Y = cv::Mat(matrix_size, CV_32F, cv::Scalar::all(0));
	estimated_Z = cv::Mat(matrix_size, CV_32F, cv::Scalar::all(0));
	*/

	MatrixLimits matrix_limits;
	left_image_limits = matrix_limits.defineMatrixLimits(0, image_left.rows, 0, image_left.cols).t();
	right_image_limits = matrix_limits.defineMatrixLimits(0, image_right.rows, 0, image_right.cols).t();

	float no_edges_step = 0.f;
	int size = 4;
	std::vector<ObservationData> no_edge_estimations(size, ObservationData());
	std::vector<ObservationData> strong_edge_estimations(size, ObservationData());
	std::vector<ObservationData> soft_edge_estimations(size, ObservationData());
	int best_index = 0;

	// SubPatch indeces
	int TL = 0;
	int TR = 0;
	int BL = 0;
	int BR = 0;

	int d = 0;

	bool is_strong_edge = false;
	bool is_soft_edge = false;

	for (size_t i = 0; i < gridSquaresVector.size(); i++)
	{
		TL = gridSquaresVector[i].top_left_index;
		TR = gridSquaresVector[i].top_right_index;
		BL = gridSquaresVector[i].bottom_left_index;
		BR = gridSquaresVector[i].bottom_right_index;

		d = 0;
		EdgeShape edg = EdgeShape::undefined;
		is_strong_edge = findStrongEdges(TL, TR, BL, BR, edg);
		is_soft_edge = findSoftEdges(TL, TR, BL, BR, edg);
		if (is_soft_edge == false)
		{
			// Check if there could be strong edges
			if (is_strong_edge == false)
			{
				// No edges
				for (float ii = 0.f; ii <= 1.f; ii += step)
				{
					for (float jj = 0.f; jj <= 1.f; jj += step)
					{
						// Fast bilinear interpolation using the corner values
						// Is it better to have the mean value among the corners or do only one estimation using only one corner ?
						noEdgesEstimations(TL, TR, BL, BR, ii, jj, no_edge_estimations);
						estimated_Disparity.at<float>(i, d) = final_estimastion_no_edge_internal_deriv.disp;
						estimated_Z.at<float>(i, d) = final_estimastion_no_edge_internal_deriv.Z_mt;
						estimated_X.at<float>(i, d) = final_estimastion_no_edge_internal_deriv.X_mt;
						estimated_Y.at<float>(i, d) = final_estimastion_no_edge_internal_deriv.Y_mt;
						
						estimated_Disparity_no_edge.at<float>(i, d) = final_estimastion_no_edge_internal_deriv.disp;
						estimated_Z_no_edge.at<float>(i, d) = final_estimastion_no_edge_internal_deriv.Z_mt;
						estimated_X_no_edge.at<float>(i, d) = final_estimastion_no_edge_internal_deriv.X_mt;
						estimated_Y_no_edge.at<float>(i, d) = final_estimastion_no_edge_internal_deriv.Y_mt;
						
						d++;
					}
				}
			}
			else
			{
				// Strong edge case
				std::fill(raw_differences.begin(), raw_differences.end(), FLT_MAX);
				if (edg == EdgeShape::pure_vertical)
				{
					for (float ii = 0.f; ii <= 1.f; ii += step)
					{
						for (float jj = 0.f; jj <= 1.f; jj += step)
						{
							strongEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, strong_edge_estimations);
							// Is there the need to add penalties? Maybe yes in order to be really sure about the results
							// Then with the penalty it is possible to enhance the different edge shape cases
							raw_differences[0] = raw_differences[0] + P1 * ii + P2 * jj;
							raw_differences[1] = raw_differences[1] + P1 * ii + P2 * (1 - jj);
							raw_differences[2] = raw_differences[2] + P1 * (1 - ii) + P2 * jj;
							raw_differences[3] = raw_differences[3] + P1 * (1 - ii) + P2 * (1 - jj);

							best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							estimated_X.at<float>(i, d) = (strong_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
							estimated_Y.at<float>(i, d) = (strong_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
							estimated_Z.at<float>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity.at<float>(i, d) = strong_edge_estimations[best_index].disp;
							
							estimated_X_strong_edge.at<float>(i, d) = (strong_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
							//std::cout << "Estimated point at TL: " << estimated_X_strong_edge.at<float>(i, d) << std::endl;
							//std::cout << "Real point at TL: " << strong_edge_estimations[best_index].X_mt << std::endl;
							estimated_Y_strong_edge.at<float>(i, d) = (strong_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
							//std::cout << estimated_Y_strong_edge.at<float>(i, d) << std::endl;
							estimated_Z_strong_edge.at<float>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity_strong_edge.at<float>(i, d) = strong_edge_estimations[best_index].disp;
							
							d++;
						}
					}

				}
				else if (edg == EdgeShape::pure_horizontal)
				{
					for (float ii = 0.f; ii <= 1.f; ii += step)
					{
						for (float jj = 0.f; jj <= 1.f; jj += step)
						{
							strongEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, strong_edge_estimations);
							// Is there the need to add penalties? Maybe yes in order to be really sure about the results
							// Then with the penalty it is possible to enhance the different edge shape cases
							raw_differences[0] = raw_differences[0] + P2 * ii + P1 * jj;
							raw_differences[1] = raw_differences[1] + P2 * ii + P1 * (1 - jj);
							raw_differences[2] = raw_differences[2] + P2 * (1 - ii) + P1 * jj;
							raw_differences[3] = raw_differences[3] + P2 * (1 - ii) + P1 * (1 - jj);

							best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							estimated_X.at<float>(i, d) = (strong_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
							estimated_Y.at<float>(i, d) = (strong_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
							estimated_Z.at<float>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity.at<float>(i, d) = strong_edge_estimations[best_index].disp;
							
							estimated_X_strong_edge.at<float>(i, d) = (strong_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
							estimated_Y_strong_edge.at<float>(i, d) = (strong_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
							estimated_Z_strong_edge.at<float>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity_strong_edge.at<float>(i, d) = strong_edge_estimations[best_index].disp;
							
							d++;
						}
					}
				}
				else if (edg == EdgeShape::diagonal_top_left)
				{
					for (float ii = 0.f; ii <= 1.f; ii += step)
					{
						for (float jj = 0.f; jj <= 1.f; jj += step)
						{
							strongEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, strong_edge_estimations);
							// Is there the need to add penalties? Maybe yes in order to be really sure about the results
							// Then with the penalty it is possible to enhance the different edge shape cases
							// The few bigger penalty should go on the different edge or on the other three edges ?
							raw_differences[0] = raw_differences[0] + P4 * ii + P4 * jj;
							raw_differences[1] = raw_differences[1] + P3 * ii + P3 * (1 - jj);
							raw_differences[2] = raw_differences[2] + P3 * (1 - ii) + P3 * jj;
							raw_differences[3] = raw_differences[3] + P3 * (1 - ii) + P3 * (1 - jj);

							best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							estimated_X.at<float>(i, d) = (strong_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
							estimated_Y.at<float>(i, d) = (strong_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
							estimated_Z.at<float>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity.at<float>(i, d) = strong_edge_estimations[best_index].disp;
							
							estimated_X_strong_edge.at<float>(i, d) = (strong_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
							estimated_Y_strong_edge.at<float>(i, d) = (strong_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
							estimated_Z_strong_edge.at<float>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity_strong_edge.at<float>(i, d) = strong_edge_estimations[best_index].disp;
							
							d++;
						}
					}
				}
				else if (edg == EdgeShape::diagonal_top_right)
				{
					for (float ii = 0.f; ii <= 1.f; ii += step)
					{
						for (float jj = 0.f; jj <= 1.f; jj += step)
						{
							strongEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, strong_edge_estimations);
							// Is there the need to add penalties? Maybe yes in order to be really sure about the results
							// Then with the penalty it is possible to enhance the different edge shape cases
							raw_differences[0] = raw_differences[0] + P3 * ii + P3 * jj;
							raw_differences[1] = raw_differences[1] + P4 * ii + P4 * (1 - jj);
							raw_differences[2] = raw_differences[2] + P3 * (1 - ii) + P3 * jj;
							raw_differences[3] = raw_differences[3] + P3 * (1 - ii) + P3 * (1 - jj);

							best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							estimated_X.at<float>(i, d) = (strong_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
							estimated_Y.at<float>(i, d) = (strong_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
							estimated_Z.at<float>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity.at<float>(i, d) = strong_edge_estimations[best_index].disp;
							
							estimated_X_strong_edge.at<float>(i, d) = (strong_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
							estimated_Y_strong_edge.at<float>(i, d) = (strong_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
							estimated_Z_strong_edge.at<float>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity_strong_edge.at<float>(i, d) = strong_edge_estimations[best_index].disp;
							
							d++;
						}
					}
				}
				else if (edg == EdgeShape::diagonal_bottom_left)
				{
					for (float ii = 0.f; ii <= 1.f; ii += step)
					{
						for (float jj = 0.f; jj <= 1.f; jj += step)
						{
							strongEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, strong_edge_estimations);
							// Is there the need to add penalties? Maybe yes in order to be really sure about the results
							// Then with the penalty it is possible to enhance the different edge shape cases
							raw_differences[0] = raw_differences[0] + P3 * ii + P3 * jj;
							raw_differences[1] = raw_differences[1] + P3 * ii + P3 * (1 - jj);
							raw_differences[2] = raw_differences[2] + P4 * (1 - ii) + P4 * jj;
							raw_differences[3] = raw_differences[3] + P3 * (1 - ii) + P3 * (1 - jj);

							best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							estimated_X.at<float>(i, d) = (strong_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
							estimated_Y.at<float>(i, d) = (strong_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
							estimated_Z.at<float>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity.at<float>(i, d) = strong_edge_estimations[best_index].disp;
							
							estimated_X_strong_edge.at<float>(i, d) = (strong_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
							estimated_Y_strong_edge.at<float>(i, d) = (strong_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
							estimated_Z_strong_edge.at<float>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity_strong_edge.at<float>(i, d) = strong_edge_estimations[best_index].disp;
							
							d++;
						}
					}
				}
				else if (edg == EdgeShape::diagonal_bottom_right)
				{
					for (float ii = 0.f; ii <= 1.f; ii += step)
					{
						for (float jj = 0.f; jj <= 1.f; jj += step)
						{
							strongEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, strong_edge_estimations);
							// Is there the need to add penalties? Maybe yes in order to be really sure about the results
							// Then with the penalty it is possible to enhance the different edge shape cases
							raw_differences[0] = raw_differences[0] + P3 * ii + P3 * jj;
							raw_differences[1] = raw_differences[1] + P3 * ii + P3 * (1 - jj);
							raw_differences[2] = raw_differences[2] + P3 * (1 - ii) + P3 * jj;
							raw_differences[3] = raw_differences[3] + P4 * (1 - ii) + P4 * (1 - jj);

							best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							estimated_X.at<float>(i, d) = (strong_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
							estimated_Y.at<float>(i, d) = (strong_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
							estimated_Z.at<float>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity.at<float>(i, d) = strong_edge_estimations[best_index].disp;
							
							estimated_X_strong_edge.at<float>(i, d) = (strong_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
							estimated_Y_strong_edge.at<float>(i, d) = (strong_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
							estimated_Z_strong_edge.at<float>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity_strong_edge.at<float>(i, d) = strong_edge_estimations[best_index].disp;
							
							d++;
						}
					}
				}
				else
				{
					// Unknown type of edge --> maybe use a big penalty for all the estimations 
					for (float ii = 0.f; ii <= 1.f; ii += step)
					{
						for (float jj = 0.f; jj <= 1.f; jj += step)
						{
							strongEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, strong_edge_estimations);
							// Is there the need to add penalties? Maybe yes in order to be really sure about the results
							// Then with the penalty it is possible to enhance the different edge shape cases
							raw_differences[0] = raw_differences[0] + P0 * ii + P0 * jj;
							raw_differences[1] = raw_differences[1] + P0 * ii + P0 * (1 - jj);
							raw_differences[2] = raw_differences[2] + P0 * (1 - ii) + P0 * jj;
							raw_differences[3] = raw_differences[3] + P0 * (1 - ii) + P0 * (1 - jj);

							best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							estimated_X.at<float>(i, d) = (strong_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
							estimated_Y.at<float>(i, d) = (strong_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
							estimated_Z.at<float>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity.at<float>(i, d) = strong_edge_estimations[best_index].disp;
							
							estimated_X_strong_edge.at<float>(i, d) = (strong_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
							estimated_Y_strong_edge.at<float>(i, d) = (strong_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * strong_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
							estimated_Z_strong_edge.at<float>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity_strong_edge.at<float>(i, d) = strong_edge_estimations[best_index].disp;
							
							d++;
						}
					}
				}
			}
		}
		else
		{
			
			// Soft edges case
			if (edg == EdgeShape::pure_vertical)
			{
				for (float ii = 0.f; ii <= 1.f; ii += step)
				{
					for (float jj = 0.f; jj <= 1.f; jj += step)
					{
						softEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, soft_edge_estimations);
						// Is there the need to add penalties? Maybe yes in order to be really sure about the results
						// Then with the penalty it is possible to enhance the different edge shape cases
						raw_differences[0] = raw_differences[0] + P1 * ii + P2 * jj;
						raw_differences[1] = raw_differences[1] + P1 * ii + P2 * (1 - jj);
						raw_differences[2] = raw_differences[2] + P1 * (1 - ii) + P2 * jj;
						raw_differences[3] = raw_differences[3] + P1 * (1 - ii) + P2 * (1 - jj);

						best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
						estimated_X.at<float>(i, d) = (soft_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
						estimated_Y.at<float>(i, d) = (soft_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
						estimated_Z.at<float>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity.at<float>(i, d) = soft_edge_estimations[best_index].disp;
						
						estimated_X_soft_edge.at<float>(i, d) = (soft_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
						estimated_Y_soft_edge.at<float>(i, d) = (soft_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
						estimated_Z_soft_edge.at<float>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity_soft_edge.at<float>(i, d) = soft_edge_estimations[best_index].disp;
						
						d++;
					}
				}

			}
			else if (edg == EdgeShape::pure_horizontal)
			{
				for (float ii = 0.f; ii <= 1.f; ii += step)
				{
					for (float jj = 0.f; jj <= 1.f; jj += step)
					{
						softEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, soft_edge_estimations);
						// Is there the need to add penalties? Maybe yes in order to be really sure about the results
						// Then with the penalty it is possible to enhance the different edge shape cases
						raw_differences[0] = raw_differences[0] + P2 * ii + P1 * jj;
						raw_differences[1] = raw_differences[1] + P2 * ii + P1 * (1 - jj);
						raw_differences[2] = raw_differences[2] + P2 * (1 - ii) + P1 * jj;
						raw_differences[3] = raw_differences[3] + P2 * (1 - ii) + P1 * (1 - jj);

						best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
						estimated_X.at<float>(i, d) = (soft_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
						estimated_Y.at<float>(i, d) = (soft_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
						estimated_Z.at<float>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity.at<float>(i, d) = soft_edge_estimations[best_index].disp;
						
						estimated_X_soft_edge.at<float>(i, d) = (soft_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
						estimated_Y_soft_edge.at<float>(i, d) = (soft_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
						estimated_Z_soft_edge.at<float>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity_soft_edge.at<float>(i, d) = soft_edge_estimations[best_index].disp;
						
						d++;
					}
				}
			}
			else if (edg == EdgeShape::diagonal_top_left)
			{
				for (float ii = 0.f; ii <= 1.f; ii += step)
				{
					for (float jj = 0.f; jj <= 1.f; jj += step)
					{
						softEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, soft_edge_estimations);
						// Is there the need to add penalties? Maybe yes in order to be really sure about the results
						// Then with the penalty it is possible to enhance the different edge shape cases
						raw_differences[0] = raw_differences[0] + P4 * ii + P4 * jj;
						raw_differences[1] = raw_differences[1] + P3 * ii + P3 * (1 - jj);
						raw_differences[2] = raw_differences[2] + P3 * (1 - ii) + P3 * jj;
						raw_differences[3] = raw_differences[3] + P3 * (1 - ii) + P3 * (1 - jj);

						best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
						estimated_X.at<float>(i, d) = (soft_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
						estimated_Y.at<float>(i, d) = (soft_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
						estimated_Z.at<float>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity.at<float>(i, d) = soft_edge_estimations[best_index].disp;
						
						estimated_X_soft_edge.at<float>(i, d) = (soft_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
						estimated_Y_soft_edge.at<float>(i, d) = (soft_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
						estimated_Z_soft_edge.at<float>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity_soft_edge.at<float>(i, d) = soft_edge_estimations[best_index].disp;
						
						d++;
					}
				}
			}
			else if (edg == EdgeShape::diagonal_top_right)
			{
				for (float ii = 0.f; ii <= 1.f; ii += step)
				{
					for (float jj = 0.f; jj <= 1.f; jj += step)
					{
						softEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, soft_edge_estimations);
						// Is there the need to add penalties? Maybe yes in order to be really sure about the results
						// Then with the penalty it is possible to enhance the different edge shape cases
						raw_differences[0] = raw_differences[0] + P3 * ii + P3 * jj;
						raw_differences[1] = raw_differences[1] + P4 * ii + P4 * (1 - jj);
						raw_differences[2] = raw_differences[2] + P3 * (1 - ii) + P3 * jj;
						raw_differences[3] = raw_differences[3] + P3 * (1 - ii) + P3 * (1 - jj);

						best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
						estimated_X.at<float>(i, d) = (soft_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
						estimated_Y.at<float>(i, d) = (soft_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
						estimated_Z.at<float>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity.at<float>(i, d) = soft_edge_estimations[best_index].disp;
						
						estimated_X_soft_edge.at<float>(i, d) = (soft_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
						estimated_Y_soft_edge.at<float>(i, d) = (soft_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
						estimated_Z_soft_edge.at<float>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity_soft_edge.at<float>(i, d) = soft_edge_estimations[best_index].disp;
						
						d++;
					}
				}
			}
			else if (edg == EdgeShape::diagonal_bottom_left)
			{
				for (float ii = 0.f; ii <= 1.f; ii += step)
				{
					for (float jj = 0.f; jj <= 1.f; jj += step)
					{
						softEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, soft_edge_estimations);
						// Is there the need to add penalties? Maybe yes in order to be really sure about the results
						// Then with the penalty it is possible to enhance the different edge shape cases
						raw_differences[0] = raw_differences[0] + P3 * ii + P3 * jj;
						raw_differences[1] = raw_differences[1] + P3 * ii + P3 * (1 - jj);
						raw_differences[2] = raw_differences[2] + P4 * (1 - ii) + P4 * jj;
						raw_differences[3] = raw_differences[3] + P3 * (1 - ii) + P3 * (1 - jj);

						best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
						estimated_X.at<float>(i, d) = (soft_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
						estimated_Y.at<float>(i, d) = (soft_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
						estimated_Z.at<float>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity.at<float>(i, d) = soft_edge_estimations[best_index].disp;
						
						estimated_X_soft_edge.at<float>(i, d) = (soft_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
						estimated_Y_soft_edge.at<float>(i, d) = (soft_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
						estimated_Z_soft_edge.at<float>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity_soft_edge.at<float>(i, d) = soft_edge_estimations[best_index].disp;
						
						d++;
					}
				}
			}
			else if (edg == EdgeShape::diagonal_bottom_right)
			{
				for (float ii = 0.f; ii <= 1.f; ii += step)
				{
					for (float jj = 0.f; jj <= 1.f; jj += step)
					{
						softEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, soft_edge_estimations);
						// Is there the need to add penalties? Maybe yes in order to be really sure about the results
						// Then with the penalty it is possible to enhance the different edge shape cases
						raw_differences[0] = raw_differences[0] + P3 * ii + P3 * jj;
						raw_differences[1] = raw_differences[1] + P3 * ii + P3 * (1 - jj);
						raw_differences[2] = raw_differences[2] + P3 * (1 - ii) + P3 * jj;
						raw_differences[3] = raw_differences[3] + P4 * (1 - ii) + P4 * (1 - jj);

						best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
						estimated_X.at<float>(i, d) = (soft_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
						estimated_Y.at<float>(i, d) = (soft_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
						estimated_Z.at<float>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity.at<float>(i, d) = soft_edge_estimations[best_index].disp;
						
						estimated_X_soft_edge.at<float>(i, d) = (soft_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
						estimated_Y_soft_edge.at<float>(i, d) = (soft_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
						estimated_Z_soft_edge.at<float>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity_soft_edge.at<float>(i, d) = soft_edge_estimations[best_index].disp;
						
						d++;
					}
				}
			}
			else
			{
				// Undefined edge case
				for (float ii = 0.f; ii <= 1.f; ii += step)
				{
					for (float jj = 0.f; jj <= 1.f; jj += step)
					{
						softEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, soft_edge_estimations);
						// Is there the need to add penalties? Maybe yes in order to be really sure about the results
						// Then with the penalty it is possible to enhance the different edge shape cases
						raw_differences[0] = raw_differences[0] + P0 * ii + P0 * jj;
						raw_differences[1] = raw_differences[1] + P0 * ii + P0 * (1 - jj);
						raw_differences[2] = raw_differences[2] + P0 * (1 - ii) + P0 * jj;
						raw_differences[3] = raw_differences[3] + P0 * (1 - ii) + P0 * (1 - jj);

						best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
						estimated_X.at<float>(i, d) = (soft_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
						estimated_Y.at<float>(i, d) = (soft_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
						estimated_Z.at<float>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity.at<float>(i, d) = soft_edge_estimations[best_index].disp;
						
						estimated_X_soft_edge.at<float>(i, d) = (soft_edge_estimations[best_index].x_px - camera_matrix_left.at<float>(0, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(0, 0);
						estimated_Y_soft_edge.at<float>(i, d) = (soft_edge_estimations[best_index].y_px - camera_matrix_left.at<float>(1, 2)) * soft_edge_estimations[best_index].Z_mt / camera_matrix_left.at<float>(1, 1);
						estimated_Z_soft_edge.at<float>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity_soft_edge.at<float>(i, d) = soft_edge_estimations[best_index].disp;
						
						d++;
					}
				}
			}
		}
	}
}


void RealGridOperationsTesting::setGuesses()
{
	int TL_indx{ 0 };
	int TR_indx{ 0 };
	int BL_indx{ 0 };
	int BR_indx{ 0 };

	/*
	// Matrix initialization
	int rows = estimated_Disparity.rows * estimated_Disparity.cols;
	cv::Size matrix_size = cv::Size(1, rows);
	guessed_Disparity = cv::Mat(matrix_size, CV_32F, cv::Scalar::all(0));
	guessed_X = cv::Mat(matrix_size, CV_32F, cv::Scalar::all(0));
	guessed_Y = cv::Mat(matrix_size, CV_32F, cv::Scalar::all(0));
	guessed_Z = cv::Mat(matrix_size, CV_32F, cv::Scalar::all(0));
	*/

	int r = 0;

	for (size_t i = 0; i < estimated_Disparity.rows; i++)
	{
		for (size_t j = 0; j < estimated_Disparity.cols; j++)
		{
			guessed_Disparity.at<float>(r, 0) = estimated_Disparity.at<float>(i, j);
			guessed_X.at<float>(r, 0) = estimated_X.at<float>(i, j);
			guessed_Y.at<float>(r, 0) = estimated_Y.at<float>(i, j);
			guessed_Z.at<float>(r, 0) = estimated_Z.at<float>(i, j);

			guessed_Disparity_no_edge.at<float>(r, 0) = estimated_Disparity_no_edge.at<float>(i, j);
			guessed_X_no_edge.at<float>(r, 0) = estimated_X_no_edge.at<float>(i, j);
			guessed_Y_no_edge.at<float>(r, 0) = estimated_Y_no_edge.at<float>(i, j);
			guessed_Z_no_edge.at<float>(r, 0) = estimated_Z_no_edge.at<float>(i, j);

			guessed_Disparity_strong_edge.at<float>(r, 0) = estimated_Disparity_strong_edge.at<float>(i, j);
			guessed_X_strong_edge.at<float>(r, 0) = estimated_X_strong_edge.at<float>(i, j);
			guessed_Y_strong_edge.at<float>(r, 0) = estimated_Y_strong_edge.at<float>(i, j);
			guessed_Z_strong_edge.at<float>(r, 0) = estimated_Z_strong_edge.at<float>(i, j);

			guessed_Disparity_soft_edge.at<float>(r, 0) = estimated_Disparity_soft_edge.at<float>(i, j);
			guessed_X_soft_edge.at<float>(r, 0) = estimated_X_soft_edge.at<float>(i, j);
			guessed_Y_soft_edge.at<float>(r, 0) = estimated_Y_soft_edge.at<float>(i, j);
			guessed_Z_soft_edge.at<float>(r, 0) = estimated_Z_soft_edge.at<float>(i, j);

			if (j == 0)
			{
				TL_indx = gridSquaresVector[i].top_left_index;
				guessed_Disparity.at<float>(r, 0) = observationsVector[TL_indx].disp;
				guessed_X.at<float>(r, 0) = observationsVector[TL_indx].X_mt;
				guessed_Y.at<float>(r, 0) = observationsVector[TL_indx].Y_mt;
				guessed_Z.at<float>(r, 0) = observationsVector[TL_indx].Z_mt;

				guessed_Disparity_no_edge.at<float>(r, 0) = observationsVector[TL_indx].disp;
				guessed_X_no_edge.at<float>(r, 0) = observationsVector[TL_indx].X_mt;
				guessed_Y_no_edge.at<float>(r, 0) = observationsVector[TL_indx].Y_mt;
				guessed_Z_no_edge.at<float>(r, 0) = observationsVector[TL_indx].Z_mt;

				guessed_Disparity_strong_edge.at<float>(r, 0) = observationsVector[TL_indx].disp;
				guessed_X_strong_edge.at<float>(r, 0) = observationsVector[TL_indx].X_mt;
				guessed_Y_strong_edge.at<float>(r, 0) = observationsVector[TL_indx].Y_mt;
				guessed_Z_strong_edge.at<float>(r, 0) = observationsVector[TL_indx].Z_mt;

				guessed_Disparity_soft_edge.at<float>(r, 0) = observationsVector[TL_indx].disp;
				guessed_X_soft_edge.at<float>(r, 0) = observationsVector[TL_indx].X_mt;
				guessed_Y_soft_edge.at<float>(r, 0) = observationsVector[TL_indx].Y_mt;
				guessed_Z_soft_edge.at<float>(r, 0) = observationsVector[TL_indx].Z_mt;
			}
			else if (j == sampling_factor)
			{
				TR_indx = gridSquaresVector[i].top_right_index;
				guessed_Disparity.at<float>(r, 0) = observationsVector[TR_indx].disp;
				guessed_X.at<float>(r, 0) = observationsVector[TR_indx].X_mt;
				guessed_Y.at<float>(r, 0) = observationsVector[TR_indx].Y_mt;
				guessed_Z.at<float>(r, 0) = observationsVector[TR_indx].Z_mt;

				guessed_Disparity_no_edge.at<float>(r, 0) = observationsVector[TR_indx].disp;
				guessed_X_no_edge.at<float>(r, 0) = observationsVector[TR_indx].X_mt;
				guessed_Y_no_edge.at<float>(r, 0) = observationsVector[TR_indx].Y_mt;
				guessed_Z_no_edge.at<float>(r, 0) = observationsVector[TR_indx].Z_mt;

				guessed_Disparity_strong_edge.at<float>(r, 0) = observationsVector[TR_indx].disp;
				guessed_X_strong_edge.at<float>(r, 0) = observationsVector[TR_indx].X_mt;
				guessed_Y_strong_edge.at<float>(r, 0) = observationsVector[TR_indx].Y_mt;
				guessed_Z_strong_edge.at<float>(r, 0) = observationsVector[TR_indx].Z_mt;

				guessed_Disparity_soft_edge.at<float>(r, 0) = observationsVector[TR_indx].disp;
				guessed_X_soft_edge.at<float>(r, 0) = observationsVector[TR_indx].X_mt;
				guessed_Y_soft_edge.at<float>(r, 0) = observationsVector[TR_indx].Y_mt;
				guessed_Z_soft_edge.at<float>(r, 0) = observationsVector[TR_indx].Z_mt;
			}
			else if (j == float(estimated_Disparity.cols - 1 - sampling_factor))
			{
				BL_indx = gridSquaresVector[i].bottom_left_index;
				guessed_Disparity.at<float>(r, 0) = observationsVector[BL_indx].disp;
				guessed_X.at<float>(r, 0) = observationsVector[BL_indx].X_mt;
				guessed_Y.at<float>(r, 0) = observationsVector[BL_indx].Y_mt;
				guessed_Z.at<float>(r, 0) = observationsVector[BL_indx].Z_mt;

				guessed_Disparity_no_edge.at<float>(r, 0) = observationsVector[BL_indx].disp;
				guessed_X_no_edge.at<float>(r, 0) = observationsVector[BL_indx].X_mt;
				guessed_Y_no_edge.at<float>(r, 0) = observationsVector[BL_indx].Y_mt;
				guessed_Z_no_edge.at<float>(r, 0) = observationsVector[BL_indx].Z_mt;

				guessed_Disparity_strong_edge.at<float>(r, 0) = observationsVector[BL_indx].disp;
				guessed_X_strong_edge.at<float>(r, 0) = observationsVector[BL_indx].X_mt;
				guessed_Y_strong_edge.at<float>(r, 0) = observationsVector[BL_indx].Y_mt;
				guessed_Z_strong_edge.at<float>(r, 0) = observationsVector[BL_indx].Z_mt;

				guessed_Disparity_soft_edge.at<float>(r, 0) = observationsVector[BL_indx].disp;
				guessed_X_soft_edge.at<float>(r, 0) = observationsVector[BL_indx].X_mt;
				guessed_Y_soft_edge.at<float>(r, 0) = observationsVector[BL_indx].Y_mt;
				guessed_Z_soft_edge.at<float>(r, 0) = observationsVector[BL_indx].Z_mt;
			}
			else if (j == float(estimated_Disparity.cols - 1))
			{
				BR_indx = gridSquaresVector[i].bottom_right_index;
				guessed_Disparity.at<float>(r, 0) = observationsVector[BR_indx].disp;
				guessed_X.at<float>(r, 0) = observationsVector[BR_indx].X_mt;
				guessed_Y.at<float>(r, 0) = observationsVector[BR_indx].Y_mt;
				guessed_Z.at<float>(r, 0) = observationsVector[BR_indx].Z_mt;

				guessed_Disparity_no_edge.at<float>(r, 0) = observationsVector[BR_indx].disp;
				guessed_X_no_edge.at<float>(r, 0) = observationsVector[BR_indx].X_mt;
				guessed_Y_no_edge.at<float>(r, 0) = observationsVector[BR_indx].Y_mt;
				guessed_Z_no_edge.at<float>(r, 0) = observationsVector[BR_indx].Z_mt;

				guessed_Disparity_strong_edge.at<float>(r, 0) = observationsVector[BR_indx].disp;
				guessed_X_strong_edge.at<float>(r, 0) = observationsVector[BR_indx].X_mt;
				guessed_Y_strong_edge.at<float>(r, 0) = observationsVector[BR_indx].Y_mt;
				guessed_Z_strong_edge.at<float>(r, 0) = observationsVector[BR_indx].Z_mt;

				guessed_Disparity_soft_edge.at<float>(r, 0) = observationsVector[BR_indx].disp;
				guessed_X_soft_edge.at<float>(r, 0) = observationsVector[BR_indx].X_mt;
				guessed_Y_soft_edge.at<float>(r, 0) = observationsVector[BR_indx].Y_mt;
				guessed_Z_soft_edge.at<float>(r, 0) = observationsVector[BR_indx].Z_mt;
			}
			r++;
		}
	}
}

void RealGridOperationsTesting::setSGMCostCubes(const std::vector<cv::Mat>& SGMCostCubesVector_)
{
	SGMCostCubesVector = SGMCostCubesVector_;
	int SGMVectors_size = avg_square_size_4eye * avg_square_size_4eye * SGMCostCubesVector.size();
	SGMEstimatedDisparity.resize(SGMVectors_size);
	SGMEstimated_X.resize(SGMVectors_size);
	SGMEstimated_Y.resize(SGMVectors_size);
	SGMEstimated_Z.resize(SGMVectors_size);
	SGMEstimated_x_px.resize(SGMVectors_size);
	SGMEstimated_y_px.resize(SGMVectors_size);
}

void RealGridOperationsTesting::setSGMBasedEstimations(const cv::Mat& image_left, const cv::Mat& image_right)
{

	// Disparity levels vector
	std::vector<float> patch_disparity_values(4, 0.f);

	// SubPatch indeces
	int TL = 0;
	int TR = 0;
	int BL = 0;
	int BR = 0;

	bool is_edge = false;
	int k = 0;

	for (size_t i = 0; i < gridSquaresVector.size(); i++)
	{
		TL = gridSquaresVector[i].top_left_index;
		TR = gridSquaresVector[i].top_right_index;
		BL = gridSquaresVector[i].bottom_left_index;
		BR = gridSquaresVector[i].bottom_right_index;

		patch_disparity_values[0] = observationsVector[TL].disp;
		patch_disparity_values[1] = observationsVector[TR].disp;
		patch_disparity_values[2] = observationsVector[BL].disp;
		patch_disparity_values[3] = observationsVector[BR].disp;

		fillSGMCostCube(TL, TR, BL, BR, i, image_left, image_right, patch_disparity_values);
		SGMEdgeDirection edg = SGMEdgeDirection::undefined;
		is_edge = findSGMEdgeDirection(TL, TR, BL, BR, edg);
		
		if (is_edge == false)
		{
			// No edges
			// No need to aggregate the cost
			// Take the best among the 4 corners' disparity values
			noEdgesSGMEstimations(TL, TR, BL, BR, i, patch_disparity_values, k);
		}
		else
		{
			// Edge Case
			if (edg == SGMEdgeDirection::vertical)
			{
				verticalEdgesSGMEstimations(TL, TR, BL, BR, i, patch_disparity_values, k);
			}
			else if (edg == SGMEdgeDirection::horizontal)
			{
				horizontalEdgesSGMEstimations(TL, TR, BL, BR, i, patch_disparity_values, k);
			}
			else if (edg == SGMEdgeDirection::undefined)
			{
				undefinedEdgesSGMEstimations(TL, TR, BL, BR, i, patch_disparity_values, k);
			}
		}
	}

}

cv::Mat RealGridOperationsTesting::getEstimations_disp()
{
	return guessed_Disparity;
}

cv::Mat RealGridOperationsTesting::getEstimations_X()
{
	return guessed_X;
}

cv::Mat RealGridOperationsTesting::getEstimations_Y()
{
	return guessed_Y;
}

cv::Mat RealGridOperationsTesting::getEstimations_Z()
{
	return guessed_Z;
}

cv::Mat RealGridOperationsTesting::getEstimations_disp_no_edge()
{
	return guessed_Disparity_no_edge;
}

cv::Mat RealGridOperationsTesting::getEstimations_X_no_edge()
{
	return guessed_X_no_edge;
}

cv::Mat RealGridOperationsTesting::getEstimations_Y_no_edge()
{
	return guessed_Y_no_edge;
}

cv::Mat RealGridOperationsTesting::getEstimations_Z_no_edge()
{
	return guessed_Z_no_edge;
}

cv::Mat RealGridOperationsTesting::getEstimations_disp_strong_edge()
{
	return guessed_Disparity_strong_edge;
}

cv::Mat RealGridOperationsTesting::getEstimations_X_strong_edge()
{
	return guessed_X_strong_edge;
}

cv::Mat RealGridOperationsTesting::getEstimations_Y_strong_edge()
{
	return guessed_Y_strong_edge;
}

cv::Mat RealGridOperationsTesting::getEstimations_Z_strong_edge()
{
	return guessed_Z_strong_edge;
}

cv::Mat RealGridOperationsTesting::getEstimations_disp_soft_edge()
{
	return guessed_Disparity_soft_edge;
}

cv::Mat RealGridOperationsTesting::getEstimations_X_soft_edge()
{
	return guessed_X_soft_edge;
}

cv::Mat RealGridOperationsTesting::getEstimations_Y_soft_edge()
{
	return guessed_Y_soft_edge;
}

cv::Mat RealGridOperationsTesting::getEstimations_Z_soft_edge()
{
	return guessed_Z_soft_edge;
}

std::vector<float> RealGridOperationsTesting::getSGMEstimations_disp()
{
	return SGMEstimatedDisparity;
}

std::vector<float> RealGridOperationsTesting::getSGMEstimations_X()
{
	return SGMEstimated_X;
}

std::vector<float> RealGridOperationsTesting::getSGMEstimations_Y()
{
	return SGMEstimated_Y;
}

std::vector<float> RealGridOperationsTesting::getSGMEstimations_Z()
{
	return SGMEstimated_Z;
}

std::vector<float> RealGridOperationsTesting::getSGMEstimations_x_px()
{
	return SGMEstimated_x_px;
}

std::vector<float> RealGridOperationsTesting::getSGMEstimations_y_px()
{
	return SGMEstimated_y_px;
}

bool RealGridOperationsTesting::findStrongEdges(const int& TL, const int& TR, const int& BL, const int& BR, EdgeShape& edge_shape)
{
	EdgeThresholds edg_thresh;
	bool is_strong_edge = false;

	// CASE STRONG (REALLY VISIBLE EDGES) -- USE NOW THIS AS UNIQUE METHOD (FOR CHECK IF IMPROVEMENTS)
	if (abs(observationsVector[TL].Z_mt - observationsVector[TR].Z_mt) >= edg_thresh.light_depth_threshold) is_strong_edge = true;
	if (abs(observationsVector[BL].Z_mt - observationsVector[BR].Z_mt) >= edg_thresh.light_depth_threshold) is_strong_edge = true;
	if (abs(observationsVector[TL].Z_mt - observationsVector[BL].Z_mt) >= edg_thresh.light_depth_threshold) is_strong_edge = true;
	if (abs(observationsVector[TR].Z_mt - observationsVector[BR].Z_mt) >= edg_thresh.light_depth_threshold) is_strong_edge = true;


	if (is_strong_edge == true)
	{
		edge_shape = EdgeShape::undefined;
		if (abs(observationsVector[TL].Z_mt - observationsVector[TR].Z_mt) >= edg_thresh.light_depth_threshold &&
			abs(observationsVector[BL].Z_mt - observationsVector[BR].Z_mt) >= edg_thresh.light_depth_threshold) edge_shape = EdgeShape::pure_vertical;
		if (abs(observationsVector[TL].Z_mt - observationsVector[BL].Z_mt) >= edg_thresh.light_depth_threshold &&
			abs(observationsVector[TR].Z_mt - observationsVector[BR].Z_mt) >= edg_thresh.light_depth_threshold) edge_shape = EdgeShape::pure_horizontal;
		if (abs(observationsVector[TL].Z_mt - observationsVector[TR].Z_mt) >= edg_thresh.light_depth_threshold &&
			abs(observationsVector[TL].Z_mt - observationsVector[BL].Z_mt) >= edg_thresh.light_depth_threshold) edge_shape = EdgeShape::diagonal_top_left;
		if (abs(observationsVector[TR].Z_mt - observationsVector[TL].Z_mt) >= edg_thresh.light_depth_threshold &&
			abs(observationsVector[TR].Z_mt - observationsVector[BR].Z_mt) >= edg_thresh.light_depth_threshold) edge_shape = EdgeShape::diagonal_top_right;
		if (abs(observationsVector[BL].Z_mt - observationsVector[TL].Z_mt) >= edg_thresh.light_depth_threshold &&
			abs(observationsVector[BL].Z_mt - observationsVector[BR].Z_mt) >= edg_thresh.light_depth_threshold) edge_shape = EdgeShape::diagonal_bottom_left;
		if (abs(observationsVector[BR].Z_mt - observationsVector[TR].Z_mt) >= edg_thresh.light_depth_threshold &&
			abs(observationsVector[BR].Z_mt - observationsVector[BL].Z_mt) >= edg_thresh.light_depth_threshold) edge_shape = EdgeShape::diagonal_bottom_right;
	}
	
	return is_strong_edge;
}

bool RealGridOperationsTesting::findSoftEdges(const int& TL, const int& TR, const int& BL, const int& BR, EdgeShape& edge_shape)
{
	EdgeThresholds edg_thresh;
	bool is_soft_edge = false;

	// CASE STRONG (REALLY VISIBLE EDGES) -- USE NOW THIS AS UNIQUE METHOD (FOR CHECK IF IMPROVEMENTS)
	if (abs(observationsVector[TL].Z_mt - observationsVector[TR].Z_mt) >= edg_thresh.strong_depth_threshold && abs(observationsVector[TL].Z_mt - observationsVector[TR].Z_mt) < edg_thresh.light_depth_threshold) is_soft_edge = true;
	if (abs(observationsVector[BL].Z_mt - observationsVector[BR].Z_mt) >= edg_thresh.strong_depth_threshold && abs(observationsVector[BL].Z_mt - observationsVector[BR].Z_mt) < edg_thresh.light_depth_threshold) is_soft_edge = true;
	if (abs(observationsVector[TL].Z_mt - observationsVector[BL].Z_mt) >= edg_thresh.strong_depth_threshold && abs(observationsVector[TL].Z_mt - observationsVector[BL].Z_mt) < edg_thresh.light_depth_threshold) is_soft_edge = true;
	if (abs(observationsVector[TR].Z_mt - observationsVector[BR].Z_mt) >= edg_thresh.strong_depth_threshold && abs(observationsVector[TR].Z_mt - observationsVector[BR].Z_mt) < edg_thresh.light_depth_threshold) is_soft_edge = true;


	if (is_soft_edge == true)
	{
		edge_shape = EdgeShape::undefined;
		if (abs(observationsVector[TL].Z_mt - observationsVector[TR].Z_mt) >= edg_thresh.strong_depth_threshold &&
			abs(observationsVector[BL].Z_mt - observationsVector[BR].Z_mt) >= edg_thresh.strong_depth_threshold) edge_shape = EdgeShape::pure_vertical;
		if (abs(observationsVector[TL].Z_mt - observationsVector[BL].Z_mt) >= edg_thresh.strong_depth_threshold &&
			abs(observationsVector[TR].Z_mt - observationsVector[BR].Z_mt) >= edg_thresh.strong_depth_threshold) edge_shape = EdgeShape::pure_horizontal;
		if (abs(observationsVector[TL].Z_mt - observationsVector[TR].Z_mt) >= edg_thresh.strong_depth_threshold &&
			abs(observationsVector[TL].Z_mt - observationsVector[BL].Z_mt) >= edg_thresh.strong_depth_threshold) edge_shape = EdgeShape::diagonal_top_left;
		if (abs(observationsVector[TR].Z_mt - observationsVector[TL].Z_mt) >= edg_thresh.strong_depth_threshold &&
			abs(observationsVector[TR].Z_mt - observationsVector[BR].Z_mt) >= edg_thresh.strong_depth_threshold) edge_shape = EdgeShape::diagonal_top_right;
		if (abs(observationsVector[BL].Z_mt - observationsVector[TL].Z_mt) >= edg_thresh.strong_depth_threshold &&
			abs(observationsVector[BL].Z_mt - observationsVector[BR].Z_mt) >= edg_thresh.strong_depth_threshold) edge_shape = EdgeShape::diagonal_bottom_left;
		if (abs(observationsVector[BR].Z_mt - observationsVector[TR].Z_mt) >= edg_thresh.strong_depth_threshold &&
			abs(observationsVector[BR].Z_mt - observationsVector[BL].Z_mt) >= edg_thresh.strong_depth_threshold) edge_shape = EdgeShape::diagonal_bottom_right;
	}

	return is_soft_edge;
}

bool RealGridOperationsTesting::findSGMEdgeDirection(const int& TL, const int& TR, const int& BL, const int& BR, SGMEdgeDirection& edge_direction)
{
	EdgeThresholds edg_thresh;
	bool is_edge = false;
	if (abs(observationsVector[TL].Z_mt - observationsVector[TR].Z_mt) >= edg_thresh.general_depth_threshold) is_edge = true;
	if (abs(observationsVector[BL].Z_mt - observationsVector[BR].Z_mt) >= edg_thresh.general_depth_threshold) is_edge = true;
	if (abs(observationsVector[TL].Z_mt - observationsVector[BL].Z_mt) >= edg_thresh.general_depth_threshold) is_edge = true;
	if (abs(observationsVector[TR].Z_mt - observationsVector[BR].Z_mt) >= edg_thresh.general_depth_threshold) is_edge = true;

	if (is_edge == true)
	{
		edge_direction = SGMEdgeDirection::undefined;
		if (abs(observationsVector[TL].Z_mt - observationsVector[TR].Z_mt) >= edg_thresh.general_depth_threshold &&
			abs(observationsVector[BL].Z_mt - observationsVector[BR].Z_mt) >= edg_thresh.general_depth_threshold) edge_direction = SGMEdgeDirection::vertical;
		if (abs(observationsVector[TL].Z_mt - observationsVector[BL].Z_mt) >= edg_thresh.general_depth_threshold &&
			abs(observationsVector[TR].Z_mt - observationsVector[BR].Z_mt) >= edg_thresh.general_depth_threshold) edge_direction = SGMEdgeDirection::horizontal;
	}

	return is_edge;
}

void RealGridOperationsTesting::bilinearInterpolation(const int& TL, const int& TR, const int& BL, const int& BR, cv::Vec4f& sub_square_estimation)
{
	cv::Vec2f top_interp{};
	cv::Vec2f bottom_interp{};

	float coeff_col_1 = (observationsVector[TR].X_mt - sub_square_estimation[0]) / (observationsVector[TR].X_mt - observationsVector[TL].X_mt);
	float coeff_col_2 = (sub_square_estimation[0] - observationsVector[TL].X_mt) / (observationsVector[TR].X_mt - observationsVector[TL].X_mt);
	float coeff_row_1 = (observationsVector[BL].Y_mt - sub_square_estimation[1]) / (observationsVector[BL].Y_mt - observationsVector[TL].Y_mt);
	float coeff_row_2 = (sub_square_estimation[1] - observationsVector[TL].Y_mt) / (observationsVector[BL].X_mt - observationsVector[TL].Y_mt);

	top_interp[0] = coeff_col_1 * observationsVector[TL].disp + coeff_col_2 * observationsVector[TR].disp;
	top_interp[1] = coeff_col_1 * observationsVector[TL].Z_mt + coeff_col_2 * observationsVector[TR].Z_mt;

	bottom_interp[0] = coeff_col_1 * observationsVector[BL].disp + coeff_col_2 * observationsVector[BR].disp;
	bottom_interp[1] = coeff_col_1 * observationsVector[BL].Z_mt + coeff_col_2 * observationsVector[BR].Z_mt;

	no_edge_disp = coeff_row_1 * top_interp[0] + coeff_row_2 * bottom_interp[0];
	no_edge_Z = coeff_row_1 * top_interp[1] + coeff_row_2 * bottom_interp[1];
}

void RealGridOperationsTesting::noEdgesEstimations(const int& TL, const int& TR, const int& BL, const int& BR, const float& indx_r, const float& indx_c, std::vector<ObservationData>& no_edge_est_vec)
{
	// Using Internal Derivatives
	no_edge_est_vec[0] = observationsVector[TL] + internalCompleteDerivativesVector[TL].right_deriv * indx_c + internalCompleteDerivativesVector[TL].bottom_deriv * indx_r;
	no_edge_est_vec[1] = observationsVector[TR] + internalCompleteDerivativesVector[TR].left_deriv * (1 - indx_c) + internalCompleteDerivativesVector[TR].bottom_deriv * indx_r;
	no_edge_est_vec[2] = observationsVector[BL] + internalCompleteDerivativesVector[BL].right_deriv * indx_c + internalCompleteDerivativesVector[BL].top_deriv * (1 - indx_r);
	no_edge_est_vec[3] = observationsVector[BR] + internalCompleteDerivativesVector[BR].left_deriv * (1 - indx_c) + internalCompleteDerivativesVector[BR].top_deriv * (1 - indx_r);

	final_estimastion_no_edge_internal_deriv = std::accumulate(no_edge_est_vec.begin(), no_edge_est_vec.end(), ObservationData(0.f, 0.f, 0.f, 0.f, 0.f, 0.f));
	final_estimastion_no_edge_internal_deriv = final_estimastion_no_edge_internal_deriv * static_cast<float>(1.f / no_edge_est_vec.size());
}

void RealGridOperationsTesting::strongEdgesEstimations(const int& TL, const int& TR, const int& BL, const int& BR, const float& indx_r, const float& indx_c, const cv::Mat& im_left, const cv::Mat& im_right, std::vector<ObservationData>& strong_edge_est_vec)
{
	EdgeThresholds edg_thresholds;
	float esternal_der_thresh_check = edg_thresholds.strong_depth_threshold;
	// CHANGE VECTORS NAME FROM NO EDGE TO FOR EDGE 
	// CODE CORRECT BUT WRONG CORRESPONDENCES
	cv::Point2i pixel_pos_left = (0, 0);
	cv::Point2i pixel_pos_right = (0, 0);
	
	// Estimation to top-left
	/*
	strong_edge_est_vec[0].x_px = observationsVector[TL].x_px + internalCompleteDerivativesVector[TL].right_deriv.x_px * indx_c + internalCompleteDerivativesVector[TL].bottom_deriv.x_px * indx_r;
	strong_edge_est_vec[0].y_px = observationsVector[TL].y_px + internalCompleteDerivativesVector[TL].right_deriv.y_px * indx_c + internalCompleteDerivativesVector[TL].bottom_deriv.y_px * indx_r;
	strong_edge_est_vec[0].disp = observationsVector[TL].disp;
	strong_edge_est_vec[0].Z_mt = observationsVector[TL].Z_mt;
	*/
	
	if (esternalCompleteDerivativesVector[TL].top_deriv.Z_mt <= esternal_der_thresh_check && esternalCompleteDerivativesVector[TL].left_deriv.Z_mt <= esternal_der_thresh_check &&
		esternalCompleteDerivativesVector[TL].top_deriv.Z_mt > 0.f && esternalCompleteDerivativesVector[TL].left_deriv.Z_mt > 0.f)
	{
		// if there is no edges between the corner and the external points use the esternal derivative that have to be correct (--> actually in the same plane - among threshold limits -)
		strong_edge_est_vec[0] = observationsVector[TL] + esternalCompleteDerivativesVector[TL].left_deriv * indx_c + esternalCompleteDerivativesVector[TL].top_deriv * indx_r;
	}
	else
	{
		// Edge also from esternal direction !! // Make estimation using internal derivatives and only for x and y that are not affected by edges
		// Assigned same Z and disparity of the corresponding corner
		strong_edge_est_vec[0].x_px = observationsVector[TL].x_px + internalCompleteDerivativesVector[TL].right_deriv.x_px * indx_c + internalCompleteDerivativesVector[TL].bottom_deriv.x_px * indx_r;
		strong_edge_est_vec[0].y_px = observationsVector[TL].y_px + internalCompleteDerivativesVector[TL].right_deriv.y_px * indx_c + internalCompleteDerivativesVector[TL].bottom_deriv.y_px * indx_r;
		strong_edge_est_vec[0].disp = observationsVector[TL].disp;
		strong_edge_est_vec[0].Z_mt = observationsVector[TL].Z_mt;
	}
	

	// Estimation to top-right
	/*
	strong_edge_est_vec[1].x_px = observationsVector[TR].x_px + internalCompleteDerivativesVector[TR].left_deriv.x_px * (1 - indx_c) + internalCompleteDerivativesVector[TR].bottom_deriv.x_px * indx_r;
	strong_edge_est_vec[1].y_px = observationsVector[TR].y_px + internalCompleteDerivativesVector[TR].left_deriv.y_px * (1 - indx_c) + internalCompleteDerivativesVector[TR].bottom_deriv.y_px * indx_r;
	strong_edge_est_vec[1].disp = observationsVector[TR].disp;
	strong_edge_est_vec[1].Z_mt = observationsVector[TR].Z_mt;
	*/
	
	if (esternalCompleteDerivativesVector[TR].top_deriv.Z_mt <= esternal_der_thresh_check && esternalCompleteDerivativesVector[TR].right_deriv.Z_mt <= esternal_der_thresh_check &&
		esternalCompleteDerivativesVector[TR].top_deriv.Z_mt > 0.f && esternalCompleteDerivativesVector[TR].right_deriv.Z_mt > 0.f)
	{
		strong_edge_est_vec[1] = observationsVector[TR] + esternalCompleteDerivativesVector[TR].right_deriv * (1 - indx_c) + esternalCompleteDerivativesVector[TR].top_deriv * indx_r;
	}
	else
	{
		strong_edge_est_vec[1].x_px = observationsVector[TR].x_px + internalCompleteDerivativesVector[TR].left_deriv.x_px * (1 - indx_c) + internalCompleteDerivativesVector[TR].bottom_deriv.x_px * indx_r;
		strong_edge_est_vec[1].y_px = observationsVector[TR].y_px + internalCompleteDerivativesVector[TR].left_deriv.y_px * (1 - indx_c) + internalCompleteDerivativesVector[TR].bottom_deriv.y_px * indx_r;
		strong_edge_est_vec[1].disp = observationsVector[TR].disp;
		strong_edge_est_vec[1].Z_mt = observationsVector[TR].Z_mt;
	}
	
	// Estimation to bottom-left
	/*
	strong_edge_est_vec[2].x_px = observationsVector[BL].x_px + internalCompleteDerivativesVector[BL].right_deriv.x_px * indx_c + internalCompleteDerivativesVector[BL].top_deriv.x_px * (1 - indx_r);
	strong_edge_est_vec[2].y_px = observationsVector[BL].y_px + internalCompleteDerivativesVector[BL].right_deriv.y_px * indx_c + internalCompleteDerivativesVector[BL].top_deriv.y_px * (1 - indx_r);
	strong_edge_est_vec[2].disp = observationsVector[BL].disp;
	strong_edge_est_vec[2].Z_mt = observationsVector[BL].Z_mt;
	*/
	
	if (esternalCompleteDerivativesVector[BL].bottom_deriv.Z_mt <= esternal_der_thresh_check && esternalCompleteDerivativesVector[BL].left_deriv.Z_mt <= esternal_der_thresh_check &&
		esternalCompleteDerivativesVector[BL].bottom_deriv.Z_mt > 0.f && esternalCompleteDerivativesVector[BL].left_deriv.Z_mt > 0.f)
	{
		strong_edge_est_vec[2] = observationsVector[BL] + esternalCompleteDerivativesVector[BL].left_deriv * indx_c + esternalCompleteDerivativesVector[BL].bottom_deriv * (1 - indx_r);
	}
	else
	{
		strong_edge_est_vec[2].x_px = observationsVector[BL].x_px + internalCompleteDerivativesVector[BL].right_deriv.x_px * indx_c + internalCompleteDerivativesVector[BL].top_deriv.x_px * (1 - indx_r);
		strong_edge_est_vec[2].y_px = observationsVector[BL].y_px + internalCompleteDerivativesVector[BL].right_deriv.y_px * indx_c + internalCompleteDerivativesVector[BL].top_deriv.y_px * (1 - indx_r);
		strong_edge_est_vec[2].disp = observationsVector[BL].disp;
		strong_edge_est_vec[2].Z_mt = observationsVector[BL].Z_mt;
	}
	
	// Estimation to bottom-right
	/*
	strong_edge_est_vec[3].x_px = observationsVector[BR].x_px + internalCompleteDerivativesVector[BR].left_deriv.x_px * (1 - indx_c) + internalCompleteDerivativesVector[BR].top_deriv.x_px * (1 - indx_r);
	strong_edge_est_vec[3].y_px = observationsVector[BR].y_px + internalCompleteDerivativesVector[BR].left_deriv.y_px * (1 - indx_c) + internalCompleteDerivativesVector[BR].top_deriv.y_px * (1 - indx_r);
	strong_edge_est_vec[3].disp = observationsVector[BR].disp;
	strong_edge_est_vec[3].Z_mt = observationsVector[BR].Z_mt;
	*/
	
	if (esternalCompleteDerivativesVector[BR].bottom_deriv.Z_mt <= esternal_der_thresh_check && esternalCompleteDerivativesVector[BR].right_deriv.Z_mt <= esternal_der_thresh_check &&
		esternalCompleteDerivativesVector[BR].bottom_deriv.Z_mt > 0.f && esternalCompleteDerivativesVector[BR].right_deriv.Z_mt > 0.f)
	{
		strong_edge_est_vec[3] = observationsVector[BR] + esternalCompleteDerivativesVector[BR].right_deriv * (1 - indx_c) + esternalCompleteDerivativesVector[BR].bottom_deriv * (1 - indx_r);
	}
	else
	{
		strong_edge_est_vec[3].x_px = observationsVector[BR].x_px + internalCompleteDerivativesVector[BR].left_deriv.x_px * (1 - indx_c) + internalCompleteDerivativesVector[BR].top_deriv.x_px * (1 - indx_r);
		strong_edge_est_vec[3].y_px = observationsVector[BR].y_px + internalCompleteDerivativesVector[BR].left_deriv.y_px * (1 - indx_c) + internalCompleteDerivativesVector[BR].top_deriv.y_px * (1 - indx_r);
		strong_edge_est_vec[3].disp = observationsVector[BR].disp;
		strong_edge_est_vec[3].Z_mt = observationsVector[BR].Z_mt;

	}
	
	for (size_t i = 0; i < strong_edge_est_vec.size(); i++)
	{
		pixel_pos_left = { static_cast<int>(strong_edge_est_vec[i].x_px), static_cast<int>(strong_edge_est_vec[i].y_px) };
		pixel_pos_right = pixel_pos_left - cv::Point2i(static_cast<int>(strong_edge_est_vec[i].disp), 0);

		if (pixelInsideImage(pixel_pos_left, pixel_pos_right, left_image_limits, right_image_limits))
		{
			raw_differences[i] = fabsf(im_left.at<float>(pixel_pos_left.y, pixel_pos_left.x) - im_right.at<float>(pixel_pos_right.y, pixel_pos_right.x));
		}
	}
}

void RealGridOperationsTesting::softEdgesEstimations(const int& TL, const int& TR, const int& BL, const int& BR, const float& indx_r, const float& indx_c, const cv::Mat& im_left, const cv::Mat& im_right, std::vector<ObservationData>& soft_edge_est_vec)
{
	EdgeThresholds edg_thresholds;
	float esternal_der_thresh_check = edg_thresholds.strong_depth_threshold;
	// CHANGE VECTORS NAME FROM NO EDGE TO FOR EDGE 
	// CODE CORRECT BUT WRONG CORRESPONDENCES
	cv::Point2i pixel_pos_left = (0, 0);
	cv::Point2i pixel_pos_right = (0, 0);

	// Estimation to top-left
	if (esternalCompleteDerivativesVector[TL].top_deriv.Z_mt <= esternal_der_thresh_check && esternalCompleteDerivativesVector[TL].left_deriv.Z_mt <= esternal_der_thresh_check &&
		esternalCompleteDerivativesVector[TL].top_deriv.Z_mt > 0.f && esternalCompleteDerivativesVector[TL].left_deriv.Z_mt > 0.f)
	{
		// if there is no edges between the corner and the external points use the esternal derivative that have to be correct (--> actually in the same plane - among threshold limits -)
		soft_edge_est_vec[0] = observationsVector[TL] + esternalCompleteDerivativesVector[TL].left_deriv * indx_c + esternalCompleteDerivativesVector[TL].top_deriv * indx_r;
	}
	else
	{
		// Edge also from esternal direction !! // Make estimation using internal derivatives and only for x and y that are not affected by edges
		// Assigned same Z and disparity of the corresponding corner
		soft_edge_est_vec[0].x_px = observationsVector[TL].x_px + internalCompleteDerivativesVector[TL].right_deriv.x_px * indx_c + internalCompleteDerivativesVector[TL].bottom_deriv.x_px * indx_r;
		soft_edge_est_vec[0].y_px = observationsVector[TL].y_px + internalCompleteDerivativesVector[TL].right_deriv.y_px * indx_c + internalCompleteDerivativesVector[TL].bottom_deriv.y_px * indx_r;
		soft_edge_est_vec[0].disp = observationsVector[TL].disp;
		soft_edge_est_vec[0].Z_mt = observationsVector[TL].Z_mt;
	}

	// Estimation to top-right
	if (esternalCompleteDerivativesVector[TR].top_deriv.Z_mt <= esternal_der_thresh_check && esternalCompleteDerivativesVector[TR].right_deriv.Z_mt <= esternal_der_thresh_check &&
		esternalCompleteDerivativesVector[TR].top_deriv.Z_mt > 0.f && esternalCompleteDerivativesVector[TR].right_deriv.Z_mt > 0.f)
	{
		soft_edge_est_vec[1] = observationsVector[TR] + esternalCompleteDerivativesVector[TR].right_deriv * (1 - indx_c) + esternalCompleteDerivativesVector[TR].top_deriv * indx_r;
	}
	else
	{
		soft_edge_est_vec[1].x_px = observationsVector[TR].x_px + internalCompleteDerivativesVector[TR].left_deriv.x_px * (1 - indx_c) + internalCompleteDerivativesVector[TR].bottom_deriv.x_px * indx_r;
		soft_edge_est_vec[1].y_px = observationsVector[TR].y_px + internalCompleteDerivativesVector[TR].left_deriv.y_px * (1 - indx_c) + internalCompleteDerivativesVector[TR].bottom_deriv.y_px * indx_r;
		soft_edge_est_vec[1].disp = observationsVector[TR].disp;
		soft_edge_est_vec[1].Z_mt = observationsVector[TR].Z_mt;
	}

	// Estimation to bottom-left
	if (esternalCompleteDerivativesVector[BL].bottom_deriv.Z_mt <= esternal_der_thresh_check && esternalCompleteDerivativesVector[BL].left_deriv.Z_mt <= esternal_der_thresh_check &&
		esternalCompleteDerivativesVector[BL].bottom_deriv.Z_mt > 0.f && esternalCompleteDerivativesVector[BL].left_deriv.Z_mt > 0.f)
	{
		soft_edge_est_vec[2] = observationsVector[BL] + esternalCompleteDerivativesVector[BL].left_deriv * indx_c + esternalCompleteDerivativesVector[BL].bottom_deriv * (1 - indx_r);
	}
	else
	{
		soft_edge_est_vec[2].x_px = observationsVector[BL].x_px + internalCompleteDerivativesVector[BL].right_deriv.x_px * indx_c + internalCompleteDerivativesVector[BL].top_deriv.x_px * (1 - indx_r);
		soft_edge_est_vec[2].y_px = observationsVector[BL].y_px + internalCompleteDerivativesVector[BL].right_deriv.y_px * indx_c + internalCompleteDerivativesVector[BL].top_deriv.y_px * (1 - indx_r);
		soft_edge_est_vec[2].disp = observationsVector[BL].disp;
		soft_edge_est_vec[2].Z_mt = observationsVector[BL].Z_mt;
	}

	// Estimation to bottom-right
	if (esternalCompleteDerivativesVector[BR].bottom_deriv.Z_mt <= esternal_der_thresh_check && esternalCompleteDerivativesVector[BR].right_deriv.Z_mt <= esternal_der_thresh_check && 
		esternalCompleteDerivativesVector[BR].bottom_deriv.Z_mt > 0.f && esternalCompleteDerivativesVector[BR].right_deriv.Z_mt > 0.f)
	{
		soft_edge_est_vec[3] = observationsVector[BR] + esternalCompleteDerivativesVector[BR].right_deriv * (1 - indx_c) + esternalCompleteDerivativesVector[BR].bottom_deriv * (1 - indx_r);
	}
	else
	{
		soft_edge_est_vec[3].x_px = observationsVector[BR].x_px + internalCompleteDerivativesVector[BR].left_deriv.x_px * (1 - indx_c) + internalCompleteDerivativesVector[BR].top_deriv.x_px * (1 - indx_r);
		soft_edge_est_vec[3].y_px = observationsVector[BR].y_px + internalCompleteDerivativesVector[BR].left_deriv.y_px * (1 - indx_c) + internalCompleteDerivativesVector[BR].top_deriv.y_px * (1 - indx_r);
		soft_edge_est_vec[3].disp = observationsVector[BR].disp;
		soft_edge_est_vec[3].Z_mt = observationsVector[BR].Z_mt;

	}

	for (size_t i = 0; i < soft_edge_est_vec.size(); i++)
	{
		pixel_pos_left = { static_cast<int>(soft_edge_est_vec[i].x_px), static_cast<int>(soft_edge_est_vec[i].y_px) };
		pixel_pos_right = pixel_pos_left - cv::Point2i(static_cast<int>(soft_edge_est_vec[i].disp), 0);

		if (pixelInsideImage(pixel_pos_left, pixel_pos_right, left_image_limits, right_image_limits))
		{
			raw_differences[i] = fabsf(im_left.at<float>(pixel_pos_left.y, pixel_pos_left.x) - im_right.at<float>(pixel_pos_right.y, pixel_pos_right.x));
		}
	}
}

void RealGridOperationsTesting::fillSGMCostCube(const int& TL, const int& TR, const int& BL, const int& BR, const int& cost_cube_indx, const cv::Mat& image_left, const cv::Mat& image_right, const std::vector<float>& patch_disparity_values_)
{
	int channels = patch_disparity_values_.size();
	cv::Point2i temp_pixel_left{};
	cv::Point2i temp_pixel_right{};

	unsigned int left_binary_pixel_value{};
	unsigned int right_binary_pixel_value{};
	unsigned int pixel_match{};
	float pixel_cost = 0.f;

	for (int i = 0; i < avg_square_size_4eye; i++)
	{
		for (int j = 0; j < avg_square_size_4eye; j++)
		{
			for (int d = 0; d < patch_disparity_values_.size(); d++)
			{
				temp_pixel_left = {static_cast<int>(observationsVector[TL].x_px) + i, static_cast<int>(observationsVector[TL].y_px) + j};
				temp_pixel_right = temp_pixel_left - cv::Point2i(static_cast<int>(patch_disparity_values_[d], 0));

				centerSymmetricCensus(temp_pixel_left, image_left, window_size, left_binary_pixel_value);
				centerSymmetricCensus(temp_pixel_right, image_right, window_size, right_binary_pixel_value);
				pixel_match = left_binary_pixel_value ^ right_binary_pixel_value;
				pixel_cost = float(countSetBitsSGM(pixel_match));

				SGMCostCubesVector[cost_cube_indx].ptr<float>(i)[channels * j + d] = pixel_cost;
			}
		}
	}
}

void RealGridOperationsTesting::noEdgesSGMEstimations(const int& TL, const int& TR, const int& BL, const int& BR, const int& cost_cube_indx, const std::vector<float>& patch_disparity_values_, int& vector_indx)
{
	int channels = patch_disparity_values_.size();
	std::vector<float> temp_cost_values_vector(channels);

	for (int i = 0; i < avg_square_size_4eye; i++)
	{
		for (int j = 0; j < avg_square_size_4eye; j++)
		{
			for (int d = 0; d < patch_disparity_values_.size(); d++)
			{
				temp_cost_values_vector[d] = SGMCostCubesVector[cost_cube_indx].ptr<float>(i)[channels * j + d];
			}

			int min_cost_indx = std::min_element(temp_cost_values_vector.begin(), temp_cost_values_vector.end()) - temp_cost_values_vector.begin();
			SGMEstimatedDisparity[vector_indx] = patch_disparity_values_[min_cost_indx];
			SGMEstimated_Z[vector_indx] = baseline * camera_matrix_left.at<float>(0,0) / (patch_disparity_values_[min_cost_indx]);
			SGMEstimated_X[vector_indx] = (observationsVector[TL].x_px + static_cast<float>(j) - camera_matrix_left.at<float>(0, 2)) * SGMEstimated_Z[vector_indx] / camera_matrix_left.at<float>(0, 0);
			SGMEstimated_Y[vector_indx] = (observationsVector[TL].y_px + static_cast<float>(i) - camera_matrix_left.at<float>(1, 2)) * SGMEstimated_Z[vector_indx] / camera_matrix_left.at<float>(1, 1);
			SGMEstimated_x_px[vector_indx] = (observationsVector[TL].x_px + static_cast<float>(j));
			SGMEstimated_y_px[vector_indx] = (observationsVector[TL].y_px + static_cast<float>(i));

			vector_indx++;
		}
	}
}

void RealGridOperationsTesting::verticalEdgesSGMEstimations(const int& TL, const int& TR, const int& BL, const int& BR, const int& cost_cube_indx, const std::vector<float>& patch_disparity_values_, int& vector_indx)
{
	int channels = patch_disparity_values_.size();
	cv::Size aggregation_cost_size = cv::Size(avg_square_size_4eye, avg_square_size_4eye);
	cv::Mat direction_cost_cube_0 = cv::Mat(aggregation_cost_size, CV_32FC4, cv::Scalar::all(0.f));
	cv::Mat direction_cost_cube_2 = cv::Mat(aggregation_cost_size, CV_32FC4, cv::Scalar::all(0.f));
	cv::Mat total_cost_cube = cv::Mat(aggregation_cost_size, CV_32FC4, cv::Scalar::all(0.f));

	// Direction 0 --> FROM LEFT
	aggregationCostDirection_0(cost_cube_indx, direction_cost_cube_0);

	// Direction 2 --> FROM RIGHT
	aggregationCostDirection_2(cost_cube_indx, direction_cost_cube_2);

	total_cost_cube = direction_cost_cube_0 + direction_cost_cube_2;
	int index_min_cost = 0;
	for (int i = 0; i < total_cost_cube.rows; i++)
	{
		for (int j = 0; j < total_cost_cube.cols; j++)
		{
			index_min_cost = minElementThirdDimMatIndex(total_cost_cube, i, j);
			SGMEstimatedDisparity[vector_indx] = patch_disparity_values_[index_min_cost];
			SGMEstimated_Z[vector_indx] = baseline * camera_matrix_left.at<float>(0, 0) / (patch_disparity_values_[index_min_cost]);
			SGMEstimated_X[vector_indx] = (observationsVector[TL].x_px + static_cast<float>(j) - camera_matrix_left.at<float>(0, 2)) * SGMEstimated_Z[vector_indx] / camera_matrix_left.at<float>(0, 0);;
			SGMEstimated_Y[vector_indx] = (observationsVector[TL].y_px + static_cast<float>(i) - camera_matrix_left.at<float>(1, 2)) * SGMEstimated_Z[vector_indx] / camera_matrix_left.at<float>(1, 1);;
			SGMEstimated_x_px[vector_indx] = (observationsVector[TL].x_px + static_cast<float>(j));
			SGMEstimated_y_px[vector_indx] = (observationsVector[TL].y_px + static_cast<float>(i));
			vector_indx++;
		}
	}
}

void RealGridOperationsTesting::horizontalEdgesSGMEstimations(const int& TL, const int& TR, const int& BL, const int& BR, const int& cost_cube_indx, const std::vector<float>& patch_disparity_values_, int& vector_indx)
{
	int channels = patch_disparity_values_.size();
	cv::Size aggregation_cost_size = cv::Size(avg_square_size_4eye, avg_square_size_4eye);
	cv::Mat direction_cost_cube_1 = cv::Mat(aggregation_cost_size, CV_32FC4, cv::Scalar::all(0.f));
	cv::Mat direction_cost_cube_3 = cv::Mat(aggregation_cost_size, CV_32FC4, cv::Scalar::all(0.f));
	cv::Mat total_cost_cube = cv::Mat(aggregation_cost_size, CV_32FC4, cv::Scalar::all(0.f));

	// Direction 1 --> FROM TOP
	aggregationCostDirection_1(cost_cube_indx, direction_cost_cube_1);

	// Direction 3 --> FROM BOTTOM
	aggregationCostDirection_3(cost_cube_indx, direction_cost_cube_3);

	total_cost_cube = direction_cost_cube_1 + direction_cost_cube_3;
	int index_min_cost = 0;
	for (int i = 0; i < total_cost_cube.rows; i++)
	{
		for (int j = 0; j < total_cost_cube.cols; j++)
		{
			index_min_cost = minElementThirdDimMatIndex(total_cost_cube, i, j);
			SGMEstimatedDisparity[vector_indx] = patch_disparity_values_[index_min_cost];
			SGMEstimated_Z[vector_indx] = baseline * camera_matrix_left.at<float>(0, 0) / (patch_disparity_values_[index_min_cost]);
			SGMEstimated_X[vector_indx] = (observationsVector[TL].x_px + static_cast<float>(j) - camera_matrix_left.at<float>(0, 2)) * SGMEstimated_Z[vector_indx] / camera_matrix_left.at<float>(0, 0);
			SGMEstimated_Y[vector_indx] = (observationsVector[TL].y_px + static_cast<float>(i) - camera_matrix_left.at<float>(1, 2)) * SGMEstimated_Z[vector_indx] / camera_matrix_left.at<float>(1, 1);
			SGMEstimated_x_px[vector_indx] = (observationsVector[TL].x_px + static_cast<float>(j));
			SGMEstimated_y_px[vector_indx] = (observationsVector[TL].y_px + static_cast<float>(i));
			vector_indx++;
		}
	}
}

void RealGridOperationsTesting::undefinedEdgesSGMEstimations(const int& TL, const int& TR, const int& BL, const int& BR, const int& cost_cube_indx, const std::vector<float>& patch_disparity_values_, int& vector_indx)
{
	int channels = patch_disparity_values_.size();
	cv::Size aggregation_cost_size = cv::Size(avg_square_size_4eye, avg_square_size_4eye);
	cv::Mat direction_cost_cube_0 = cv::Mat(aggregation_cost_size, CV_32FC4, cv::Scalar::all(0.f));
	cv::Mat direction_cost_cube_1 = cv::Mat(aggregation_cost_size, CV_32FC4, cv::Scalar::all(0.f));
	cv::Mat direction_cost_cube_2 = cv::Mat(aggregation_cost_size, CV_32FC4, cv::Scalar::all(0.f));
	cv::Mat direction_cost_cube_3 = cv::Mat(aggregation_cost_size, CV_32FC4, cv::Scalar::all(0.f));
	cv::Mat total_cost_cube = cv::Mat(aggregation_cost_size, CV_32FC4, cv::Scalar::all(0.f));

	// Direction 0 --> FROM LEFT
	aggregationCostDirection_0(cost_cube_indx, direction_cost_cube_0);

	// Direction 2 --> FROM RIGHT
	aggregationCostDirection_2(cost_cube_indx, direction_cost_cube_2);

	// Direction 1 --> FROM TOP
	aggregationCostDirection_0(cost_cube_indx, direction_cost_cube_1);

	// Direction 3 --> FROM BOTTOM
	aggregationCostDirection_2(cost_cube_indx, direction_cost_cube_3);

	total_cost_cube = direction_cost_cube_0 + direction_cost_cube_1 + direction_cost_cube_2 + direction_cost_cube_3;
	
	int index_min_cost = 0;
	for (int i = 0; i < total_cost_cube.rows; i++)
	{
		for (int j = 0; j < total_cost_cube.cols; j++)
		{
			index_min_cost = minElementThirdDimMatIndex(total_cost_cube, i, j);
			SGMEstimatedDisparity[vector_indx] = patch_disparity_values_[index_min_cost];
			SGMEstimated_Z[vector_indx] = baseline * camera_matrix_left.at<float>(0, 0) / (patch_disparity_values_[index_min_cost]);
			SGMEstimated_X[vector_indx] = (observationsVector[TL].x_px + static_cast<float>(j) - camera_matrix_left.at<float>(0, 2)) * SGMEstimated_Z[vector_indx] / camera_matrix_left.at<float>(0, 0);
			SGMEstimated_Y[vector_indx] = (observationsVector[TL].y_px + static_cast<float>(i) - camera_matrix_left.at<float>(1, 2)) * SGMEstimated_Z[vector_indx] / camera_matrix_left.at<float>(1, 1);
			SGMEstimated_x_px[vector_indx] = (observationsVector[TL].x_px + static_cast<float>(j));
			SGMEstimated_y_px[vector_indx] = (observationsVector[TL].y_px + static_cast<float>(i));
			vector_indx++;
		}
	}

}

void RealGridOperationsTesting::aggregationCostDirection_0(const int& cost_cube_indx, cv::Mat& direction_cost_cube_0)
{
	int channels = direction_cost_cube_0.channels();
	SGMCostCubesVector[cost_cube_indx].col(0).copyTo(direction_cost_cube_0.col(0));

	float current_cost = 0.f;
	float direction_cost_L1 = 0.f;
	float direction_cost_L2 = 0.f;
	float direction_cost_L3 = 0.f;
	float direction_cost_L4 = 0.f;
	
	float min_cost_all_disparity_per_pixel = 0.f;
	float min_cost_per_disparity_per_pixel = 0.f;

	std::vector<float> cost_array_pixel_disp{};

	for (int i = 0; i < direction_cost_cube_0.rows; i++)
	{
		for (int j = 1; j < direction_cost_cube_0.cols; j++)
		{
			for (int d = 0; d < direction_cost_cube_0.channels(); d++)
			{
				current_cost = SGMCostCubesVector[cost_cube_indx].ptr<float>(i)[channels * j + d];
				direction_cost_L1 = direction_cost_cube_0.ptr<float>(i)[channels * (j - 1) + d];
				min_cost_all_disparity_per_pixel = minElementThirdDimMat(SGMCostCubesVector[cost_cube_indx], i, j - 1);
				direction_cost_L4 = min_cost_all_disparity_per_pixel + P2_sgm;

				if (d == 0)
				{
					direction_cost_L3 = direction_cost_cube_0.ptr<float>(i)[channels * (j - 1) + (d + 1)] + P1_sgm;
					cost_array_pixel_disp = { direction_cost_L1 , direction_cost_L3 , direction_cost_L4 };
					min_cost_per_disparity_per_pixel = *std::min_element(cost_array_pixel_disp.begin(), cost_array_pixel_disp.end());
					direction_cost_cube_0.ptr<float>(i)[channels * j + d] = current_cost + min_cost_per_disparity_per_pixel - min_cost_all_disparity_per_pixel;
				}
				if (d == channels - 1)
				{
					direction_cost_L2 = direction_cost_cube_0.ptr<float>(i)[channels * (j - 1) + (d - 1)] + P1_sgm;
					cost_array_pixel_disp = { direction_cost_L1 , direction_cost_L2 , direction_cost_L4 };
					min_cost_per_disparity_per_pixel = *std::min_element(cost_array_pixel_disp.begin(), cost_array_pixel_disp.end());
					direction_cost_cube_0.ptr<float>(i)[channels * j + d] = current_cost + min_cost_per_disparity_per_pixel - min_cost_all_disparity_per_pixel;
				}
				else
				{
					direction_cost_L2 = direction_cost_cube_0.ptr<float>(i)[channels * (j - 1) + (d - 1)] + P1_sgm;
					direction_cost_L3 = direction_cost_cube_0.ptr<float>(i)[channels * (j - 1) + (d + 1)] + P1_sgm;
					cost_array_pixel_disp = { direction_cost_L1 , direction_cost_L2, direction_cost_L2 , direction_cost_L4 };
					min_cost_per_disparity_per_pixel = *std::min_element(cost_array_pixel_disp.begin(), cost_array_pixel_disp.end());
					direction_cost_cube_0.ptr<float>(i)[channels * j + d] = current_cost + min_cost_per_disparity_per_pixel - min_cost_all_disparity_per_pixel;
				}
			}
		}
	}
}

void RealGridOperationsTesting::aggregationCostDirection_2(const int& cost_cube_indx, cv::Mat& direction_cost_cube_2)
{
	int channels = direction_cost_cube_2.channels();
	SGMCostCubesVector[cost_cube_indx].col(direction_cost_cube_2.cols - 1).copyTo(direction_cost_cube_2.col(direction_cost_cube_2.cols - 1));

	float current_cost = 0.f;
	float direction_cost_L1 = 0.f;
	float direction_cost_L2 = 0.f;
	float direction_cost_L3 = 0.f;
	float direction_cost_L4 = 0.f;

	float min_cost_all_disparity_per_pixel = 0.f;
	float min_cost_per_disparity_per_pixel = 0.f;

	std::vector<float> cost_array_pixel_disp{};

	for (int i = 0; i < direction_cost_cube_2.rows; i++)
	{
		for (int j = direction_cost_cube_2.cols - 2; j >= 0; j--)
		{
			for (int d = 0; d < direction_cost_cube_2.channels(); d++)
			{
				current_cost = SGMCostCubesVector[cost_cube_indx].ptr<float>(i)[channels * j + d];
				direction_cost_L1 = direction_cost_cube_2.ptr<float>(i)[channels * (j + 1) + d];
				min_cost_all_disparity_per_pixel = minElementThirdDimMat(SGMCostCubesVector[cost_cube_indx], i, j + 1);
				direction_cost_L4 = min_cost_all_disparity_per_pixel + P2_sgm;

				if (d == 0)
				{
					direction_cost_L3 = direction_cost_cube_2.ptr<float>(i)[channels * (j + 1) + (d + 1)] + P1_sgm;
					cost_array_pixel_disp = { direction_cost_L1 , direction_cost_L3 , direction_cost_L4 };
					min_cost_per_disparity_per_pixel = *std::min_element(cost_array_pixel_disp.begin(), cost_array_pixel_disp.end());
					direction_cost_cube_2.ptr<float>(i)[channels * j + d] = current_cost + min_cost_per_disparity_per_pixel - min_cost_all_disparity_per_pixel;
				}
				if (d == channels - 1)
				{
					direction_cost_L2 = direction_cost_cube_2.ptr<float>(i)[channels * (j + 1) + (d + 1)] + P1_sgm;
					cost_array_pixel_disp = { direction_cost_L1 , direction_cost_L2 , direction_cost_L4 };
					min_cost_per_disparity_per_pixel = *std::min_element(cost_array_pixel_disp.begin(), cost_array_pixel_disp.end());
					direction_cost_cube_2.ptr<float>(i)[channels * j + d] = current_cost + min_cost_per_disparity_per_pixel - min_cost_all_disparity_per_pixel;
				}
				else
				{
					direction_cost_L2 = direction_cost_cube_2.ptr<float>(i)[channels * (j + 1) + (d - 1)] + P1_sgm;
					direction_cost_L3 = direction_cost_cube_2.ptr<float>(i)[channels * (j + 1) + (d + 1)] + P1_sgm;
					cost_array_pixel_disp = { direction_cost_L1 , direction_cost_L2, direction_cost_L2 , direction_cost_L4 };
					min_cost_per_disparity_per_pixel = *std::min_element(cost_array_pixel_disp.begin(), cost_array_pixel_disp.end());
					direction_cost_cube_2.ptr<float>(i)[channels * j + d] = current_cost + min_cost_per_disparity_per_pixel - min_cost_all_disparity_per_pixel;
				}
			}
		}
	}
}

void RealGridOperationsTesting::aggregationCostDirection_1(const int& cost_cube_indx, cv::Mat& direction_cost_cube_1)
{
	int channels = direction_cost_cube_1.channels();
	SGMCostCubesVector[cost_cube_indx].row(0).copyTo(direction_cost_cube_1.row(0));

	float current_cost = 0.f;
	float direction_cost_L1 = 0.f;
	float direction_cost_L2 = 0.f;
	float direction_cost_L3 = 0.f;
	float direction_cost_L4 = 0.f;

	float min_cost_all_disparity_per_pixel = 0.f;
	float min_cost_per_disparity_per_pixel = 0.f;

	std::vector<float> cost_array_pixel_disp{};

	for (int j = 0; j < direction_cost_cube_1.cols; j++)
	{
		for (int i = 1; i < direction_cost_cube_1.rows; i++)
		{
			for (int d = 0; d < direction_cost_cube_1.channels(); d++)
			{
				current_cost = SGMCostCubesVector[cost_cube_indx].ptr<float>(i)[channels * j + d];
				direction_cost_L1 = direction_cost_cube_1.ptr<float>(i - 1)[channels * j + d];
				min_cost_all_disparity_per_pixel = minElementThirdDimMat(SGMCostCubesVector[cost_cube_indx], i - 1, j);
				direction_cost_L4 = min_cost_all_disparity_per_pixel + P2_sgm;

				if (d == 0)
				{
					direction_cost_L3 = direction_cost_cube_1.ptr<float>(i - 1)[channels * j + (d + 1)] + P1_sgm;
					cost_array_pixel_disp = { direction_cost_L1 , direction_cost_L3 , direction_cost_L4 };
					min_cost_per_disparity_per_pixel = *std::min_element(cost_array_pixel_disp.begin(), cost_array_pixel_disp.end());
					direction_cost_cube_1.ptr<float>(i)[channels * j + d] = current_cost + min_cost_per_disparity_per_pixel - min_cost_all_disparity_per_pixel;
				}
				if (d == channels - 1)
				{
					direction_cost_L2 = direction_cost_cube_1.ptr<float>(i - 1)[channels * j + (d - 1)] + P1_sgm;
					cost_array_pixel_disp = { direction_cost_L1 , direction_cost_L2 , direction_cost_L4 };
					min_cost_per_disparity_per_pixel = *std::min_element(cost_array_pixel_disp.begin(), cost_array_pixel_disp.end());
					direction_cost_cube_1.ptr<float>(i)[channels * j + d] = current_cost + min_cost_per_disparity_per_pixel - min_cost_all_disparity_per_pixel;
				}
				else
				{
					direction_cost_L2 = direction_cost_cube_1.ptr<float>(i - 1)[channels * j + (d - 1)] + P1_sgm;
					direction_cost_L3 = direction_cost_cube_1.ptr<float>(i - 1)[channels * j + (d + 1)] + P1_sgm;
					cost_array_pixel_disp = { direction_cost_L1 , direction_cost_L2, direction_cost_L2 , direction_cost_L4 };
					min_cost_per_disparity_per_pixel = *std::min_element(cost_array_pixel_disp.begin(), cost_array_pixel_disp.end());
					direction_cost_cube_1.ptr<float>(i)[channels * j + d] = current_cost + min_cost_per_disparity_per_pixel - min_cost_all_disparity_per_pixel;
				}
			}
		}
	}
}

void RealGridOperationsTesting::aggregationCostDirection_3(const int& cost_cube_indx, cv::Mat& direction_cost_cube_3)
{
	int channels = direction_cost_cube_3.channels();
	SGMCostCubesVector[cost_cube_indx].col(direction_cost_cube_3.rows - 1).copyTo(direction_cost_cube_3.col(direction_cost_cube_3.rows - 1));

	float current_cost = 0.f;
	float direction_cost_L1 = 0.f;
	float direction_cost_L2 = 0.f;
	float direction_cost_L3 = 0.f;
	float direction_cost_L4 = 0.f;

	float min_cost_all_disparity_per_pixel = 0.f;
	float min_cost_per_disparity_per_pixel = 0.f;

	std::vector<float> cost_array_pixel_disp{};

	for (int j = 0; j < direction_cost_cube_3.cols; j++)
	{
		for (int i = direction_cost_cube_3.rows - 2; i >= 0; i--)
		{
			for (int d = 0; d < direction_cost_cube_3.channels(); d++)
			{
				current_cost = SGMCostCubesVector[cost_cube_indx].ptr<float>(i)[channels * j + d];
				direction_cost_L1 = direction_cost_cube_3.ptr<float>(i + 1)[channels * j + d];
				min_cost_all_disparity_per_pixel = minElementThirdDimMat(SGMCostCubesVector[cost_cube_indx], i + 1, j);
				direction_cost_L4 = min_cost_all_disparity_per_pixel + P2_sgm;

				if (d == 0)
				{
					direction_cost_L3 = direction_cost_cube_3.ptr<float>(i + 1)[channels * j + (d + 1)] + P1_sgm;
					cost_array_pixel_disp = { direction_cost_L1 , direction_cost_L3 , direction_cost_L4 };
					min_cost_per_disparity_per_pixel = *std::min_element(cost_array_pixel_disp.begin(), cost_array_pixel_disp.end());
					direction_cost_cube_3.ptr<float>(i)[channels * j + d] = current_cost + min_cost_per_disparity_per_pixel - min_cost_all_disparity_per_pixel;
				}
				if (d == channels - 1)
				{
					direction_cost_L2 = direction_cost_cube_3.ptr<float>(i + 1)[channels * j + (d + 1)] + P1_sgm;
					cost_array_pixel_disp = { direction_cost_L1 , direction_cost_L2 , direction_cost_L4 };
					min_cost_per_disparity_per_pixel = *std::min_element(cost_array_pixel_disp.begin(), cost_array_pixel_disp.end());
					direction_cost_cube_3.ptr<float>(i)[channels * j + d] = current_cost + min_cost_per_disparity_per_pixel - min_cost_all_disparity_per_pixel;
				}
				else
				{
					direction_cost_L2 = direction_cost_cube_3.ptr<float>(i + 1)[channels * j + (d - 1)] + P1_sgm;
					direction_cost_L3 = direction_cost_cube_3.ptr<float>(i + 1)[channels * j + (d + 1)] + P1_sgm;
					cost_array_pixel_disp = { direction_cost_L1 , direction_cost_L2, direction_cost_L2 , direction_cost_L4 };
					min_cost_per_disparity_per_pixel = *std::min_element(cost_array_pixel_disp.begin(), cost_array_pixel_disp.end());
					direction_cost_cube_3.ptr<float>(i)[channels * j + d] = current_cost + min_cost_per_disparity_per_pixel - min_cost_all_disparity_per_pixel;
				}
			}
		}
	}
}



