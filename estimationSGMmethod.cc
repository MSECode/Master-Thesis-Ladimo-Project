#include "estimationSGMmethod.h"

void getEstimationSGMbased(const cv::Mat& lookupTable, const cv::Mat& xydMatrix, const cv::Mat& XYZMatrix, const cv::Mat& badPoints, const cv::Mat& left_image, const cv::Mat& right_image, const cv::Mat& camera_matrix_left, const cv::Mat& camera_matrix_right, const std::vector<TotalDerivatives>& derivative_vector, const int& sampl_factor, const float& baseline, const float& doffs, cv::Mat& estimation_disparity, cv::Mat& estimation_X, cv::Mat& estimation_Y, cv::Mat& estimation_Z)
{
	// Parameters to use 
	float disparity_edge_threshold = 10.f;              // disparity threshold used to find edges (if not declared in the function the default value is 15.f)
	float step = float(1.f / sampl_factor);             // gap between the points that we are going to sample among the 4 given corners
	int range_no_edge = 3;                              // range for SGM matching cost if there's no edges
	int range_yes_edge = 10;                            // range for SGM matching cost if thre's edges
	int window_size = 5;                                // window size for census/rank matching cost computation
	int half_window = round(window_size / 2);

	// index initialization
	int c = 0;
	int r = 0;
	int k = 0;

	// matrices with limits
	// left image limits
	std::array<std::array<int, 2>, 2> image_limits_array_l = { { { 0, left_image.rows}, {0, left_image.cols} } };
	cv::Mat left_image_limits_mat = cv::Mat(2, 2, CV_32S, image_limits_array_l.data());

	// right image limits
	std::array<std::array<int, 2>, 2> image_limits_array_r = { { { 0, right_image.rows}, {0, right_image.cols} } };
	cv::Mat right_image_limits_mat = cv::Mat(2, 2, CV_32S, image_limits_array_r.data());

	// left SGM limits no edges
	std::array<std::array<int, 2>, 2> SGM_limits_array_l = { { { 0, left_image.rows}, {0, left_image.cols} } };
	cv::Mat left_SGM_limits_mat = cv::Mat(2, 2, CV_32S, SGM_limits_array_l.data());

	// right SGM limits no edges
	std::array<std::array<int, 2>, 2> SGM_limits_array_r = { { { 0, right_image.rows}, {range_no_edge,  right_image.cols - range_no_edge} } };
	cv::Mat right_SGM_limits_mat = cv::Mat(2, 2, CV_32S, SGM_limits_array_r.data());

	// left SGM limits edges
	std::array<std::array<int, 2>, 2> SGM_limits_edges_array_l = { { { half_window, left_image.rows - half_window}, {half_window, left_image.cols - half_window} } };
	cv::Mat left_SGM_limits_mat_edges = cv::Mat(2, 2, CV_32S, SGM_limits_edges_array_l.data());

	// right SGM limits edges
	std::array<std::array<int, 2>, 2> SGM_limits_edges_array_r = { { { half_window, left_image.rows - half_window}, {range_yes_edge + half_window, left_image.cols - range_yes_edge - half_window} } };
	cv::Mat right_SGM_limits_mat_edges = cv::Mat(2, 2, CV_32S, SGM_limits_edges_array_r.data());


	// inizialization of matrix and vectors (get the memory space from the beginning)
	std::vector<cv::Vec2i> pixel_position_left_vec(4);
	std::vector<cv::Vec2i> pixel_position_right_vec(4);
	std::vector<float> disparity_levels_edges(4);
	std::vector<cv::Vec4f> estimation_vector(4);
	std::vector<float> raw_differences(4);

	std::pair<std::vector<cv::Vec2i>, std::vector<cv::Vec2i> > pixel_positions_left_right_vector;
	pixel_positions_left_right_vector = std::make_pair(pixel_position_left_vec, pixel_position_right_vec);

	///////////////////////////////////////

	for (int i = 0; i < lookupTable.rows - 1; i++)
	{
		for (int j = 0; j < lookupTable.cols - 1; j++)
		{
			c = 0;
			// Corners of the window
			const int TL = lookupTable.at<int>(i, j);
			const int TR = lookupTable.at<int>(i, j + 1);
			const int BL = lookupTable.at<int>(i + 1, j);
			const int BR = lookupTable.at<int>(i + 1, j + 1);

			// Check if there's an edge between the corner points (vert, hor, undef)
			// Done to adapt the SGM strategy
			EdgeDirection edg = EdgeDirection::undefined;
			findEdge_deriv(TL, TR, BL, BR, XYZMatrix, edg, disparity_edge_threshold);
			if (findEdge_deriv(TL, TR, BL, BR, XYZMatrix, edg, disparity_edge_threshold) == false)
			{
				// case no edges --> simple case --> use small range and estimation only w.r.t. the TL corner because the disparity values will be very similar
				// before proceeding check that the corners are all good points otherwise all the subsample estimations will be bad
				if (allBadPoints(TL, TR, BL, BR, badPoints) == false)
				{
					//The points are all good
					// apply simple strategy for no edges and good points
					// Only one estimation should be necessary because they will be all similar considering the disparity are almost the same
					for (float ii = 0; ii <= 1; ii += step)
					{
						for (float jj = 0; jj <= 1; jj += step)
						{
							estimation_vector = noEdgesAllEstimationsVector(TL, TR, BL, BR, ii, jj, XYZMatrix, lookupTable, badPoints, derivative_vector, estimation_vector);
							k = 0;
							for each (cv::Vec4f vec in estimation_vector)
							{
								cv::Vec2f est_left_image = project3DToImage(camera_matrix_left, vec);
								cv::Vec2f est_right_image = est_left_image - cv::Vec2f(vec[3], 0.0);
								pixel_position_left_vec[k] = ToVec2i(est_left_image);
								pixel_position_right_vec[k] = ToVec2i(est_right_image);
							}

							// calculate cost
							// check that pixel left and correspondent right are inside the image 
							for (int h = 0; h < pixel_position_left_vec.size(); h++)
							{
								if (pixelInsideImage(pixel_position_left_vec[h], pixel_position_right_vec[h], left_image_limits_mat, right_image_limits_mat) == true) {
									// make row comparison evaluating the absolute difference between pixel's intensity values
									raw_differences[h] = abs(left_image.at<float>(pixel_position_left_vec[h][0], pixel_position_left_vec[h][1]) - right_image.at<float>(pixel_position_right_vec[h][0], pixel_position_right_vec[h][1]));
								}
								else
								{
									// set very high value for the raw difference thus it will never be taken by the min function
									raw_differences.push_back(std::numeric_limits<float>::infinity());
								}
							}

							// Evaluate the best estimation
							int best_value_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							float best_value = estimation_vector[best_value_index][3];
							estimation_disparity.at<float>(r, c) = best_value;
							estimation_Z.at<float>(r, c) = baseline * camera_matrix_left.at<float>(0, 0) / (estimation_disparity.at<float>(r, c) + doffs);
							estimation_X.at<float>(r, c) = estimation_vector[best_value_index][0];
							estimation_Y.at<float>(r, c) = estimation_vector[best_value_index][1];

							c++;
						}
					}
				}
				else
				{
					// Shitty window
					// all the estimations inside the window will be bad
					// think how to set the values for further selection (not possible to discard otherwise problem with final matrix)
					estimation_X.row(r) = -1.f;
					estimation_Y.row(r) = -1.f;
					estimation_Z.row(r) = -1.f;
					estimation_disparity.row(r) = -1.f;
				}

			}
			else
			{
				// inizialize some internal vectors
				raw_differences.clear();
				raw_differences.resize(4);

				// Penalty forcing image continuity
				float P1 = 5.f;
				float P2 = 10.f;
				float P3 = 15.f;
				// There is an edge
				// With the derivative it is possible to identify if vertical, horizontal or we are in an undefined case
				if (edg == EdgeDirection::vertical)
				{
					// Vertical edge
					// Penalty if estimation different from closer corner point considering the edge
					for (float ii = 0; ii <= 1; ii += step)
					{
						for (float jj = 0; jj <= 1; jj += step)
						{

							pixel_positions_left_right_vector = yesEdgesAllEstimationsPositionsVectors(TL, TR, BL, BR, ii, jj, camera_matrix_left, XYZMatrix, lookupTable, derivative_vector, estimation_vector, pixel_positions_left_right_vector);

							// calculate cost
							// check that pixel left and correspondent right are inside the image 
							// set all the values to infinity (if pixel not inside image the value will not be updated)
							std::fill(raw_differences.begin(), raw_differences.end(), std::numeric_limits<float>::infinity());

							
							if (pixelInsideImage(pixel_positions_left_right_vector.first[0], pixel_positions_left_right_vector.second[0], left_image_limits_mat, right_image_limits_mat) == true) {
								raw_differences[0] = abs(left_image.at<float>(pixel_positions_left_right_vector.first[0][0], pixel_positions_left_right_vector.first[0][1]) - right_image.at<float>(pixel_positions_left_right_vector.second[0][0], pixel_positions_left_right_vector.second[0][1])) + jj * P2 + ii * P1;
							}
							if (pixelInsideImage(pixel_positions_left_right_vector.first[1], pixel_positions_left_right_vector.second[1], left_image_limits_mat, right_image_limits_mat) == true)
							{
								raw_differences[1] = abs(left_image.at<float>(pixel_positions_left_right_vector.first[1][0], pixel_positions_left_right_vector.first[1][1]) - right_image.at<float>(pixel_positions_left_right_vector.second[1][0], pixel_positions_left_right_vector.second[1][1])) + (1 - jj) * P2 + ii * P1;

							}
							if (pixelInsideImage(pixel_positions_left_right_vector.first[2], pixel_positions_left_right_vector.second[2], left_image_limits_mat, right_image_limits_mat) == true)
							{
								raw_differences[2] = abs(left_image.at<float>(pixel_positions_left_right_vector.first[2][0], pixel_positions_left_right_vector.first[2][1]) - right_image.at<float>(pixel_positions_left_right_vector.second[2][0], pixel_positions_left_right_vector.second[2][1])) + jj * P2 + (1 - ii) * P1;

							}
							if (pixelInsideImage(pixel_positions_left_right_vector.first[3], pixel_positions_left_right_vector.second[3], left_image_limits_mat, right_image_limits_mat) == true)
							{
								raw_differences[3] = abs(left_image.at<float>(pixel_positions_left_right_vector.first[3][0], pixel_positions_left_right_vector.first[3][1]) - right_image.at<float>(pixel_positions_left_right_vector.second[3][0], pixel_positions_left_right_vector.second[3][1])) + (1 - jj) * P2 + (1 - ii) * P1;
							}

							// Evaluate the best estimation
							int best_value_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							float best_value = estimation_vector[best_value_index][3];
							estimation_disparity.at<float>(r, c) = best_value;
							estimation_X.at<float>(r, c) = estimation_vector[best_value_index][0];
							estimation_Y.at<float>(r, c) = estimation_vector[best_value_index][1];
							estimation_Z.at<float>(r, c) = estimation_vector[best_value_index][2];
							c++;
						}
					}
				}
				else if (edg == EdgeDirection::horizontal)
				{
					// Horizontal edge
					// Penalty if estimation different from closer corner point considering the edge
					for (float ii = 0; ii <= 1; ii += step)
					{
						for (float jj = 0; jj <= 1; jj += step)
						{
							pixel_positions_left_right_vector = yesEdgesAllEstimationsPositionsVectors(TL, TR, BL, BR, ii, jj, camera_matrix_left, XYZMatrix, lookupTable, derivative_vector, estimation_vector, pixel_positions_left_right_vector);


							// calculate cost
							// check that pixel left and correspondent right are inside the image 
							// set all the values to infinity (if pixel not inside image the value will not be updated)
							std::fill(raw_differences.begin(), raw_differences.end(), std::numeric_limits<float>::infinity());

							if (pixelInsideImage(pixel_positions_left_right_vector.first[0], pixel_positions_left_right_vector.second[0], left_image_limits_mat, right_image_limits_mat) == true) {
								raw_differences[0] = abs(left_image.at<float>(pixel_positions_left_right_vector.first[0][0], pixel_positions_left_right_vector.first[0][1]) - right_image.at<float>(pixel_positions_left_right_vector.second[0][0], pixel_positions_left_right_vector.second[0][1])) + jj * P1 + ii * P2;
							}
							if (pixelInsideImage(pixel_positions_left_right_vector.first[1], pixel_positions_left_right_vector.second[1], left_image_limits_mat, right_image_limits_mat) == true)
							{
								raw_differences[1] = abs(left_image.at<float>(pixel_positions_left_right_vector.first[1][0], pixel_positions_left_right_vector.first[1][1]) - right_image.at<float>(pixel_positions_left_right_vector.second[1][0], pixel_positions_left_right_vector.second[1][1])) + (1 - jj) * P1 + ii * P2;

							}
							if (pixelInsideImage(pixel_positions_left_right_vector.first[2], pixel_positions_left_right_vector.second[2], left_image_limits_mat, right_image_limits_mat) == true)
							{
								raw_differences[2] = abs(left_image.at<float>(pixel_positions_left_right_vector.first[2][0], pixel_positions_left_right_vector.first[2][1]) - right_image.at<float>(pixel_positions_left_right_vector.second[2][0], pixel_positions_left_right_vector.second[2][1])) + jj * P1 + (1 - ii) * P2;

							}
							if (pixelInsideImage(pixel_positions_left_right_vector.first[3], pixel_positions_left_right_vector.second[3], left_image_limits_mat, right_image_limits_mat) == true)
							{
								raw_differences[3] = abs(left_image.at<float>(pixel_positions_left_right_vector.first[3][0], pixel_positions_left_right_vector.first[3][1]) - right_image.at<float>(pixel_positions_left_right_vector.second[3][0], pixel_positions_left_right_vector.second[3][1])) + (1 - jj) * P1 + (1 - ii) * P2;
							}

							// Evaluate the best estimation
							int best_value_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							float best_value = estimation_vector[best_value_index][3];
							estimation_disparity.at<float>(r, c) = best_value;
							estimation_X.at<float>(r, c) = estimation_vector[best_value_index][0];
							estimation_Y.at<float>(r, c) = estimation_vector[best_value_index][1];
							estimation_Z.at<float>(r, c) = estimation_vector[best_value_index][2];
							c++;
						}

					}
				}
				else if (edg == EdgeDirection::undefined)
				{
					// Undefined case --> edge could be any direction different from horizontal or vertical
					for (float ii = 0; ii <= 1; ii += step)
					{
						for (float jj = 0; jj <= 1; jj += step)
						{

							pixel_positions_left_right_vector = yesEdgesAllEstimationsPositionsVectors(TL, TR, BL, BR, ii, jj, camera_matrix_left, XYZMatrix, lookupTable, derivative_vector, estimation_vector, pixel_positions_left_right_vector);

							// calculate cost
							// check that pixel left and correspondent right are inside the image 
							// set all the values to infinity (if pixel not inside image the value will not be updated)
							std::fill(raw_differences.begin(), raw_differences.end(), std::numeric_limits<float>::infinity());

							if (pixelInsideImage(pixel_positions_left_right_vector.first[0], pixel_positions_left_right_vector.second[0], left_image_limits_mat, right_image_limits_mat) == true) {
								raw_differences[0] = abs(left_image.at<float>(pixel_positions_left_right_vector.first[0][0], pixel_positions_left_right_vector.first[0][1]) - right_image.at<float>(pixel_positions_left_right_vector.second[0][0], pixel_positions_left_right_vector.second[0][1])) + jj * P3 + ii * P3;
							}
							if (pixelInsideImage(pixel_positions_left_right_vector.first[1], pixel_positions_left_right_vector.second[1], left_image_limits_mat, right_image_limits_mat) == true)
							{
								raw_differences[1] = abs(left_image.at<float>(pixel_positions_left_right_vector.first[1][0], pixel_positions_left_right_vector.first[1][1]) - right_image.at<float>(pixel_positions_left_right_vector.second[1][0], pixel_positions_left_right_vector.second[1][1])) + (1 - jj) * P3 + ii * P3;

							}
							if (pixelInsideImage(pixel_positions_left_right_vector.first[2], pixel_positions_left_right_vector.second[2], left_image_limits_mat, right_image_limits_mat) == true)
							{
								raw_differences[2] = abs(left_image.at<float>(pixel_positions_left_right_vector.first[2][0], pixel_positions_left_right_vector.first[2][1]) - right_image.at<float>(pixel_positions_left_right_vector.second[2][0], pixel_positions_left_right_vector.second[2][1])) + jj * P3 + (1 - ii) * P3;

							}
							if (pixelInsideImage(pixel_positions_left_right_vector.first[3], pixel_positions_left_right_vector.second[3], left_image_limits_mat, right_image_limits_mat) == true)
							{
								raw_differences[3] = abs(left_image.at<float>(pixel_positions_left_right_vector.first[3][0], pixel_positions_left_right_vector.first[3][1]) - right_image.at<float>(pixel_positions_left_right_vector.second[3][0], pixel_positions_left_right_vector.second[3][1])) + (1 - jj) * P3 + (1 - ii) * P3;

							}
							// Evaluate the best estimation
							int best_value_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							float best_value = estimation_vector[best_value_index][3];
							estimation_disparity.at<float>(r, c) = best_value;
							estimation_X.at<float>(r, c) = estimation_vector[best_value_index][0];
							estimation_Y.at<float>(r, c) = estimation_vector[best_value_index][1];
							estimation_Z.at<float>(r, c) = estimation_vector[best_value_index][2];


							c++;
						}

					}
				}
			}
			r++;
		}
	}
}

/*
void normalSGMEstimation(const cv::Mat& lookupTable, const cv::Mat& xyd_data, const cv::Mat& XYZ_data, const cv::Mat& left_image, const cv::Mat& right_image, const int& gap, cv::Mat& disparity_matrix)
{

	EstimationParameters estimation_parameters;

	float baseline = estimation_parameters.setBaseline();
	float doffs = estimation_parameters.setDispOffset();
	float edge_threshold = estimation_parameters.setDispThreshold();
	int sampl_fact = estimation_parameters.setSamplFactor();
	float step = estimation_parameters.setStep();
	int window_size = estimation_parameters.setWindowSize();
	int hf = round(window_size / 2);
	std::pair<int, int> ranges = estimation_parameters.setRanges();

	std::vector<cv::Vec2i> pixel_positions_left_vector(4);
	std::vector<float> disparity_levels_vector(4);

	cv::Mat patch_cost_cube = cv::Mat(gap + 1, gap + 1, CV_32FC(4));

	for (int i = 0; i < lookupTable.rows - 1; i++)
	{
		for (int j = 0; j < lookupTable.cols - 1; j++)
		{
			
			// Corners of the window
			const int TL = lookupTable.at<int>(i, j);
			const int TR = lookupTable.at<int>(i, j + 1);
			const int BL = lookupTable.at<int>(i + 1, j);
			const int BR = lookupTable.at<int>(i + 1, j + 1);

			pixel_positions_left_vector = GetLeftPixelPositions(TL, TR, BL, BR, xyd_data, pixel_positions_left_vector);
			disparity_levels_vector = GetDisparityLevels(TL, TR, BL, BR, xyd_data, disparity_levels_vector);
			patch_cost_cube = getSGMCostCube(pixel_positions_left_vector, disparity_levels_vector, left_image, right_image, window_size, gap, patch_cost_cube);

			EdgeDirection edg = EdgeDirection::undefined;
			findEdge_deriv(TL, TR, BL, BR, XYZ_data, edg, edge_threshold);
			if (findEdge_deriv(TL, TR, BL, BR, XYZ_data, edg, edge_threshold) == false)
			{
				//No edges -> try to do not aggregate the cost but choose the value of disparity 
				// with the minimum cost for each pixel in the cube
				noEdgesPixelDisparity(patch_cost_cube, pixel_positions_left_vector, disparity_levels_vector, gap, disparity_matrix);
			}
			else
			{
				float P1 = 5;
				float P2 = 40;
				if (edg == EdgeDirection::vertical)
				{
					verticalEdgePixelDisparity(patch_cost_cube, pixel_positions_left_vector, disparity_levels_vector, gap, P1, P2, disparity_matrix);
				}
				else if (edg == EdgeDirection::horizontal)
				{
					horizontalEdgePixelDisparity(patch_cost_cube, pixel_positions_left_vector, disparity_levels_vector, gap, P1, P2, disparity_matrix);

				}
				else if (edg == EdgeDirection::undefined)
				{
					undefinedEdgePixelDisparity(patch_cost_cube, pixel_positions_left_vector, disparity_levels_vector, gap, P1, P2, disparity_matrix);

				}
			}
		}
		
	}

}
*/


