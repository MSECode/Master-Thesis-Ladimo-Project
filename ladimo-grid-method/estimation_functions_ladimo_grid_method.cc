#include "estimation_functions_ladimo_grid_method.h"

bool LadimoGridEstimations::findStrongEdges(const int& TL, const int& TR, 
	const int& BL, const int& BR, 
	EdgeShape& edge_shape)
{
	EdgeThresholds edg_thresh;
	bool is_strong_edge = false;

	// CASE STRONG (REALLY VISIBLE EDGES) -- USE NOW THIS AS UNIQUE METHOD (FOR CHECK IF IMPROVEMENTS)
	if (abs(grid_observations[TL].Z_mt - grid_observations[TR].Z_mt) >= edg_thresh.light_depth_threshold) is_strong_edge = true;
	if (abs(grid_observations[BL].Z_mt - grid_observations[BR].Z_mt) >= edg_thresh.light_depth_threshold) is_strong_edge = true;
	if (abs(grid_observations[TL].Z_mt - grid_observations[BL].Z_mt) >= edg_thresh.light_depth_threshold) is_strong_edge = true;
	if (abs(grid_observations[TR].Z_mt - grid_observations[BR].Z_mt) >= edg_thresh.light_depth_threshold) is_strong_edge = true;


	if (is_strong_edge == true)
	{
		edge_shape = EdgeShape::undefined;
		if (abs(grid_observations[TL].Z_mt - grid_observations[TR].Z_mt) >= edg_thresh.light_depth_threshold &&
			abs(grid_observations[BL].Z_mt - grid_observations[BR].Z_mt) >= edg_thresh.light_depth_threshold) edge_shape = EdgeShape::pure_vertical;
		if (abs(grid_observations[TL].Z_mt - grid_observations[BL].Z_mt) >= edg_thresh.light_depth_threshold &&
			abs(grid_observations[TR].Z_mt - grid_observations[BR].Z_mt) >= edg_thresh.light_depth_threshold) edge_shape = EdgeShape::pure_horizontal;
		if (abs(grid_observations[TL].Z_mt - grid_observations[TR].Z_mt) >= edg_thresh.light_depth_threshold &&
			abs(grid_observations[TL].Z_mt - grid_observations[BL].Z_mt) >= edg_thresh.light_depth_threshold) edge_shape = EdgeShape::diagonal_top_left;
		if (abs(grid_observations[TR].Z_mt - grid_observations[TL].Z_mt) >= edg_thresh.light_depth_threshold &&
			abs(grid_observations[TR].Z_mt - grid_observations[BR].Z_mt) >= edg_thresh.light_depth_threshold) edge_shape = EdgeShape::diagonal_top_right;
		if (abs(grid_observations[BL].Z_mt - grid_observations[TL].Z_mt) >= edg_thresh.light_depth_threshold &&
			abs(grid_observations[BL].Z_mt - grid_observations[BR].Z_mt) >= edg_thresh.light_depth_threshold) edge_shape = EdgeShape::diagonal_bottom_left;
		if (abs(grid_observations[BR].Z_mt - grid_observations[TR].Z_mt) >= edg_thresh.light_depth_threshold &&
			abs(grid_observations[BR].Z_mt - grid_observations[BL].Z_mt) >= edg_thresh.light_depth_threshold) edge_shape = EdgeShape::diagonal_bottom_right;
	}

	return is_strong_edge;
}

bool LadimoGridEstimations::findSoftEdges(const int& TL, const int& TR, 
	const int& BL, const int& BR, 
	EdgeShape& edge_shape)
{
	EdgeThresholds edg_thresh;
	bool is_soft_edge = false;

	// CASE STRONG (REALLY VISIBLE EDGES) -- USE NOW THIS AS UNIQUE METHOD (FOR CHECK IF IMPROVEMENTS)
	if (abs(grid_observations[TL].Z_mt - grid_observations[TR].Z_mt) >= edg_thresh.strong_depth_threshold && 
		abs(grid_observations[TL].Z_mt - grid_observations[TR].Z_mt) < edg_thresh.light_depth_threshold) is_soft_edge = true;
	if (abs(grid_observations[BL].Z_mt - grid_observations[BR].Z_mt) >= edg_thresh.strong_depth_threshold && 
		abs(grid_observations[BL].Z_mt - grid_observations[BR].Z_mt) < edg_thresh.light_depth_threshold) is_soft_edge = true;
	if (abs(grid_observations[TL].Z_mt - grid_observations[BL].Z_mt) >= edg_thresh.strong_depth_threshold && 
		abs(grid_observations[TL].Z_mt - grid_observations[BL].Z_mt) < edg_thresh.light_depth_threshold) is_soft_edge = true;
	if (abs(grid_observations[TR].Z_mt - grid_observations[BR].Z_mt) >= edg_thresh.strong_depth_threshold && 
		abs(grid_observations[TR].Z_mt - grid_observations[BR].Z_mt) < edg_thresh.light_depth_threshold) is_soft_edge = true;


	if (is_soft_edge == true)
	{
		edge_shape = EdgeShape::undefined;
		if (abs(grid_observations[TL].Z_mt - grid_observations[TR].Z_mt) >= edg_thresh.strong_depth_threshold &&
			abs(grid_observations[BL].Z_mt - grid_observations[BR].Z_mt) >= edg_thresh.strong_depth_threshold) edge_shape = EdgeShape::pure_vertical;
		if (abs(grid_observations[TL].Z_mt - grid_observations[BL].Z_mt) >= edg_thresh.strong_depth_threshold &&
			abs(grid_observations[TR].Z_mt - grid_observations[BR].Z_mt) >= edg_thresh.strong_depth_threshold) edge_shape = EdgeShape::pure_horizontal;
		if (abs(grid_observations[TL].Z_mt - grid_observations[TR].Z_mt) >= edg_thresh.strong_depth_threshold &&
			abs(grid_observations[TL].Z_mt - grid_observations[BL].Z_mt) >= edg_thresh.strong_depth_threshold) edge_shape = EdgeShape::diagonal_top_left;
		if (abs(grid_observations[TR].Z_mt - grid_observations[TL].Z_mt) >= edg_thresh.strong_depth_threshold &&
			abs(grid_observations[TR].Z_mt - grid_observations[BR].Z_mt) >= edg_thresh.strong_depth_threshold) edge_shape = EdgeShape::diagonal_top_right;
		if (abs(grid_observations[BL].Z_mt - grid_observations[TL].Z_mt) >= edg_thresh.strong_depth_threshold &&
			abs(grid_observations[BL].Z_mt - grid_observations[BR].Z_mt) >= edg_thresh.strong_depth_threshold) edge_shape = EdgeShape::diagonal_bottom_left;
		if (abs(grid_observations[BR].Z_mt - grid_observations[TR].Z_mt) >= edg_thresh.strong_depth_threshold &&
			abs(grid_observations[BR].Z_mt - grid_observations[BL].Z_mt) >= edg_thresh.strong_depth_threshold) edge_shape = EdgeShape::diagonal_bottom_right;
	}

	return is_soft_edge;
}

void LadimoGridEstimations::bilinearInterpolation(const int& TL, const int& TR, 
	const int& BL, const int& BR, 
	cv::Vec4d& sub_square_estimation)
{
	cv::Vec2d top_interp{};
	cv::Vec2d bottom_interp{};

	double coeff_col_1 = (grid_observations[TR].X_mt - sub_square_estimation[0]) / (grid_observations[TR].X_mt - grid_observations[TL].X_mt);
	double coeff_col_2 = (sub_square_estimation[0] - grid_observations[TL].X_mt) / (grid_observations[TR].X_mt - grid_observations[TL].X_mt);
	double coeff_row_1 = (grid_observations[BL].Y_mt - sub_square_estimation[1]) / (grid_observations[BL].Y_mt - grid_observations[TL].Y_mt);
	double coeff_row_2 = (sub_square_estimation[1] - grid_observations[TL].Y_mt) / (grid_observations[BL].X_mt - grid_observations[TL].Y_mt);

	top_interp[0] = coeff_col_1 * grid_observations[TL].disp + coeff_col_2 * grid_observations[TR].disp;
	top_interp[1] = coeff_col_1 * grid_observations[TL].Z_mt + coeff_col_2 * grid_observations[TR].Z_mt;

	bottom_interp[0] = coeff_col_1 * grid_observations[BL].disp + coeff_col_2 * grid_observations[BR].disp;
	bottom_interp[1] = coeff_col_1 * grid_observations[BL].Z_mt + coeff_col_2 * grid_observations[BR].Z_mt;

	no_edge_disp = coeff_row_1 * top_interp[0] + coeff_row_2 * bottom_interp[0];
	no_edge_Z = coeff_row_1 * top_interp[1] + coeff_row_2 * bottom_interp[1];
}

void LadimoGridEstimations::noEdgesEstimations(const int& TL, const int& TR, 
	const int& BL, const int& BR, 
	const double& indx_r, const double& indx_c, 
	std::vector<ObservationData>& no_edge_est_vec)
{
	// Using Internal Derivatives
	no_edge_est_vec[0] = grid_observations[TL] + 
		internal_derivatives[TL].right_deriv * indx_c + 
		internal_derivatives[TL].bottom_deriv * indx_r;
	no_edge_est_vec[1] = grid_observations[TR] + 
		internal_derivatives[TR].left_deriv * (1.0 - indx_c) + 
		internal_derivatives[TR].bottom_deriv * indx_r;
	no_edge_est_vec[2] = grid_observations[BL] + 
		internal_derivatives[BL].right_deriv * indx_c + 
		internal_derivatives[BL].top_deriv * (1.0 - indx_r);
	no_edge_est_vec[3] = grid_observations[BR] + 
		internal_derivatives[BR].left_deriv * (1.0 - indx_c) + 
		internal_derivatives[BR].top_deriv * (1.0 - indx_r);

	final_estimastion_no_edge_internal_deriv = std::accumulate(no_edge_est_vec.begin(), no_edge_est_vec.end(), ObservationData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
	final_estimastion_no_edge_internal_deriv = final_estimastion_no_edge_internal_deriv * static_cast<double>(1.0 / no_edge_est_vec.size());
}

void LadimoGridEstimations::strongEdgesEstimations(const int& TL, const int& TR, 
	const int& BL, const int& BR, 
	const double& indx_r, const double& indx_c, 
	const cv::Mat& im_left, const cv::Mat& im_right, 
	std::vector<ObservationData>& strong_edge_est_vec)
{
	EdgeThresholds edg_thresholds;
	double esternal_der_thresh_check = edg_thresholds.strong_depth_threshold;
	// CHANGE VECTORS NAME FROM NO EDGE TO FOR EDGE 
	// CODE CORRECT BUT WRONG CORRESPONDENCES
	cv::Point2i pixel_pos_left = (0, 0);
	cv::Point2i pixel_pos_right = (0, 0);

	// Estimation to top-left
	/*
	strong_edge_est_vec[0].x_px = grid_observations[TL].x_px + internal_derivatives[TL].right_deriv.x_px * indx_c + internal_derivatives[TL].bottom_deriv.x_px * indx_r;
	strong_edge_est_vec[0].y_px = grid_observations[TL].y_px + internal_derivatives[TL].right_deriv.y_px * indx_c + internal_derivatives[TL].bottom_deriv.y_px * indx_r;
	strong_edge_est_vec[0].disp = grid_observations[TL].disp;
	strong_edge_est_vec[0].Z_mt = grid_observations[TL].Z_mt;
	*/

	if (external_derivatives[TL].top_deriv.Z_mt <= esternal_der_thresh_check && external_derivatives[TL].left_deriv.Z_mt <= esternal_der_thresh_check &&
		external_derivatives[TL].top_deriv.Z_mt > 0.0 && external_derivatives[TL].left_deriv.Z_mt > 0.0)
	{
		// if there is no edges between the corner and the external points use the esternal derivative that have to be correct (--> actually in the same plane - among threshold limits -)
		strong_edge_est_vec[0] = grid_observations[TL] + 
			external_derivatives[TL].left_deriv * indx_c +
			external_derivatives[TL].top_deriv * indx_r;
	}
	else
	{
		// Edge also from esternal direction !! // Make estimation using internal derivatives and only for x and y that are not affected by edges
		// Assigned same Z and disparity of the corresponding corner
		strong_edge_est_vec[0].x_px = grid_observations[TL].x_px + 
			internal_derivatives[TL].right_deriv.x_px * indx_c + 
			internal_derivatives[TL].bottom_deriv.x_px * indx_r;
		strong_edge_est_vec[0].y_px = grid_observations[TL].y_px + 
			internal_derivatives[TL].right_deriv.y_px * indx_c + 
			internal_derivatives[TL].bottom_deriv.y_px * indx_r;
		strong_edge_est_vec[0].X_mt = grid_observations[TL].X_mt +
			internal_derivatives[TL].right_deriv.X_mt * indx_c +
			internal_derivatives[TL].bottom_deriv.X_mt * indx_r;
		strong_edge_est_vec[0].Y_mt = grid_observations[TL].Y_mt +
			internal_derivatives[TL].right_deriv.Y_mt * indx_c +
			internal_derivatives[TL].bottom_deriv.Y_mt * indx_r;
		strong_edge_est_vec[0].disp = grid_observations[TL].disp;
		strong_edge_est_vec[0].Z_mt = grid_observations[TL].Z_mt;
	}


	// Estimation to top-right
	/*
	strong_edge_est_vec[1].x_px = grid_observations[TR].x_px + internal_derivatives[TR].left_deriv.x_px * (1 - indx_c) + internal_derivatives[TR].bottom_deriv.x_px * indx_r;
	strong_edge_est_vec[1].y_px = grid_observations[TR].y_px + internal_derivatives[TR].left_deriv.y_px * (1 - indx_c) + internal_derivatives[TR].bottom_deriv.y_px * indx_r;
	strong_edge_est_vec[1].disp = grid_observations[TR].disp;
	strong_edge_est_vec[1].Z_mt = grid_observations[TR].Z_mt;
	*/

	if (external_derivatives[TR].top_deriv.Z_mt <= esternal_der_thresh_check && external_derivatives[TR].right_deriv.Z_mt <= esternal_der_thresh_check &&
		external_derivatives[TR].top_deriv.Z_mt > 0.0 && external_derivatives[TR].right_deriv.Z_mt > 0.0)
	{
		strong_edge_est_vec[1] = grid_observations[TR] + 
			external_derivatives[TR].right_deriv * (1.0 - indx_c) + 
			external_derivatives[TR].top_deriv * indx_r;
	}
	else
	{
		strong_edge_est_vec[1].x_px = grid_observations[TR].x_px + 
			internal_derivatives[TR].left_deriv.x_px * (1.0 - indx_c) +
			internal_derivatives[TR].bottom_deriv.x_px * indx_r;
		strong_edge_est_vec[1].y_px = grid_observations[TR].y_px + 
			internal_derivatives[TR].left_deriv.y_px * (1.0 - indx_c) + 
			internal_derivatives[TR].bottom_deriv.y_px * indx_r;
		strong_edge_est_vec[1].X_mt = grid_observations[TR].X_mt +
			internal_derivatives[TR].left_deriv.X_mt * (1.0 - indx_c) +
			internal_derivatives[TR].bottom_deriv.X_mt * indx_r;
		strong_edge_est_vec[1].Y_mt = grid_observations[TR].Y_mt +
			internal_derivatives[TR].left_deriv.Y_mt * (1.0 - indx_c) +
			internal_derivatives[TR].bottom_deriv.Y_mt * indx_r;
		strong_edge_est_vec[1].disp = grid_observations[TR].disp;
		strong_edge_est_vec[1].Z_mt = grid_observations[TR].Z_mt;
	}

	// Estimation to bottom-left
	/*
	strong_edge_est_vec[2].x_px = grid_observations[BL].x_px + internal_derivatives[BL].right_deriv.x_px * indx_c + internal_derivatives[BL].top_deriv.x_px * (1 - indx_r);
	strong_edge_est_vec[2].y_px = grid_observations[BL].y_px + internal_derivatives[BL].right_deriv.y_px * indx_c + internal_derivatives[BL].top_deriv.y_px * (1 - indx_r);
	strong_edge_est_vec[2].disp = grid_observations[BL].disp;
	strong_edge_est_vec[2].Z_mt = grid_observations[BL].Z_mt;
	*/

	if (external_derivatives[BL].bottom_deriv.Z_mt <= esternal_der_thresh_check && external_derivatives[BL].left_deriv.Z_mt <= esternal_der_thresh_check &&
		external_derivatives[BL].bottom_deriv.Z_mt > 0.0 && external_derivatives[BL].left_deriv.Z_mt > 0.0)
	{
		strong_edge_est_vec[2] = grid_observations[BL] + 
			external_derivatives[BL].left_deriv * indx_c + 
			external_derivatives[BL].bottom_deriv * (1.0 - indx_r);
	}
	else
	{
		strong_edge_est_vec[2].x_px = grid_observations[BL].x_px + 
			internal_derivatives[BL].right_deriv.x_px * indx_c + 
			internal_derivatives[BL].top_deriv.x_px * (1.0 - indx_r);
		strong_edge_est_vec[2].y_px = grid_observations[BL].y_px + 
			internal_derivatives[BL].right_deriv.y_px * indx_c + 
			internal_derivatives[BL].top_deriv.y_px * (1.0 - indx_r);
		strong_edge_est_vec[2].X_mt = grid_observations[BL].X_mt +
			internal_derivatives[BL].right_deriv.X_mt * indx_c +
			internal_derivatives[BL].top_deriv.X_mt * (1.0 - indx_r);
		strong_edge_est_vec[2].Y_mt = grid_observations[BL].Y_mt +
			internal_derivatives[BL].right_deriv.Y_mt * indx_c +
			internal_derivatives[BL].top_deriv.Y_mt * (1.0 - indx_r);
		strong_edge_est_vec[2].disp = grid_observations[BL].disp;
		strong_edge_est_vec[2].Z_mt = grid_observations[BL].Z_mt;
	}

	// Estimation to bottom-right
	/*
	strong_edge_est_vec[3].x_px = grid_observations[BR].x_px + internal_derivatives[BR].left_deriv.x_px * (1 - indx_c) + internal_derivatives[BR].top_deriv.x_px * (1 - indx_r);
	strong_edge_est_vec[3].y_px = grid_observations[BR].y_px + internal_derivatives[BR].left_deriv.y_px * (1 - indx_c) + internal_derivatives[BR].top_deriv.y_px * (1 - indx_r);
	strong_edge_est_vec[3].disp = grid_observations[BR].disp;
	strong_edge_est_vec[3].Z_mt = grid_observations[BR].Z_mt;
	*/

	if (external_derivatives[BR].bottom_deriv.Z_mt <= esternal_der_thresh_check && external_derivatives[BR].right_deriv.Z_mt <= esternal_der_thresh_check &&
		external_derivatives[BR].bottom_deriv.Z_mt > 0.0 && external_derivatives[BR].right_deriv.Z_mt > 0.0)
	{
		strong_edge_est_vec[3] = grid_observations[BR] + 
			external_derivatives[BR].right_deriv * (1.0 - indx_c) + 
			external_derivatives[BR].bottom_deriv * (1.0 - indx_r);
	}
	else
	{
		strong_edge_est_vec[3].x_px = grid_observations[BR].x_px + 
			internal_derivatives[BR].left_deriv.x_px * (1.0 - indx_c) + 
			internal_derivatives[BR].top_deriv.x_px * (1.0 - indx_r);
		strong_edge_est_vec[3].y_px = grid_observations[BR].y_px + 
			internal_derivatives[BR].left_deriv.y_px * (1.0 - indx_c) + 
			internal_derivatives[BR].top_deriv.y_px * (1.0 - indx_r);
		strong_edge_est_vec[3].X_mt = grid_observations[BR].X_mt +
			internal_derivatives[BR].left_deriv.X_mt * (1.0 - indx_c) +
			internal_derivatives[BR].top_deriv.X_mt * (1.0 - indx_r);
		strong_edge_est_vec[3].Y_mt = grid_observations[BR].Y_mt +
			internal_derivatives[BR].left_deriv.Y_mt * (1.0 - indx_c) +
			internal_derivatives[BR].top_deriv.Y_mt * (1.0 - indx_r);
		strong_edge_est_vec[3].disp = grid_observations[BR].disp;
		strong_edge_est_vec[3].Z_mt = grid_observations[BR].Z_mt;

	}

	for (int i = 0; i < strong_edge_est_vec.size(); i++)
	{
		pixel_pos_left = { static_cast<int>(strong_edge_est_vec[i].x_px), static_cast<int>(strong_edge_est_vec[i].y_px) };
		pixel_pos_right = pixel_pos_left - cv::Point2i(static_cast<int>(strong_edge_est_vec[i].disp), 0);

		if (isPixelInsideImage(pixel_pos_left, pixel_pos_right, left_image_limits, right_image_limits))
		{
			cv::Vec3b temp_diff(im_left.at<cv::Vec3b>(pixel_pos_left.y, pixel_pos_left.x) - im_right.at<cv::Vec3b>(pixel_pos_right.y, pixel_pos_right.x));
			raw_differences[i] = static_cast<double>(temp_diff[0]) + static_cast<double>(temp_diff[1]) + static_cast<double>(temp_diff[2]);
		}
	}
}

void LadimoGridEstimations::softEdgesEstimations(const int& TL, const int& TR, 
	const int& BL, const int& BR, 
	const double& indx_r, const double& indx_c, 
	const cv::Mat& im_left, const cv::Mat& im_right,
	std::vector<ObservationData>& soft_edge_est_vec)
{
	EdgeThresholds edg_thresholds;
	double esternal_der_thresh_check = edg_thresholds.strong_depth_threshold;
	// CHANGE VECTORS NAME FROM NO EDGE TO FOR EDGE 
	// CODE CORRECT BUT WRONG CORRESPONDENCES
	cv::Point2i pixel_pos_left = (0, 0);
	cv::Point2i pixel_pos_right = (0, 0);

	// Estimation to top-left
	if (external_derivatives[TL].top_deriv.Z_mt <= esternal_der_thresh_check && external_derivatives[TL].left_deriv.Z_mt <= esternal_der_thresh_check &&
		external_derivatives[TL].top_deriv.Z_mt > 0.0 && external_derivatives[TL].left_deriv.Z_mt > 0.0)
	{
		// if there is no edges between the corner and the external points use the esternal derivative that have to be correct (--> actually in the same plane - among threshold limits -)
		soft_edge_est_vec[0] = grid_observations[TL] + 
			external_derivatives[TL].left_deriv * indx_c + 
			external_derivatives[TL].top_deriv * indx_r;
	}
	else
	{
		// Edge also from esternal direction !! // Make estimation using internal derivatives and only for x and y that are not affected by edges
		// Assigned same Z and disparity of the corresponding corner
		soft_edge_est_vec[0].x_px = grid_observations[TL].x_px + 
			internal_derivatives[TL].right_deriv.x_px * indx_c + 
			internal_derivatives[TL].bottom_deriv.x_px * indx_r;
		soft_edge_est_vec[0].y_px = grid_observations[TL].y_px + 
			internal_derivatives[TL].right_deriv.y_px * indx_c + 
			internal_derivatives[TL].bottom_deriv.y_px * indx_r;
		soft_edge_est_vec[0].X_mt = grid_observations[TL].X_mt +
			internal_derivatives[TL].right_deriv.X_mt * indx_c +
			internal_derivatives[TL].bottom_deriv.X_mt * indx_r;
		soft_edge_est_vec[0].Y_mt = grid_observations[TL].Y_mt +
			internal_derivatives[TL].right_deriv.Y_mt * indx_c +
			internal_derivatives[TL].bottom_deriv.Y_mt * indx_r;
		soft_edge_est_vec[0].disp = grid_observations[TL].disp;
		soft_edge_est_vec[0].Z_mt = grid_observations[TL].Z_mt;
	}

	// Estimation to top-right
	if (external_derivatives[TR].top_deriv.Z_mt <= esternal_der_thresh_check && external_derivatives[TR].right_deriv.Z_mt <= esternal_der_thresh_check &&
		external_derivatives[TR].top_deriv.Z_mt > 0.0 && external_derivatives[TR].right_deriv.Z_mt > 0.0)
	{
		soft_edge_est_vec[1] = grid_observations[TR] + 
			external_derivatives[TR].right_deriv * (1.0 - indx_c) + 
			external_derivatives[TR].top_deriv * indx_r;
	}
	else
	{
		soft_edge_est_vec[1].x_px = grid_observations[TR].x_px + 
			internal_derivatives[TR].left_deriv.x_px * (1.0 - indx_c) + 
			internal_derivatives[TR].bottom_deriv.x_px * indx_r;
		soft_edge_est_vec[1].y_px = grid_observations[TR].y_px + 
			internal_derivatives[TR].left_deriv.y_px * (1.0 - indx_c) + 
			internal_derivatives[TR].bottom_deriv.y_px * indx_r;
		soft_edge_est_vec[1].X_mt = grid_observations[TR].X_mt +
			internal_derivatives[TR].left_deriv.X_mt * (1.0 - indx_c) +
			internal_derivatives[TR].bottom_deriv.X_mt * indx_r;
		soft_edge_est_vec[1].Y_mt = grid_observations[TR].Y_mt +
			internal_derivatives[TR].left_deriv.Y_mt * (1.0 - indx_c) +
			internal_derivatives[TR].bottom_deriv.Y_mt * indx_r;
		soft_edge_est_vec[1].disp = grid_observations[TR].disp;
		soft_edge_est_vec[1].Z_mt = grid_observations[TR].Z_mt;
	}

	// Estimation to bottom-left
	if (external_derivatives[BL].bottom_deriv.Z_mt <= esternal_der_thresh_check && external_derivatives[BL].left_deriv.Z_mt <= esternal_der_thresh_check &&
		external_derivatives[BL].bottom_deriv.Z_mt > 0.0 && external_derivatives[BL].left_deriv.Z_mt > 0.0)
	{
		soft_edge_est_vec[2] = grid_observations[BL] + 
			external_derivatives[BL].left_deriv * indx_c + 
			external_derivatives[BL].bottom_deriv * (1.0 - indx_r);
	}
	else
	{
		soft_edge_est_vec[2].x_px = grid_observations[BL].x_px + 
			internal_derivatives[BL].right_deriv.x_px * indx_c + 
			internal_derivatives[BL].top_deriv.x_px * (1.0 - indx_r);
		soft_edge_est_vec[2].y_px = grid_observations[BL].y_px + 
			internal_derivatives[BL].right_deriv.y_px * indx_c + 
			internal_derivatives[BL].top_deriv.y_px * (1.0 - indx_r);
		soft_edge_est_vec[2].X_mt = grid_observations[BL].X_mt +
			internal_derivatives[BL].right_deriv.X_mt * indx_c +
			internal_derivatives[BL].top_deriv.X_mt * (1.0 - indx_r);
		soft_edge_est_vec[2].Y_mt = grid_observations[BL].Y_mt +
			internal_derivatives[BL].right_deriv.Y_mt * indx_c +
			internal_derivatives[BL].top_deriv.Y_mt * (1.0 - indx_r);
		soft_edge_est_vec[2].disp = grid_observations[BL].disp;
		soft_edge_est_vec[2].Z_mt = grid_observations[BL].Z_mt;
	}

	// Estimation to bottom-right
	if (external_derivatives[BR].bottom_deriv.Z_mt <= esternal_der_thresh_check && external_derivatives[BR].right_deriv.Z_mt <= esternal_der_thresh_check &&
		external_derivatives[BR].bottom_deriv.Z_mt > 0.0 && external_derivatives[BR].right_deriv.Z_mt > 0.0)
	{
		soft_edge_est_vec[3] = grid_observations[BR] + 
			external_derivatives[BR].right_deriv * (1.0 - indx_c) + 
			external_derivatives[BR].bottom_deriv * (1.0 - indx_r);
	}
	else
	{
		soft_edge_est_vec[3].x_px = grid_observations[BR].x_px + 
			internal_derivatives[BR].left_deriv.x_px * (1.0 - indx_c) + 
			internal_derivatives[BR].top_deriv.x_px * (1.0 - indx_r);
		soft_edge_est_vec[3].y_px = grid_observations[BR].y_px + 
			internal_derivatives[BR].left_deriv.y_px * (1.0 - indx_c) + 
			internal_derivatives[BR].top_deriv.y_px * (1.0 - indx_r);
		soft_edge_est_vec[3].X_mt = grid_observations[BR].X_mt +
			internal_derivatives[BR].left_deriv.X_mt * (1.0 - indx_c) +
			internal_derivatives[BR].top_deriv.X_mt * (1.0 - indx_r);
		soft_edge_est_vec[3].Y_mt = grid_observations[BR].Y_mt +
			internal_derivatives[BR].left_deriv.Y_mt * (1.0 - indx_c) +
			internal_derivatives[BR].top_deriv.Y_mt * (1.0 - indx_r);
		soft_edge_est_vec[3].disp = grid_observations[BR].disp;
		soft_edge_est_vec[3].Z_mt = grid_observations[BR].Z_mt;

	}

	for (size_t i = 0; i < soft_edge_est_vec.size(); i++)
	{
		pixel_pos_left = { static_cast<int>(soft_edge_est_vec[i].x_px), static_cast<int>(soft_edge_est_vec[i].y_px) };
		pixel_pos_right = pixel_pos_left - cv::Point2i(static_cast<int>(soft_edge_est_vec[i].disp), 0);

		if (isPixelInsideImage(pixel_pos_left, pixel_pos_right, left_image_limits, right_image_limits))
		{
			cv::Vec3b temp_diff(im_left.at<cv::Vec3b>(pixel_pos_left.y, pixel_pos_left.x) - im_right.at<cv::Vec3b>(pixel_pos_right.y, pixel_pos_right.x));
			raw_differences[i] = static_cast<double>(temp_diff[0]) + static_cast<double>(temp_diff[1]) + static_cast<double>(temp_diff[2]);
		}
	}
}

void LadimoGridEstimations::setEstimationParameters(const cv::Mat& camera_matrix_left_, 
	const cv::Mat& camera_matrix_right_, 
	const double& baseline_, 
	const int& sampling_factor_)
{
	camera_matrix_left = camera_matrix_left_;
	camera_matrix_right = camera_matrix_right_;
	sampling_factor = sampling_factor_;
	step = static_cast<double>(1.0 / sampling_factor_);
	baseline = baseline_;

	CostPenalties cost_penalties;
	P0 = cost_penalties.P_0;
	P1 = cost_penalties.P_1;
	P2 = cost_penalties.P_2;
	P3 = cost_penalties.P_3;
	P4 = cost_penalties.P_4;
}

void LadimoGridEstimations::setNecessaryVectors(const std::vector<GridSquare>& grid_squares_, 
	const std::vector<ObservationData>& grid_observations_, 
	const std::vector<CompleteDerivatives>& internal_derivatives_, 
	const std::vector<CompleteDerivatives>& external_derivatives_)
{
	grid_squares = grid_squares_;
	grid_observations = grid_observations_;
	internal_derivatives = internal_derivatives_;
	external_derivatives = external_derivatives_;
}

void LadimoGridEstimations::initilizeEstimationMatrices()
{
	int cols_est = (sampling_factor + 1) * (sampling_factor + 1);
	int rows_est = grid_squares.size();
	cv::Size matrix_size_est = cv::Size(cols_est, rows_est);
	estimated_Disparity = cv::Mat::zeros(matrix_size_est, CV_64F);
	estimated_X = cv::Mat::zeros(matrix_size_est, CV_64F);
	estimated_Y = cv::Mat::zeros(matrix_size_est, CV_64F);
	estimated_Z = cv::Mat::zeros(matrix_size_est, CV_64F);

	estimated_Disparity_no_edge = cv::Mat::zeros(matrix_size_est, CV_64F);
	estimated_X_no_edge = cv::Mat::zeros(matrix_size_est, CV_64F);
	estimated_Y_no_edge = cv::Mat::zeros(matrix_size_est, CV_64F);
	estimated_Z_no_edge = cv::Mat::zeros(matrix_size_est, CV_64F);

	estimated_Disparity_strong_edge = cv::Mat::zeros(matrix_size_est, CV_64F);
	estimated_X_strong_edge = cv::Mat::zeros(matrix_size_est, CV_64F);
	estimated_Y_strong_edge = cv::Mat::zeros(matrix_size_est, CV_64F);
	estimated_Z_strong_edge = cv::Mat::zeros(matrix_size_est, CV_64F);

	estimated_Disparity_soft_edge = cv::Mat::zeros(matrix_size_est, CV_64F);
	estimated_X_soft_edge = cv::Mat::zeros(matrix_size_est, CV_64F);
	estimated_Y_soft_edge = cv::Mat::zeros(matrix_size_est, CV_64F);
	estimated_Z_soft_edge = cv::Mat::zeros(matrix_size_est, CV_64F);

	int rows_guess = estimated_Disparity_no_edge.rows * estimated_Disparity_no_edge.cols;
	cv::Size matrix_size_guess = cv::Size(1, rows_guess);
	guessed_Disparity = cv::Mat::zeros(matrix_size_guess, CV_64F);
	guessed_X = cv::Mat::zeros(matrix_size_guess, CV_64F);
	guessed_Y = cv::Mat::zeros(matrix_size_guess, CV_64F);
	guessed_Z = cv::Mat::zeros(matrix_size_guess, CV_64F);

	guessed_Disparity_no_edge = cv::Mat::zeros(matrix_size_guess, CV_64F);
	guessed_X_no_edge = cv::Mat::zeros(matrix_size_guess, CV_64F);
	guessed_Y_no_edge = cv::Mat::zeros(matrix_size_guess, CV_64F);
	guessed_Z_no_edge = cv::Mat::zeros(matrix_size_guess, CV_64F);

	guessed_Disparity_strong_edge = cv::Mat::zeros(matrix_size_guess, CV_64F);
	guessed_X_strong_edge = cv::Mat::zeros(matrix_size_guess, CV_64F);
	guessed_Y_strong_edge = cv::Mat::zeros(matrix_size_guess, CV_64F);
	guessed_Z_strong_edge = cv::Mat::zeros(matrix_size_guess, CV_64F);

	guessed_Disparity_soft_edge = cv::Mat::zeros(matrix_size_guess, CV_64F);
	guessed_X_soft_edge = cv::Mat::zeros(matrix_size_guess, CV_64F);
	guessed_Y_soft_edge = cv::Mat::zeros(matrix_size_guess, CV_64F);
	guessed_Z_soft_edge = cv::Mat::zeros(matrix_size_guess, CV_64F);
}

void LadimoGridEstimations::setEstimations(const cv::Mat& image_left, const cv::Mat& image_right)
{

	left_image_limits = defineMatrixLimits(0, image_left.rows, 0, image_left.cols).t();
	right_image_limits = defineMatrixLimits(0, image_right.rows, 0, image_right.cols).t();

	double no_edges_step = 0.0;
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

	for (int i = 0; i < grid_squares.size(); i++)
	{
		TL = grid_squares[i].top_left_index;
		TR = grid_squares[i].top_right_index;
		BL = grid_squares[i].bottom_left_index;
		BR = grid_squares[i].bottom_right_index;

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
				for (double ii = 0.0; ii <= 1.0; ii += step)
				{
					for (double jj = 0.0; jj <= 1.0; jj += step)
					{
						// Fast bilinear interpolation using the corner values
						// Is it better to have the mean value among the corners or do only one estimation using only one corner ?
						noEdgesEstimations(TL, TR, BL, BR, ii, jj, no_edge_estimations);
						estimated_Disparity.at<double>(i, d) = final_estimastion_no_edge_internal_deriv.disp;
						estimated_Z.at<double>(i, d) = final_estimastion_no_edge_internal_deriv.Z_mt;
						estimated_X.at<double>(i, d) = final_estimastion_no_edge_internal_deriv.X_mt;
						estimated_Y.at<double>(i, d) = final_estimastion_no_edge_internal_deriv.Y_mt;

						estimated_Disparity_no_edge.at<double>(i, d) = final_estimastion_no_edge_internal_deriv.disp;
						estimated_Z_no_edge.at<double>(i, d) = final_estimastion_no_edge_internal_deriv.Z_mt;
						estimated_X_no_edge.at<double>(i, d) = final_estimastion_no_edge_internal_deriv.X_mt;
						estimated_Y_no_edge.at<double>(i, d) = final_estimastion_no_edge_internal_deriv.Y_mt;

						d++;
					}
				}
			}
			else
			{
				// Strong edge case
				std::fill(raw_differences.begin(), raw_differences.end(), DBL_MAX);
				if (edg == EdgeShape::pure_vertical)
				{
					for (double ii = 0.0; ii <= 1.0; ii += step)
					{
						for (double jj = 0.0; jj <= 1.0; jj += step)
						{
							strongEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, strong_edge_estimations);
							// Is there the need to add penalties? Maybe yes in order to be really sure about the results
							// Then with the penalty it is possible to enhance the different edge shape cases
							raw_differences[0] = raw_differences[0] + P1 * ii + P2 * jj;
							raw_differences[1] = raw_differences[1] + P1 * ii + P2 * (1.0 - jj);
							raw_differences[2] = raw_differences[2] + P1 * (1.0 - ii) + P2 * jj;
							raw_differences[3] = raw_differences[3] + P1 * (1.0 - ii) + P2 * (1.0 - jj);

							best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							estimated_X.at<double>(i, d) = strong_edge_estimations[best_index].X_mt;
							estimated_Y.at<double>(i, d) = strong_edge_estimations[best_index].Y_mt;
							estimated_Z.at<double>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity.at<double>(i, d) = strong_edge_estimations[best_index].disp;

							estimated_X_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].X_mt;
							estimated_Y_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].Y_mt;
							estimated_Z_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].disp;

							d++;
						}
					}

				}
				else if (edg == EdgeShape::pure_horizontal)
				{
					for (double ii = 0.0; ii <= 1.0; ii += step)
					{
						for (double jj = 0.0; jj <= 1.0; jj += step)
						{
							strongEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, strong_edge_estimations);
							// Is there the need to add penalties? Maybe yes in order to be really sure about the results
							// Then with the penalty it is possible to enhance the different edge shape cases
							raw_differences[0] = raw_differences[0] + P2 * ii + P1 * jj;
							raw_differences[1] = raw_differences[1] + P2 * ii + P1 * (1.0 - jj);
							raw_differences[2] = raw_differences[2] + P2 * (1.0 - ii) + P1 * jj;
							raw_differences[3] = raw_differences[3] + P2 * (1.0 - ii) + P1 * (1.0 - jj);

							best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							estimated_X.at<double>(i, d) = strong_edge_estimations[best_index].X_mt;
							estimated_Y.at<double>(i, d) = strong_edge_estimations[best_index].Y_mt;
							estimated_Z.at<double>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity.at<double>(i, d) = strong_edge_estimations[best_index].disp;

							estimated_X_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].X_mt;
							estimated_Y_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].Y_mt;
							estimated_Z_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].disp;

							d++;
						}
					}
				}
				else if (edg == EdgeShape::diagonal_top_left)
				{
					for (double ii = 0.0; ii <= 1.0; ii += step)
					{
						for (double jj = 0.0; jj <= 1.0; jj += step)
						{
							strongEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, strong_edge_estimations);
							// Is there the need to add penalties? Maybe yes in order to be really sure about the results
							// Then with the penalty it is possible to enhance the different edge shape cases
							// The few bigger penalty should go on the different edge or on the other three edges ?
							raw_differences[0] = raw_differences[0] + P4 * ii + P4 * jj;
							raw_differences[1] = raw_differences[1] + P3 * ii + P3 * (1.0 - jj);
							raw_differences[2] = raw_differences[2] + P3 * (1.0 - ii) + P3 * jj;
							raw_differences[3] = raw_differences[3] + P3 * (1.0 - ii) + P3 * (1.0 - jj);

							best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							estimated_X.at<double>(i, d) = strong_edge_estimations[best_index].X_mt;
							estimated_Y.at<double>(i, d) = strong_edge_estimations[best_index].Y_mt;
							estimated_Z.at<double>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity.at<double>(i, d) = strong_edge_estimations[best_index].disp;

							estimated_X_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].X_mt;
							estimated_Y_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].Y_mt;
							estimated_Z_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].disp;

							d++;
						}
					}
				}
				else if (edg == EdgeShape::diagonal_top_right)
				{
					for (double ii = 0.0; ii <= 1.0; ii += step)
					{
						for (double jj = 0.0; jj <= 1.0; jj += step)
						{
							strongEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, strong_edge_estimations);
							// Is there the need to add penalties? Maybe yes in order to be really sure about the results
							// Then with the penalty it is possible to enhance the different edge shape cases
							raw_differences[0] = raw_differences[0] + P3 * ii + P3 * jj;
							raw_differences[1] = raw_differences[1] + P4 * ii + P4 * (1.0 - jj);
							raw_differences[2] = raw_differences[2] + P3 * (1.0 - ii) + P3 * jj;
							raw_differences[3] = raw_differences[3] + P3 * (1.0 - ii) + P3 * (1.0 - jj);

							best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							estimated_X.at<double>(i, d) = strong_edge_estimations[best_index].X_mt;
							estimated_Y.at<double>(i, d) = strong_edge_estimations[best_index].Y_mt;
							estimated_Z.at<double>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity.at<double>(i, d) = strong_edge_estimations[best_index].disp;

							estimated_X_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].X_mt;
							estimated_Y_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].Y_mt;
							estimated_Z_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].disp;

							d++;
						}
					}
				}
				else if (edg == EdgeShape::diagonal_bottom_left)
				{
					for (double ii = 0.0; ii <= 1.0; ii += step)
					{
						for (double jj = 0.0; jj <= 1.0; jj += step)
						{
							strongEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, strong_edge_estimations);
							// Is there the need to add penalties? Maybe yes in order to be really sure about the results
							// Then with the penalty it is possible to enhance the different edge shape cases
							raw_differences[0] = raw_differences[0] + P3 * ii + P3 * jj;
							raw_differences[1] = raw_differences[1] + P3 * ii + P3 * (1.0 - jj);
							raw_differences[2] = raw_differences[2] + P4 * (1.0 - ii) + P4 * jj;
							raw_differences[3] = raw_differences[3] + P3 * (1.0 - ii) + P3 * (1.0 - jj);

							best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							estimated_X.at<double>(i, d) = strong_edge_estimations[best_index].X_mt;
							estimated_Y.at<double>(i, d) = strong_edge_estimations[best_index].Y_mt;
							estimated_Z.at<double>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity.at<double>(i, d) = strong_edge_estimations[best_index].disp;

							estimated_X_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].X_mt;
							estimated_Y_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].Y_mt;
							estimated_Z_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].disp;

							d++;
						}
					}
				}
				else if (edg == EdgeShape::diagonal_bottom_right)
				{
					for (double ii = 0.0; ii <= 1.0; ii += step)
					{
						for (double jj = 0.0; jj <= 1.0; jj += step)
						{
							strongEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, strong_edge_estimations);
							// Is there the need to add penalties? Maybe yes in order to be really sure about the results
							// Then with the penalty it is possible to enhance the different edge shape cases
							raw_differences[0] = raw_differences[0] + P3 * ii + P3 * jj;
							raw_differences[1] = raw_differences[1] + P3 * ii + P3 * (1.0 - jj);
							raw_differences[2] = raw_differences[2] + P3 * (1.0 - ii) + P3 * jj;
							raw_differences[3] = raw_differences[3] + P4 * (1.0 - ii) + P4 * (1.0 - jj);

							best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							estimated_X.at<double>(i, d) = strong_edge_estimations[best_index].X_mt;
							estimated_Y.at<double>(i, d) = strong_edge_estimations[best_index].Y_mt;
							estimated_Z.at<double>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity.at<double>(i, d) = strong_edge_estimations[best_index].disp;

							estimated_X_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].X_mt;
							estimated_Y_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].Y_mt;
							estimated_Z_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].disp;

							d++;
						}
					}
				}
				else
				{
					// Unknown type of edge --> maybe use a big penalty for all the estimations 
					for (double ii = 0.0; ii <= 1.0; ii += step)
					{
						for (double jj = 0.0; jj <= 1.0; jj += step)
						{
							strongEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, strong_edge_estimations);
							// Is there the need to add penalties? Maybe yes in order to be really sure about the results
							// Then with the penalty it is possible to enhance the different edge shape cases
							raw_differences[0] = raw_differences[0] + P0 * ii + P0 * jj;
							raw_differences[1] = raw_differences[1] + P0 * ii + P0 * (1.0 - jj);
							raw_differences[2] = raw_differences[2] + P0 * (1.0 - ii) + P0 * jj;
							raw_differences[3] = raw_differences[3] + P0 * (1.0 - ii) + P0 * (1.0 - jj);

							best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
							estimated_X.at<double>(i, d) = strong_edge_estimations[best_index].X_mt;
							estimated_Y.at<double>(i, d) = strong_edge_estimations[best_index].Y_mt;
							estimated_Z.at<double>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity.at<double>(i, d) = strong_edge_estimations[best_index].disp;

							estimated_X_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].X_mt;
							estimated_Y_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].Y_mt;
							estimated_Z_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].Z_mt;
							estimated_Disparity_strong_edge.at<double>(i, d) = strong_edge_estimations[best_index].disp;

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
				for (double ii = 0.0; ii <= 1.0; ii += step)
				{
					for (double jj = 0.0; jj <= 1.0; jj += step)
					{
						softEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, soft_edge_estimations);
						// Is there the need to add penalties? Maybe yes in order to be really sure about the results
						// Then with the penalty it is possible to enhance the different edge shape cases
						raw_differences[0] = raw_differences[0] + P1 * ii + P2 * jj;
						raw_differences[1] = raw_differences[1] + P1 * ii + P2 * (1.0 - jj);
						raw_differences[2] = raw_differences[2] + P1 * (1.0 - ii) + P2 * jj;
						raw_differences[3] = raw_differences[3] + P1 * (1.0 - ii) + P2 * (1.0 - jj);

						best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
						estimated_X.at<double>(i, d) = soft_edge_estimations[best_index].X_mt;
						estimated_Y.at<double>(i, d) = soft_edge_estimations[best_index].Y_mt;
						estimated_Z.at<double>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity.at<double>(i, d) = soft_edge_estimations[best_index].disp;

						estimated_X_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].X_mt;
						estimated_Y_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].Y_mt;
						estimated_Z_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].disp;

						d++;
					}
				}

			}
			else if (edg == EdgeShape::pure_horizontal)
			{
				for (double ii = 0.0; ii <= 1.0; ii += step)
				{
					for (double jj = 0.0; jj <= 1.0; jj += step)
					{
						softEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, soft_edge_estimations);
						// Is there the need to add penalties? Maybe yes in order to be really sure about the results
						// Then with the penalty it is possible to enhance the different edge shape cases
						raw_differences[0] = raw_differences[0] + P2 * ii + P1 * jj;
						raw_differences[1] = raw_differences[1] + P2 * ii + P1 * (1.0 - jj);
						raw_differences[2] = raw_differences[2] + P2 * (1.0 - ii) + P1 * jj;
						raw_differences[3] = raw_differences[3] + P2 * (1.0 - ii) + P1 * (1.0 - jj);

						best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
						estimated_X.at<double>(i, d) = soft_edge_estimations[best_index].X_mt;
						estimated_Y.at<double>(i, d) = soft_edge_estimations[best_index].Y_mt;
						estimated_Z.at<double>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity.at<double>(i, d) = soft_edge_estimations[best_index].disp;

						estimated_X_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].X_mt;
						estimated_Y_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].Y_mt;
						estimated_Z_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].disp;

						d++;
					}
				}
			}
			else if (edg == EdgeShape::diagonal_top_left)
			{
				for (double ii = 0.0; ii <= 1.0; ii += step)
				{
					for (double jj = 0.0; jj <= 1.0; jj += step)
					{
						softEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, soft_edge_estimations);
						// Is there the need to add penalties? Maybe yes in order to be really sure about the results
						// Then with the penalty it is possible to enhance the different edge shape cases
						raw_differences[0] = raw_differences[0] + P4 * ii + P4 * jj;
						raw_differences[1] = raw_differences[1] + P3 * ii + P3 * (1.0 - jj);
						raw_differences[2] = raw_differences[2] + P3 * (1.0 - ii) + P3 * jj;
						raw_differences[3] = raw_differences[3] + P3 * (1.0 - ii) + P3 * (1.0 - jj);

						best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
						estimated_X.at<double>(i, d) = soft_edge_estimations[best_index].X_mt;
						estimated_Y.at<double>(i, d) = soft_edge_estimations[best_index].Y_mt;
						estimated_Z.at<double>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity.at<double>(i, d) = soft_edge_estimations[best_index].disp;

						estimated_X_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].X_mt;
						estimated_Y_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].Y_mt;
						estimated_Z_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].disp;

						d++;
					}
				}
			}
			else if (edg == EdgeShape::diagonal_top_right)
			{
				for (double ii = 0.0; ii <= 1.0; ii += step)
				{
					for (double jj = 0.0; jj <= 1.0; jj += step)
					{
						softEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, soft_edge_estimations);
						// Is there the need to add penalties? Maybe yes in order to be really sure about the results
						// Then with the penalty it is possible to enhance the different edge shape cases
						raw_differences[0] = raw_differences[0] + P3 * ii + P3 * jj;
						raw_differences[1] = raw_differences[1] + P4 * ii + P4 * (1.0 - jj);
						raw_differences[2] = raw_differences[2] + P3 * (1.0 - ii) + P3 * jj;
						raw_differences[3] = raw_differences[3] + P3 * (1.0 - ii) + P3 * (1.0 - jj);

						best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
						estimated_X.at<double>(i, d) = soft_edge_estimations[best_index].X_mt;
						estimated_Y.at<double>(i, d) = soft_edge_estimations[best_index].Y_mt;
						estimated_Z.at<double>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity.at<double>(i, d) = soft_edge_estimations[best_index].disp;

						estimated_X_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].X_mt;
						estimated_Y_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].Y_mt;
						estimated_Z_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].disp;

						d++;
					}
				}
			}
			else if (edg == EdgeShape::diagonal_bottom_left)
			{
				for (double ii = 0.0; ii <= 1.0; ii += step)
				{
					for (double jj = 0.0; jj <= 1.0; jj += step)
					{
						softEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, soft_edge_estimations);
						// Is there the need to add penalties? Maybe yes in order to be really sure about the results
						// Then with the penalty it is possible to enhance the different edge shape cases
						raw_differences[0] = raw_differences[0] + P3 * ii + P3 * jj;
						raw_differences[1] = raw_differences[1] + P3 * ii + P3 * (1.0 - jj);
						raw_differences[2] = raw_differences[2] + P4 * (1.0 - ii) + P4 * jj;
						raw_differences[3] = raw_differences[3] + P3 * (1.0 - ii) + P3 * (1.0 - jj);

						best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
						estimated_X.at<double>(i, d) = soft_edge_estimations[best_index].X_mt;
						estimated_Y.at<double>(i, d) = soft_edge_estimations[best_index].Y_mt;
						estimated_Z.at<double>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity.at<double>(i, d) = soft_edge_estimations[best_index].disp;

						estimated_X_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].X_mt;
						estimated_Y_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].Y_mt;
						estimated_Z_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].disp;

						d++;
					}
				}
			}
			else if (edg == EdgeShape::diagonal_bottom_right)
			{
				for (double ii = 0.0; ii <= 1.0; ii += step)
				{
					for (double jj = 0.0; jj <= 1.0; jj += step)
					{
						softEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, soft_edge_estimations);
						// Is there the need to add penalties? Maybe yes in order to be really sure about the results
						// Then with the penalty it is possible to enhance the different edge shape cases
						raw_differences[0] = raw_differences[0] + P3 * ii + P3 * jj;
						raw_differences[1] = raw_differences[1] + P3 * ii + P3 * (1.0 - jj);
						raw_differences[2] = raw_differences[2] + P3 * (1.0 - ii) + P3 * jj;
						raw_differences[3] = raw_differences[3] + P4 * (1.0 - ii) + P4 * (1.0 - jj);

						best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
						estimated_X.at<double>(i, d) = soft_edge_estimations[best_index].X_mt;
						estimated_Y.at<double>(i, d) = soft_edge_estimations[best_index].Y_mt;
						estimated_Z.at<double>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity.at<double>(i, d) = soft_edge_estimations[best_index].disp;

						estimated_X_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].X_mt;
						estimated_Y_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].Y_mt;
						estimated_Z_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].disp;

						d++;
					}
				}
			}
			else
			{
				// Undefined edge case
				for (double ii = 0.0; ii <= 1.0; ii += step)
				{
					for (double jj = 0.0; jj <= 1.0; jj += step)
					{
						softEdgesEstimations(TL, TR, BL, BR, ii, jj, image_left, image_right, soft_edge_estimations);
						// Is there the need to add penalties? Maybe yes in order to be really sure about the results
						// Then with the penalty it is possible to enhance the different edge shape cases
						raw_differences[0] = raw_differences[0] + P0 * ii + P0 * jj;
						raw_differences[1] = raw_differences[1] + P0 * ii + P0 * (1.0 - jj);
						raw_differences[2] = raw_differences[2] + P0 * (1.0 - ii) + P0 * jj;
						raw_differences[3] = raw_differences[3] + P0 * (1.0 - ii) + P0 * (1.0 - jj);

						best_index = std::min_element(raw_differences.begin(), raw_differences.end()) - raw_differences.begin();
						estimated_X.at<double>(i, d) = soft_edge_estimations[best_index].X_mt;
						estimated_Y.at<double>(i, d) = soft_edge_estimations[best_index].Y_mt;
						estimated_Z.at<double>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity.at<double>(i, d) = soft_edge_estimations[best_index].disp;

						estimated_X_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].X_mt;
						estimated_Y_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].Y_mt;
						estimated_Z_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].Z_mt;
						estimated_Disparity_soft_edge.at<double>(i, d) = soft_edge_estimations[best_index].disp;

						d++;
					}
				}
			}
		}
	}
}

void LadimoGridEstimations::setGuesses()
{
	int TL_indx{ 0 };
	int TR_indx{ 0 };
	int BL_indx{ 0 };
	int BR_indx{ 0 };

	int r = 0;

	for (int i = 0; i < estimated_Disparity.rows; i++)
	{
		for (int j = 0; j < estimated_Disparity.cols; j++)
		{
			guessed_Disparity.at<double>(r, 0) = estimated_Disparity.at<double>(i, j);
			guessed_X.at<double>(r, 0) = estimated_X.at<double>(i, j);
			guessed_Y.at<double>(r, 0) = estimated_Y.at<double>(i, j);
			guessed_Z.at<double>(r, 0) = estimated_Z.at<double>(i, j);

			guessed_Disparity_no_edge.at<double>(r, 0) = estimated_Disparity_no_edge.at<double>(i, j);
			guessed_X_no_edge.at<double>(r, 0) = estimated_X_no_edge.at<double>(i, j);
			guessed_Y_no_edge.at<double>(r, 0) = estimated_Y_no_edge.at<double>(i, j);
			guessed_Z_no_edge.at<double>(r, 0) = estimated_Z_no_edge.at<double>(i, j);

			guessed_Disparity_strong_edge.at<double>(r, 0) = estimated_Disparity_strong_edge.at<double>(i, j);
			guessed_X_strong_edge.at<double>(r, 0) = estimated_X_strong_edge.at<double>(i, j);
			guessed_Y_strong_edge.at<double>(r, 0) = estimated_Y_strong_edge.at<double>(i, j);
			guessed_Z_strong_edge.at<double>(r, 0) = estimated_Z_strong_edge.at<double>(i, j);

			guessed_Disparity_soft_edge.at<double>(r, 0) = estimated_Disparity_soft_edge.at<double>(i, j);
			guessed_X_soft_edge.at<double>(r, 0) = estimated_X_soft_edge.at<double>(i, j);
			guessed_Y_soft_edge.at<double>(r, 0) = estimated_Y_soft_edge.at<double>(i, j);
			guessed_Z_soft_edge.at<double>(r, 0) = estimated_Z_soft_edge.at<double>(i, j);

			if (j == 0)
			{
				TL_indx = grid_squares[i].top_left_index;
				guessed_Disparity.at<double>(r, 0) = grid_observations[TL_indx].disp;
				guessed_X.at<double>(r, 0) = grid_observations[TL_indx].X_mt;
				guessed_Y.at<double>(r, 0) = grid_observations[TL_indx].Y_mt;
				guessed_Z.at<double>(r, 0) = grid_observations[TL_indx].Z_mt;

				guessed_Disparity_no_edge.at<double>(r, 0) = grid_observations[TL_indx].disp;
				guessed_X_no_edge.at<double>(r, 0) = grid_observations[TL_indx].X_mt;
				guessed_Y_no_edge.at<double>(r, 0) = grid_observations[TL_indx].Y_mt;
				guessed_Z_no_edge.at<double>(r, 0) = grid_observations[TL_indx].Z_mt;

				guessed_Disparity_strong_edge.at<double>(r, 0) = grid_observations[TL_indx].disp;
				guessed_X_strong_edge.at<double>(r, 0) = grid_observations[TL_indx].X_mt;
				guessed_Y_strong_edge.at<double>(r, 0) = grid_observations[TL_indx].Y_mt;
				guessed_Z_strong_edge.at<double>(r, 0) = grid_observations[TL_indx].Z_mt;

				guessed_Disparity_soft_edge.at<double>(r, 0) = grid_observations[TL_indx].disp;
				guessed_X_soft_edge.at<double>(r, 0) = grid_observations[TL_indx].X_mt;
				guessed_Y_soft_edge.at<double>(r, 0) = grid_observations[TL_indx].Y_mt;
				guessed_Z_soft_edge.at<double>(r, 0) = grid_observations[TL_indx].Z_mt;
			}
			else if (j == sampling_factor)
			{
				TR_indx = grid_squares[i].top_right_index;
				guessed_Disparity.at<double>(r, 0) = grid_observations[TR_indx].disp;
				guessed_X.at<double>(r, 0) = grid_observations[TR_indx].X_mt;
				guessed_Y.at<double>(r, 0) = grid_observations[TR_indx].Y_mt;
				guessed_Z.at<double>(r, 0) = grid_observations[TR_indx].Z_mt;

				guessed_Disparity_no_edge.at<double>(r, 0) = grid_observations[TR_indx].disp;
				guessed_X_no_edge.at<double>(r, 0) = grid_observations[TR_indx].X_mt;
				guessed_Y_no_edge.at<double>(r, 0) = grid_observations[TR_indx].Y_mt;
				guessed_Z_no_edge.at<double>(r, 0) = grid_observations[TR_indx].Z_mt;

				guessed_Disparity_strong_edge.at<double>(r, 0) = grid_observations[TR_indx].disp;
				guessed_X_strong_edge.at<double>(r, 0) = grid_observations[TR_indx].X_mt;
				guessed_Y_strong_edge.at<double>(r, 0) = grid_observations[TR_indx].Y_mt;
				guessed_Z_strong_edge.at<double>(r, 0) = grid_observations[TR_indx].Z_mt;

				guessed_Disparity_soft_edge.at<double>(r, 0) = grid_observations[TR_indx].disp;
				guessed_X_soft_edge.at<double>(r, 0) = grid_observations[TR_indx].X_mt;
				guessed_Y_soft_edge.at<double>(r, 0) = grid_observations[TR_indx].Y_mt;
				guessed_Z_soft_edge.at<double>(r, 0) = grid_observations[TR_indx].Z_mt;
			}
			else if (j == estimated_Disparity.cols - 1 - sampling_factor)
			{
				BL_indx = grid_squares[i].bottom_left_index;
				guessed_Disparity.at<double>(r, 0) = grid_observations[BL_indx].disp;
				guessed_X.at<double>(r, 0) = grid_observations[BL_indx].X_mt;
				guessed_Y.at<double>(r, 0) = grid_observations[BL_indx].Y_mt;
				guessed_Z.at<double>(r, 0) = grid_observations[BL_indx].Z_mt;

				guessed_Disparity_no_edge.at<double>(r, 0) = grid_observations[BL_indx].disp;
				guessed_X_no_edge.at<double>(r, 0) = grid_observations[BL_indx].X_mt;
				guessed_Y_no_edge.at<double>(r, 0) = grid_observations[BL_indx].Y_mt;
				guessed_Z_no_edge.at<double>(r, 0) = grid_observations[BL_indx].Z_mt;

				guessed_Disparity_strong_edge.at<double>(r, 0) = grid_observations[BL_indx].disp;
				guessed_X_strong_edge.at<double>(r, 0) = grid_observations[BL_indx].X_mt;
				guessed_Y_strong_edge.at<double>(r, 0) = grid_observations[BL_indx].Y_mt;
				guessed_Z_strong_edge.at<double>(r, 0) = grid_observations[BL_indx].Z_mt;

				guessed_Disparity_soft_edge.at<double>(r, 0) = grid_observations[BL_indx].disp;
				guessed_X_soft_edge.at<double>(r, 0) = grid_observations[BL_indx].X_mt;
				guessed_Y_soft_edge.at<double>(r, 0) = grid_observations[BL_indx].Y_mt;
				guessed_Z_soft_edge.at<double>(r, 0) = grid_observations[BL_indx].Z_mt;
			}
			else if (j == estimated_Disparity.cols - 1)
			{
				BR_indx = grid_squares[i].bottom_right_index;
				guessed_Disparity.at<double>(r, 0) = grid_observations[BR_indx].disp;
				guessed_X.at<double>(r, 0) = grid_observations[BR_indx].X_mt;
				guessed_Y.at<double>(r, 0) = grid_observations[BR_indx].Y_mt;
				guessed_Z.at<double>(r, 0) = grid_observations[BR_indx].Z_mt;

				guessed_Disparity_no_edge.at<double>(r, 0) = grid_observations[BR_indx].disp;
				guessed_X_no_edge.at<double>(r, 0) = grid_observations[BR_indx].X_mt;
				guessed_Y_no_edge.at<double>(r, 0) = grid_observations[BR_indx].Y_mt;
				guessed_Z_no_edge.at<double>(r, 0) = grid_observations[BR_indx].Z_mt;

				guessed_Disparity_strong_edge.at<double>(r, 0) = grid_observations[BR_indx].disp;
				guessed_X_strong_edge.at<double>(r, 0) = grid_observations[BR_indx].X_mt;
				guessed_Y_strong_edge.at<double>(r, 0) = grid_observations[BR_indx].Y_mt;
				guessed_Z_strong_edge.at<double>(r, 0) = grid_observations[BR_indx].Z_mt;

				guessed_Disparity_soft_edge.at<double>(r, 0) = grid_observations[BR_indx].disp;
				guessed_X_soft_edge.at<double>(r, 0) = grid_observations[BR_indx].X_mt;
				guessed_Y_soft_edge.at<double>(r, 0) = grid_observations[BR_indx].Y_mt;
				guessed_Z_soft_edge.at<double>(r, 0) = grid_observations[BR_indx].Z_mt;
			}
			r++;
		}
	}
}
