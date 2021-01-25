#include "support_functions_ladimo_grid_method.h"

cv::Mat getInitialGridData(const cv::Mat initial_raw_grid_data_from_csv, 
	const cv::Mat rot_dist_to_unidist_left, 
	const cv::Mat cam_mat_left, 
	const double& baseline)
{
	// Columns will be the following:
	/*
	1 - x grid position label
	2 - y grid position label
	3 - X_mt 3D position in the left stereo rectified image coordinate system
	4 - Y_mt 3D position in the left stereo rectified image coordinate system
	5 - Z_mt 3D position in the left stereo rectified image coordinate system
	6 - x_px pixel position in the left stereo rectified image coordinate system
	7 - y_px pixel position in the left stereo rectified image coordinate system
	8 - disparity value
	*/
	cv::Mat initial_grid_data_to_use(initial_raw_grid_data_from_csv.rows, 8, CV_64F, cv::Scalar::all(1));
	// Copy grid position labels
	initial_raw_grid_data_from_csv.colRange(0, 3).copyTo(initial_grid_data_to_use.colRange(0, 3));
	for (int i = 0; i < initial_raw_grid_data_from_csv.rows; i++)
	{
		initial_grid_data_to_use.at<double>(i, 7) = baseline * cam_mat_left.at<double>(0, 0) * 0.001 / 
			initial_raw_grid_data_from_csv.at<double>(i, 4);
		initial_grid_data_to_use.row(i).colRange(2, 5) = (rot_dist_to_unidist_left * initial_raw_grid_data_from_csv.row(i).colRange(2, 5).t()).t();
		initial_grid_data_to_use.at<double>(i, 5) = (initial_raw_grid_data_from_csv.at<double>(i, 2) / 
			initial_raw_grid_data_from_csv.at<double>(i, 4)
			* cam_mat_left.at<double>(0, 0)) + cam_mat_left.at<double>(0, 2);
		initial_grid_data_to_use.at<double>(i, 6) = (initial_raw_grid_data_from_csv.at<double>(i, 3) / 
			initial_raw_grid_data_from_csv.at<double>(i, 4)
			* cam_mat_left.at<double>(1, 1)) + 
			cam_mat_left.at<double>(1, 2);
	}
	return initial_grid_data_to_use;
}


void LadimoGridAccelerationStructure::setDataParameters(const cv::Mat& inputData, const double& depth_diff_threshold_, const double& fill_data_threshold_)
{
	for (size_t i = 0; i < inputData.rows; i++)
	{
		int x_pos = static_cast<int>(inputData.at<double>(i, 0));
		int y_pos = static_cast<int>(inputData.at<double>(i, 1));
		grid_positions.push_back(GridPositions::GridPositions(x_pos, y_pos));
	}
	/*
	for (auto grid_pos : gridPosVector)
	{
		std::cout << " x position: " << grid_pos.x << " | " <<
			" y position: " << grid_pos.y << std::endl;
	}
	*/
	//std::cout << gridOffsets[0] << "  " << gridOffsets[1] << std::endl;

	depth_difference_threshold = depth_diff_threshold_;
	fill_data_threshold = fill_data_threshold_;
}

void LadimoGridAccelerationStructure::setObservationData(const cv::Mat& inputData)
{
	for (size_t i = 0; i < inputData.rows; i++)
	{
		grid_observations.push_back(ObservationData::ObservationData(
			inputData.at<double>(i, 5),
			inputData.at<double>(i, 6),
			inputData.at<double>(i, 2),
			inputData.at<double>(i, 3),
			inputData.at<double>(i, 4),
			inputData.at<double>(i, 7)));
	}
}

void LadimoGridAccelerationStructure::setLaserGrid(std::vector<GridPositions> grid)
{
	laser_grid.grid = grid;
	createNeighbourIndexes();
	int counter_in = 0;
	for (int i = 0; i < grid_observations.size(); i++)
	{
		if (grid_observations[i].Z_mt <= 0.0)
		{
			counter_in++;
		}
	}
	// Check the percentage of bad data
	// Thus to have a first estimation 
	// of the percentage of errors in the initial data
	std::cout << "No data points number: " << counter_in << std::endl;
	std::cout << "Percentage of no data over the all grid points: " << double(double(counter_in) / double(grid_observations.size())) * 100.0 << std::endl;
	// Fill only the missing data that can be estimated very accurately
	fillDataGaps();
	int counter_fin = 0;
	for (int i = 0; i < grid_observations.size(); i++)
	{
		if (grid_observations[i].Z_mt <= 0.0)
		{
			counter_fin++;
		}
	}
	// Check now how much the initial errors have been reduced
	std::cout << "No data points number: " << counter_fin << std::endl;
	std::cout << "Percentage of no data over the all grid points: " << double(double(counter_fin) / double(grid_observations.size())) * 100.0 << std::endl;
	std::cout << "Reduction in missing data after estimating missing data in a precise manner: " << counter_in - counter_fin << std::endl;
	fillGridSquares();
	calculateInternalDerivatives();
	calculateExternalDerivatives();
}


void LadimoGridAccelerationStructure::createNeighbourIndexes()
{
	for (size_t i = 0; i < laser_grid.grid.size(); i++)
	{
		NeighbourIndexes ni;
		GridPositions gp_anchor = laser_grid.grid[i];
		GridPositions gp_top = gp_anchor.top();
		GridPositions gp_bot = gp_anchor.bottom();
		GridPositions gp_left = gp_anchor.left();
		GridPositions gp_right = gp_anchor.right();

		for (size_t j = 0; j < laser_grid.grid.size(); j++)
		{
			if (laser_grid.grid[j].equals(gp_top)) ni.top = j;
			if (laser_grid.grid[j].equals(gp_bot)) ni.bot = j;
			if (laser_grid.grid[j].equals(gp_left)) ni.left = j;
			if (laser_grid.grid[j].equals(gp_right)) ni.right = j;
		}
		neighbour_indeces.push_back(ni);
	}

	// Checking the neighbors
	/*
	for (auto neigh : neighbour_indeces)
	{
		std::cout << " top: " << neigh.top << " | " <<
			"bottom: " << neigh.bot << " | " <<
			"left: " << neigh.left << " | " <<
			"right: " << neigh.right << " | " << std::endl;
	}
	*/

	// resize
	internal_derivatives.resize(laser_grid.grid.size());
	external_derivatives.resize(laser_grid.grid.size());
}

void LadimoGridAccelerationStructure::fillGridSquares()
{
	for (size_t i = 0; i < laser_grid.grid.size(); i++) {
		// take point as candidate for top left position 
		// only if the observed value at that position is different from zero (the point does contain data)
		// check if top right, bottom left, bottom right exist and they do contain data
		// (if right exists, bottom exists, right neighbor of bottom exist
		if (grid_observations[i].Z_mt > 0.0)
		{
			if (neighbour_indeces[i].right != -1 && neighbour_indeces[i].bot != -1)
			{
				if (neighbour_indeces[neighbour_indeces[i].right].bot != -1)
				{
					if (grid_observations[neighbour_indeces[i].right].Z_mt > 0.0 &&
						grid_observations[neighbour_indeces[i].bot].Z_mt > 0.0 &&
						grid_observations[neighbour_indeces[neighbour_indeces[i].right].bot].Z_mt > 0.0)
					{
						GridSquare sqr;
						sqr.top_left_index = i;
						sqr.top_right_index = neighbour_indeces[i].right;
						sqr.bottom_left_index = neighbour_indeces[i].bot;
						sqr.bottom_right_index = neighbour_indeces[neighbour_indeces[i].right].bot;
						grid_squares.push_back(sqr);
					}
				}
			}
		}
	}
}

void LadimoGridAccelerationStructure::fillDataGaps()
{
	for (size_t i = 0; i < grid_observations.size(); i++)
	{
		if (grid_observations[i].Z_mt <= 0.0)
		{
			if (neighbour_indeces[i].isFull())
			{
				if (grid_observations[neighbour_indeces[i].top].Z_mt > 0.0 && grid_observations[neighbour_indeces[i].right].Z_mt > 0.0 &&
					grid_observations[neighbour_indeces[i].bot].Z_mt > 0.0 && grid_observations[neighbour_indeces[i].left].Z_mt > 0.0)
				{
					if (abs(grid_observations[neighbour_indeces[i].left].Z_mt - grid_observations[neighbour_indeces[i].top].Z_mt) <= fill_data_threshold)
					{
						grid_observations[i] = (grid_observations[neighbour_indeces[i].left] + grid_observations[neighbour_indeces[i].top]) * 0.5;
					}
					else if (abs(grid_observations[neighbour_indeces[i].right].Z_mt - grid_observations[neighbour_indeces[i].top].Z_mt) <= fill_data_threshold)
					{
						grid_observations[i] = (grid_observations[neighbour_indeces[i].right] + grid_observations[neighbour_indeces[i].top]) * 0.5;
					}
					else if (abs(grid_observations[neighbour_indeces[i].right].Z_mt - grid_observations[neighbour_indeces[i].bot].Z_mt) <= fill_data_threshold)
					{
						grid_observations[i] = (grid_observations[neighbour_indeces[i].right] + grid_observations[neighbour_indeces[i].bot]) * 0.5;
					}
					else if (abs(grid_observations[neighbour_indeces[i].left].Z_mt - grid_observations[neighbour_indeces[i].bot].Z_mt) <= fill_data_threshold)
					{
						grid_observations[i] = (grid_observations[neighbour_indeces[i].left] + grid_observations[neighbour_indeces[i].bot]) * 0.5;
					}
				}
			}
			else
			{
				if (neighbour_indeces[i].top != -1 && neighbour_indeces[i].right != -1)
				{
					if (abs(grid_observations[neighbour_indeces[i].top].Z_mt - grid_observations[neighbour_indeces[i].right].Z_mt) <= fill_data_threshold &&
						abs(grid_observations[neighbour_indeces[i].top].Z_mt - grid_observations[neighbour_indeces[i].right].Z_mt) > 0.0)
					{
						grid_observations[i] = (grid_observations[neighbour_indeces[i].top] + grid_observations[neighbour_indeces[i].right]) * 0.5;
					}
				}
				else if (neighbour_indeces[i].top != -1 && neighbour_indeces[i].left != -1)
				{
					if (abs(grid_observations[neighbour_indeces[i].top].Z_mt - grid_observations[neighbour_indeces[i].left].Z_mt) <= fill_data_threshold &&
						abs(grid_observations[neighbour_indeces[i].top].Z_mt - grid_observations[neighbour_indeces[i].left].Z_mt) > 0.0)
					{
						grid_observations[i] = (grid_observations[neighbour_indeces[i].top] + grid_observations[neighbour_indeces[i].left]) * 0.5;
					}
				}
				else if (neighbour_indeces[i].bot != -1 && neighbour_indeces[i].left != -1)
				{
					if (abs(grid_observations[neighbour_indeces[i].bot].Z_mt - grid_observations[neighbour_indeces[i].left].Z_mt) <= fill_data_threshold &&
						abs(grid_observations[neighbour_indeces[i].bot].Z_mt - grid_observations[neighbour_indeces[i].left].Z_mt) > 0.0)
					{
						grid_observations[i] = (grid_observations[neighbour_indeces[i].bot] + grid_observations[neighbour_indeces[i].left]) * 0.5;
					}
				}
				else if (neighbour_indeces[i].bot != -1 && neighbour_indeces[i].right != -1)
				{
					if (abs(grid_observations[neighbour_indeces[i].bot].Z_mt - grid_observations[neighbour_indeces[i].right].Z_mt) <= fill_data_threshold &&
						abs(grid_observations[neighbour_indeces[i].bot].Z_mt - grid_observations[neighbour_indeces[i].right].Z_mt) > 0.0)
					{
						grid_observations[i] = (grid_observations[neighbour_indeces[i].bot] + grid_observations[neighbour_indeces[i].right]) * 0.5;
					}
				}
			}
		}
	}
}

void LadimoGridAccelerationStructure::calculateInternalDerivatives()
{
	ObservationData top_left{}; ObservationData* top_left_pnt = &top_left;
	ObservationData top_right{}; ObservationData* top_right_pnt = &top_right;
	ObservationData bottom_left{}; ObservationData* bottom_left_pnt = &bottom_left;
	ObservationData bottom_right{}; ObservationData* bottom_right_pnt = &bottom_right;

	for (size_t i = 0; i < grid_squares.size(); i++)
	{
		*top_left_pnt = grid_observations[grid_squares[i].top_left_index];
		*top_right_pnt = grid_observations[grid_squares[i].top_right_index];
		*bottom_left_pnt = grid_observations[grid_squares[i].bottom_left_index];
		*bottom_right_pnt = grid_observations[grid_squares[i].bottom_right_index];

		// Top-left corner of the square
		// it needs only the right and bottom derivatives
		// the others or are not needed in the estimations or they will be recursively calculated
		// the last things means that this top-left will be the top right for another square
		// Same process for the other corners
		// All the derivatives points outward the point 
		// So when defining the estimation we don't need to put minus in from of the derivatives

		// TOP LEFT
		internal_derivatives[grid_squares[i].top_left_index].right_deriv = top_right - top_left;
		internal_derivatives[grid_squares[i].top_left_index].bottom_deriv = bottom_left - top_left;

		// TOP RIGHT
		internal_derivatives[grid_squares[i].top_right_index].left_deriv = top_left - top_right;
		internal_derivatives[grid_squares[i].top_right_index].bottom_deriv = bottom_right - top_right;

		// BOTTOM LEFT
		internal_derivatives[grid_squares[i].bottom_left_index].right_deriv = bottom_right - bottom_left;
		internal_derivatives[grid_squares[i].bottom_left_index].top_deriv = top_left - bottom_left;

		// BOTTOM RIGHT
		internal_derivatives[grid_squares[i].bottom_right_index].left_deriv = bottom_left - bottom_right;
		internal_derivatives[grid_squares[i].bottom_right_index].top_deriv = top_right - bottom_right;
	}
}

void LadimoGridAccelerationStructure::calculateExternalDerivatives()
{
	ObservationData top_left{}; ObservationData* top_left_pnt = &top_left;
	ObservationData top_right{}; ObservationData* top_right_pnt = &top_right;
	ObservationData bottom_left{}; ObservationData* bottom_left_pnt = &bottom_left;
	ObservationData bottom_right{}; ObservationData* bottom_right_pnt = &bottom_right;

	NeighbourIndexes ni_TL;
	NeighbourIndexes ni_TR;
	NeighbourIndexes ni_BL;
	NeighbourIndexes ni_BR;

	for (size_t i = 0; i < grid_squares.size(); i++)
	{
		ni_TL = neighbour_indeces[grid_squares[i].top_left_index];
		ni_TR = neighbour_indeces[grid_squares[i].top_right_index];
		ni_BL = neighbour_indeces[grid_squares[i].bottom_left_index];
		ni_BR = neighbour_indeces[grid_squares[i].bottom_right_index];

		*top_left_pnt = grid_observations[grid_squares[i].top_left_index];
		*top_right_pnt = grid_observations[grid_squares[i].top_right_index];
		*bottom_left_pnt = grid_observations[grid_squares[i].bottom_left_index];
		*bottom_right_pnt = grid_observations[grid_squares[i].bottom_right_index];


		// Case all neighbours for all square points
		if (ni_TL.isFull() && ni_TR.isFull() && ni_BL.isFull() && ni_BR.isFull())
		{
			// Case all neighbours for all square points
			// TL corner
			external_derivatives[grid_squares[i].top_left_index].top_deriv = top_left - grid_observations[ni_TL.top];
			external_derivatives[grid_squares[i].top_left_index].left_deriv = top_left - grid_observations[ni_TL.left];

			// TR corner
			external_derivatives[grid_squares[i].top_right_index].top_deriv = top_right - grid_observations[ni_TR.top];
			external_derivatives[grid_squares[i].top_right_index].right_deriv = top_right - grid_observations[ni_TR.right];

			// BL corner
			external_derivatives[grid_squares[i].bottom_left_index].bottom_deriv = bottom_left - grid_observations[ni_BL.bot];
			external_derivatives[grid_squares[i].bottom_left_index].left_deriv = bottom_left - grid_observations[ni_BL.left];

			// BR corner
			external_derivatives [grid_squares[i].bottom_right_index].bottom_deriv = bottom_right - grid_observations[ni_BR.bot];
			external_derivatives[grid_squares[i].bottom_right_index].right_deriv = bottom_right - grid_observations[ni_BR.right];
		}

		/*
		// Check case 1 by 1 in order to have all the possible external derivatives wrt the perfect squares
		// TL corner
		if (ni_TL.top != -1) externalCompleteDerivativesVector[grid_squares[i].top_left_index].top_deriv = top_left - gridObservationData[ni_TL.top];
		if (ni_TL.left != -1) externalCompleteDerivativesVector[grid_squares[i].top_left_index].left_deriv = top_left - gridObservationData[ni_TL.left];

		// TR corner
		if (ni_TR.top != -1) externalCompleteDerivativesVector[grid_squares[i].top_right_index].top_deriv = top_right - gridObservationData[ni_TR.top];
		if (ni_TR.right != -1) externalCompleteDerivativesVector[grid_squares[i].top_right_index].right_deriv = top_right - gridObservationData[ni_TR.right];

		// BL corner
		if (ni_BL.bot != -1) externalCompleteDerivativesVector[grid_squares[i].bottom_left_index].bottom_deriv = bottom_left - gridObservationData[ni_BL.bot];
		if (ni_BL.left != -1) externalCompleteDerivativesVector[grid_squares[i].bottom_left_index].left_deriv = bottom_left - gridObservationData[ni_BL.left];

		// BR corner
		if (ni_BR.bot != -1) externalCompleteDerivativesVector[grid_squares[i].bottom_right_index].bottom_deriv = bottom_right - gridObservationData[ni_BR.bot];
		if (ni_BR.right != -1) externalCompleteDerivativesVector[grid_squares[i].bottom_right_index].right_deriv = bottom_right - gridObservationData[ni_BR.right];
		*/
	}
}


