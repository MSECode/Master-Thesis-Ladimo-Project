#pragma once

#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

#include "common_structures.h"
#include "support_structures_ladimo_grid_method.h"

cv::Mat getInitialGridData(const cv::Mat initial_raw_grid_data_from_csv, 
	const cv::Mat transf_dist_to_undist_left,
	const cv::Mat camera_matrix_left,
	const double& baseline);


class LadimoGridAccelerationStructure {

public:
	void setDataParameters(const cv::Mat& inputData, const double& depth_diff_threshold_, const double& fill_data_threshold_);
	void setObservationData(const cv::Mat& inputData);
	void setLaserGrid(std::vector<GridPositions> grid);
	std::vector<GridPositions> getGridPositions() {
		return grid_positions;
	}

	std::vector<ObservationData> getObservationVector() {
		// Display the data to check them
		int space = 12;
		std::string filename = std::string(getenv("DATASET_LADIMO")) + "stereo_matching_set_2b/" + "observations.txt";
		std::ofstream output_file;
		output_file.open(filename);
		output_file << std::left << std::setw(space) << "X pos" << std::setw(space) <<
			"Y pos" << std::setw(space) <<
			"Z pos" << std::setw(space) <<
			"disp " << std::setw(space) << std::endl;

		for (auto& obs : grid_observations)
		{
			output_file << std::left << std::setw(space) << obs.X_mt << std::setw(space) <<
				obs.Y_mt << std::setw(space) <<
				obs.Z_mt << std::setw(space) <<
				obs.disp << std::setw(space) << std::endl;
		}
		output_file.close();
		return grid_observations;
	}

	std::vector<CompleteDerivatives> getInternalDerivatives() {
		int space = 30;
		std::string filename = std::string(getenv("DATASET_LADIMO")) + "stereo_matching_set_2b/" + "derivatives_internal_complete.txt";
		std::ofstream output_file;
		output_file.open(filename);
		output_file << std::left << std::setw(space) << "deriv-top: " << std::setw(space) <<
			"deriv-right: " << std::setw(space) <<
			"deriv-left: " << std::setw(space) <<
			"deriv-bottom: " << std::endl;

		for (auto& der : internal_derivatives)
		{
			output_file << " [ " << der.top_deriv.x_px << " , " << der.top_deriv.y_px << " , " << der.top_deriv.disp << " ] " << " | " <<
				" [ " << der.left_deriv.x_px << " , " << der.left_deriv.y_px << " , " << der.left_deriv.disp << " ] " << " | " <<
				" [ " << der.right_deriv.x_px << " , " << der.right_deriv.y_px << " , " << der.right_deriv.disp << " ] " << " | " <<
				" [ " << der.bottom_deriv.x_px << " , " << der.bottom_deriv.y_px << " , " << der.bottom_deriv.disp << " ] " << " | " <<
				std::endl;
		}
		output_file.close();
		return internal_derivatives;
	}
	std::vector<CompleteDerivatives> getExternalDerivatives() {
		int space = 30;
		std::string filename = std::string(getenv("DATASET_LADIMO")) + "stereo_matching_set_2b/" + "derivatives_external_complete.txt";
		std::ofstream output_file;
		output_file.open(filename);
		output_file << std::left << std::setw(space) << "deriv-top: " << std::setw(space) <<
			"deriv-right: " << std::setw(space) <<
			"deriv-left: " << std::setw(space) <<
			"deriv-bottom: " << std::endl;

		for (auto& der : external_derivatives)
		{
			output_file << " [ " << der.top_deriv.x_px << " , " << der.top_deriv.y_px << " , " << der.top_deriv.disp << " ] " << " | " <<
				" [ " << der.left_deriv.x_px << " , " << der.left_deriv.y_px << " , " << der.left_deriv.disp << " ] " << " | " <<
				" [ " << der.right_deriv.x_px << " , " << der.right_deriv.y_px << " , " << der.right_deriv.disp << " ] " << " | " <<
				" [ " << der.bottom_deriv.x_px << " , " << der.bottom_deriv.y_px << " , " << der.bottom_deriv.disp << " ] " << " | " <<
				std::endl;
		}
		output_file.close();
		return external_derivatives;
	}
	std::vector<GridSquare> getGridSquares() {
		return grid_squares;
	}

private:
	std::vector<GridPositions> grid_positions{};
	LaserGrid laser_grid{};
	std::vector<NeighbourIndexes> neighbour_indeces{};
	std::vector<ObservationData> grid_observations{};
	std::vector<CompleteDerivatives> internal_derivatives{};
	std::vector<CompleteDerivatives> external_derivatives{};
	std::vector<GridSquare> grid_squares;

	double depth_difference_threshold{ 0.15 };
	double fill_data_threshold{ 0.07 };

	// Internal Methods
	void createNeighbourIndexes();
	void fillGridSquares();
	void fillDataGaps();
	void calculateInternalDerivatives();
	void calculateExternalDerivatives();
};