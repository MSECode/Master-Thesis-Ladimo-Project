#pragma once

#include <opencv2/core/mat.hpp>

#include "common_structures.h"
#include "support_structures_simulated_grid.h"

class SimulatedGridParameters
{
public:
	// Setters
	void setGap(int gap_){
		gap = gap_;
	}
	void setSamples(int samples_) {
		samples = samples_;
	}
	void setGridPoints(const int& im_width, const int& im_height) {
		width_points = (im_width - gap) / gap;
		height_points = (im_height - gap) / gap;
	}

	// Getters
	int getGap() {
		return gap;
	}
	int getSamples() {
		return samples;
	}
	int getWidthPoints() {
		return width_points;
	}
	int getHeightPoints() {
		return height_points;
	}

private:
	int gap;
	int samples;
	int width_points;
	int height_points;
};


class SimulatedGridData
{
public:
	// Setter
	void setPixelData(const int& gap, const cv::Mat& left_gt, const cv::Mat& right_gt, const int& points_on_width, const int& points_on_height);
	void setSpaceData(const double& baseline, const double& doffs, const cv::Mat& cam_0, const cv::Mat& cam_1, const int& points_on_width, const int& points_on_height);

	// Getter
	cv::Mat getPixelDataLeft();
	cv::Mat getPixelDataRight();
	cv::Mat getSpaceDataLeft();
	cv::Mat getSpaceDataRight();


private:
	cv::Mat pixel_data_left;
	cv::Mat pixel_data_right;
	cv::Mat space_data_left;
	cv::Mat space_data_right;
};

class LocalDerivativeClaculator
{
public:
	// Setter
	void setConsistencyParams(const cv::Mat& camera_matrix_left, const int& gap, const double& depth_threshold_);
	void setDerivativesVector(const cv::Mat& complete_data_matrix, const cv::Mat& lookup_matrix);

	// Getter
	std::vector<TotalDerivatives> getDerivativesVector();

private:
	std::vector<TotalDerivatives> derivatives_vector;
	cv::Mat camera_matrix_left;
	double depth_threshold = 0.0;
	double focal_length = 0.0;
	double sampling_distance = 0.0;
	// Consistency check over the derivative when evaluate them
	// Consistency left right
	void derivativeConsistencyLeftRight(ObservationData& from_left, ObservationData& from_right, ObservationData& anchor_point);
	// Consistency top bottom
	void derivativeConsistencyTopBottom(ObservationData& from_top, ObservationData& from_bot, ObservationData& anchor_point);
};


ObservationData lookupIndexToObservation(const int& i, const cv::Mat& complete_data_matrix);

cv::Point2i fromObsToPoint2i(ObservationData& point_data, const cv::Mat& camera_matrix_left);

