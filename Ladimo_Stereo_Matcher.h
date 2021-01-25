#pragma once
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

struct GridPos{
	GridPos() {};
	GridPos(int x_, int y_) {
		x = x_;
		y = y_;
	};
	int x = 0;
	int y = 0;
	bool equals(const GridPos& gp) {
		return (x == gp.x && y == gp.y);
	}
	GridPos left() {
		return GridPos(x - 1, y);
	}
	GridPos right() {
		return GridPos(x + 1, y);
	}
	GridPos top() {
		return GridPos(x, y - 1);
	}
	GridPos bottom() {
		return GridPos(x, y + 1);
	}
};

struct LaserGrid {
	std::vector<GridPos> grid;
};

struct NeighborIndices {
	int top{ -1 };
	int left{ -1 };
	int bottom{ -1 };
	int right{ -1 };
};

struct Derivative {
	cv::Point3f from_right;
	cv::Point3f from_left;
	cv::Point3f from_top;
	cv::Point3f from_bottom;
};


class Ladimo_Stereo_Matcher
{
public:
	// run this first to create accellerator structures
	void setLaserGrid(std::vector<GridPos> grid );
	void setSamplingfactor(int factor_);
	void setCameraMatrices(cv::Mat cam_mat1_, cv::Mat cam_mat2_);
	void setBaseline(float baseline_);
	void setDerivativesZThreshold(float threshold_);

	// information about laser point relation on image plane
	void setGridAngle(float angle_);
	void setAvgPointDistancePxc(float avg_pointdistance_pxc_);

	void run(const cv::Mat& image1, const cv::Mat& image2, std::vector<cv::Point3f>& observedDots);

private:
	// structures
	LaserGrid laser_grid{};
	std::vector<NeighborIndices> neighbor_indices;
	std::vector<Derivative> derivatives; // 3D derivatives
	std::vector<cv::Point3f> observations;
	std::vector<cv::Point3f> normals;
	
	// parameters
	int sampling_factor; // if 4, every square is sampled 4 x 4 times;

	cv::Mat cam_mat1;
	float f1;
	float cx1;
	float cy1;
	
	cv::Mat cam_mat2;
	float f2;
	float cx2;
	float cy2;

	float baseline;
	float avg_point_distance_pxc{ 17.f };
	float grid_angle{ -16.7f }; // degrees

	// TODO IMPROVE DERIVATIVE SANITY CHECK
	float derivative_z_threshold{ 150.f }; // sanity check value for derivatives 

	// internal methods
	void CreateNeighborIndices();
	void FillDataGaps();
	void CalculateDerivatives();
	void CalculateNormals();

};

