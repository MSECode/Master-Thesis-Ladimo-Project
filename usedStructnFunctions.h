#ifndef USEDSTRUCTNFUNCTIONS
#define USEDSTRUCTNFUNCTIONS

#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <numeric>
#include <assert.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "ioFunctions.h"

class MatrixnCalibration
{
public:
	// Setter
	void setLeftRightCalibrationMat(const std::string& cam_0_, const std::string& cam_1_, const char& delete_char);
	void setDoffs(const std::string& doffs_);
	void setBaseline(const std::string& baseline_);
	void setWidth(const std::string& width_);
	void setHeight(const std::string& height_);
	void setLeftRightGroundTruth(const std::string& ground_truth_0, const std::string& ground_truth_1);
	
	// Getter
	cv::Mat getLeftImage(const std::string& image_path);
	cv::Mat getRightImage(const std::string& image_path);
	cv::Mat getLeftCamMatrix();
	cv::Mat getRightCamMatrix();
	float getDoffs();
	float getBaseline();
	int getWidth();
	int getHeight();
	cv::Mat getLeftGT();
	cv::Mat getRightGT();

private:
	cv::Mat imageLeft;
	cv::Mat imageRight;
	cv::Mat groundTruthLeft;
	cv::Mat groundTruthRight;
	cv::Mat cam_0;
	cv::Mat cam_1;
	float disparity_offset;
	float camera_baseline;
	int image_width;
	int image_height;
};


class GridParameters
{
public:
	// Setters
	void setGap(int gap_);
	void setSamples(int samples_);
	void setGridPoints(const int& im_width, const int& im_height);

	// Getters
	int getGap();
	int getSamples();
	int getWidthPoints();
	int getHeightPoints();

private:
	int gap;
	int samples;
	int widthPoints;
	int heightPoints;
};

class LadimoGridData
{
public:
	// Setter
	void setPixelData(const int& gap, const cv::Mat& left_gt, const cv::Mat& right_gt, const int& points_on_width, const int& points_on_height);
	void setSpaceData(const float& baseline, const float& doffs, const cv::Mat& cam_0, const cv::Mat& cam_1, const int& points_on_width, const int& points_on_height);

	// Getter
	cv::Mat getPixelDataLeft();
	cv::Mat getPixelDataRight();
	cv::Mat getSpaceDataLeft();
	cv::Mat getSpaceDataRight();

private:
	cv::Mat pixelDataLeft;
	cv::Mat pixelDataRight;
	cv::Mat spaceDataLeft;
	cv::Mat spaceDataRight;
};

class RealLadimoGridData
{
public:
	// Setter
	void setPixelnSpaceData(const float& baseline, const cv::Mat& cam_0, const cv::Mat& cam_1, const int& gap, const cv::Mat& left_gt, const cv::Mat& right_gt, const int& points_on_width, const int& points_on_height);
	
	// Getter
	cv::Mat getPixelDataLeft();
	cv::Mat getPixelDataRight();
	cv::Mat getSpaceDataLeft();
	cv::Mat getSpaceDataRight();

private:
	cv::Mat pixelDataLeft;
	cv::Mat pixelDataRight;
	cv::Mat spaceDataLeft;
	cv::Mat spaceDataRight;
};


class LookupTable
{
public:
	// Setter
	void setLookupMatrix(const int& rows_number, const int& points_on_width, const int& points_on_height);

	// Getter
	cv::Mat getLookupMatrix();

private:
	cv::Mat lookupMatrix;
};

struct ObservationData
{
	ObservationData() {};
	ObservationData(double x_px_, 
		double y_px_, 
		double X_mt_, 
		double Y_mt_, 
		double Z_mt_, 
		double disp_)
	{
		x_px = x_px_;
		y_px = y_px_;
		X_mt = X_mt_;
		Y_mt = Y_mt_;
		Z_mt = Z_mt_;
		disp = disp_;
	};

	ObservationData operator+(const ObservationData& obs) const
	{
		return ObservationData(x_px + obs.x_px,
			y_px + obs.y_px,
			X_mt + obs.X_mt,
			Y_mt + obs.Y_mt,
			Z_mt + obs.Z_mt,
			disp + obs.disp);
	}

	ObservationData operator-(const ObservationData& obs) const
	{
		return ObservationData(x_px - obs.x_px,
			y_px - obs.y_px,
			X_mt - obs.X_mt,
			Y_mt - obs.Y_mt,
			Z_mt - obs.Z_mt,
			disp - obs.disp);
	}
	ObservationData operator*(double par)
	{
		return ObservationData(x_px * par,
			y_px * par,
			X_mt * par,
			Y_mt * par,
			Z_mt * par,
			disp * par);
	}

	cv::Vec4f estimationValues()
	{
		return cv::Vec4f(X_mt, Y_mt, Z_mt, disp);
	}

	ObservationData estimationFullValues()
	{
		return ObservationData(x_px, y_px, X_mt, Y_mt, Z_mt, disp);
	}

	double x_px{};
	double y_px{};
	double X_mt{};
	double Y_mt{};
	double Z_mt{};
	double disp{};
};


struct Derivatives
{
	cv::Point3f from_top;
	cv::Point3f from_bot;
	cv::Point3f from_left;
	cv::Point3f from_right;
	void show();
};

struct TotalDerivatives
{
	cv::Vec4f der_top;
	cv::Vec4f der_bot;
	cv::Vec4f der_left;
	cv::Vec4f der_right;
	void show();
};

struct CompleteValuesDerivatives
{
	ObservationData top_deriv{0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
	ObservationData bottom_deriv{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
	ObservationData left_deriv{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
	ObservationData right_deriv{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
};

struct CostPenalties
{
	float P_0 = 25.f;         // general penalty (dealer)
	float P_1 = 5.f;          // small (blind) penalty
	float P_2 = 15.f;         // big (blind) penalty
	float P_3 = 5.f;          // first medium penalty for diagonal edges
	float P_4 = 12.f;         // second medium penalty for diagonal edges

	float P1_sgm = 8.f;
	float P2_sgm = 30.f;
};

// Final Matrices
struct EdgeShapeEstimationMatrices
{
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
};


class MatrixLimits {
public:
	cv::Mat defineMatrixLimits(int top, int bottom, int left, int right);
};

cv::Mat getExportedMatrix(const cv::Mat& inputMatrix);

// Structures for realLaDimo Grid


struct GridSquare 
{
	int top_left_index{-1};
	int top_right_index{-1};
	int bottom_left_index{-1};
	int bottom_right_index{-1};
};

struct NeighbourIndexes
{
	int top{ -1 };
	int bot{ -1 };
	int left{ -1 };
	int right{ -1 };

	bool isFull()
	{
		bool is_full = false;
		if (top != -1 && left != -1 && right != -1 && bot != -1) is_full = true;
		return is_full;
	};
};

struct GridPositions
{
	GridPositions() {};
	GridPositions(int x_, int y_) {
		x = x_;
		y = y_;
	};

	int x = 0;
	int y = 0;
	bool equals(const GridPositions& gp) {
		return (x == gp.x && y == gp.y);
	}
	GridPositions left() {
		return	GridPositions(x - 1, y);
	}
	GridPositions right() {
		return	GridPositions(x + 1, y);
	}
	GridPositions top() {
		return	GridPositions(x, y - 1);
	}
	GridPositions bottom() {
		return	GridPositions(x, y + 1);
	}
};

struct LaserGrid
{
	std::vector<GridPositions> grid;
};


#endif // !USEDSTRUCTNFUNCTIONS
