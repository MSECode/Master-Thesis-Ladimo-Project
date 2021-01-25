#pragma once

#include <string>
#include <numeric>
#include <assert.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>


class LookupTable
{
public:
	// Setter
	void setLookupMatrix(const int& rows_number, const int& points_on_width, const int& points_on_height);

	// Getter
	cv::Mat getLookupMatrix();

private:
	cv::Mat lookup_matrix;
};

struct ObservationData
{
	ObservationData() {};
	ObservationData(double x_px_, double y_px_, double X_mt_, double Y_mt_, double Z_mt_, double disp_)
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
	ObservationData operator*(const double par)
	{
		return ObservationData(x_px * par,
			y_px * par,
			X_mt * par,
			Y_mt * par,
			Z_mt * par,
			disp * par);
	}

	cv::Vec4d geteEstimationValues()
	{
		return cv::Vec4d(X_mt, Y_mt, Z_mt, disp);
	}

	ObservationData getestimationValuesFull()
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

struct CostPenalties
{
	double P_0 = 25.0;         // general penalty (dealer)
	double P_1 = 5.0;          // small (blind) penalty
	double P_2 = 15.0;         // big (blind) penalty
	double P_3 = 5.0;          // first medium penalty for diagonal edges
	double P_4 = 12.0;         // second medium penalty for diagonal edges

	double P1_sgm = 8.0;
	double P2_sgm = 30.0;
};

bool isPixelInsideImage(cv::Point2i& left_pixel_position,
	cv::Point2i& right_pixel_position,
	cv::Mat& left_matrix_limits,
	cv::Mat& right_matrix_limits);

cv::Mat defineMatrixLimits(int top, int bottom, int left, int right);