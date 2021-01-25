#pragma once

#include <iostream>

#include <opencv2/imgproc.hpp>

class ImageProcessingStandard
{
public:
	enum class FilteringTechnique
	{
		mean_filter,
		gaussian_filter,
		median_filter,
		bilateral_filter
	};

	// Set general kernel dimension and define kernel box for all the methods
	void setKernelSize(const int& kernel_size_) {
		kernel_size = kernel_size_;
		kernel_box = cv::Size(kernel_size_, kernel_size_);
	}

	/**
	 * Set overneeded Gaussian blur parameters:
	 * sigma_X
	 * sigma_Y
	 */
	void setGaussianBlurParams(const double& gaussian_sigma_X_, const double& gaussian_sigma_Y_) {
		gaussian_sigma_X = gaussian_sigma_X_;
		gaussian_sigma_Y = gaussian_sigma_Y_;
	}

	/**
	 * Set overneeded Bilateral blur parameters:
	 * diameter
	 * sigma
	 */
	void setBilateralParams(const int& diameter_bilateral_, const double& sigma_bilateral_) {
		diameter_bilateral = diameter_bilateral_;
		sigma_bilateral = sigma_bilateral_;
	}
	
	/**
	 * Set image pre-processing operations
	 * Mean blur
	 * Gaussian blur
	 * Median blur
	 * Bilateral blur
	 */
	void setProcessedImageStandard(const cv::Mat& input_image, int filter_technique);

	cv::Mat getProcessedImageStandard(int filter_technique);

private:
	// Parameters
	int kernel_size = 0;
	cv::Size kernel_box;
	double gaussian_sigma_X = 0.0;
	double gaussian_sigma_Y = 0.0;
	int diameter_bilateral = 0;
	double sigma_bilateral = 0.0;

	// Filtered images
	cv::Mat mean_blurred;
	cv::Mat gaussian_blurred;
	cv::Mat median_blurred;
	cv::Mat bilateral_blurred;
};

class ImageProcessingMorphology
{
public:
	enum class MorphologicalFilters
	{
		morph_open,
		morph_close,
		morph_dilation,
		morph_erosion,
		morph_mixed
	};

	// Setter
	void setProcessedParametersMorph(const int& morph_size, int kernel_shape);
	void setProcessedImageMorph(const cv::Mat& inputImage, int morph_technique);

	// Getter
	cv::Mat getProcessedImageMorph(int morph_technique);

private:
	cv::Mat morphElement;   // (size --> 2 * morph_size + 1)
	cv::Mat openingMat;
	cv::Mat closingMat;
	cv::Mat dilateMat;
	cv::Mat erodeMat;
	cv::Mat mixedMat;
};