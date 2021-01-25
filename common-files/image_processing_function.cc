#include "image_processing_function.h"

void ImageProcessingStandard::setProcessedImageStandard(const cv::Mat& input_image, int filter_technique)
{
	mean_blurred = cv::Mat(input_image.size(), input_image.type());
	gaussian_blurred = cv::Mat(input_image.size(), input_image.type());
	median_blurred = cv::Mat(input_image.size(), input_image.type());
	bilateral_blurred = cv::Mat(input_image.size(), input_image.type());

	switch (filter_technique)
	{
	case 0:
		// Mean blur
		cv::blur(input_image, mean_blurred, kernel_box);
		break;
	case 1:
		// Gaussian blur
		cv::GaussianBlur(input_image, gaussian_blurred, kernel_box, gaussian_sigma_X, gaussian_sigma_Y);
		break;
	case 2:
		// Median blur
		cv::medianBlur(input_image, median_blurred, kernel_size);
		break;
	case 3:
		// Bilateral blur
		cv::bilateralFilter(input_image, bilateral_blurred, diameter_bilateral, sigma_bilateral, sigma_bilateral);
		break;
	default:
		std::cout << "Other filters cannot be applied !";
		break;
	}

}

cv::Mat ImageProcessingStandard::getProcessedImageStandard(int filter_technique)
{
	switch (filter_technique)
	{
	case 0:
		return mean_blurred;
		break;
	case 1:
		return gaussian_blurred;
		break;
	case 2:
		return median_blurred;
		break;
	case 3:
		return bilateral_blurred;
		break;
	default:
		std::cout << "No filtered image to return " << std::endl;
		return cv::Mat{};
		break;
	}
}

void ImageProcessingMorphology::setProcessedParametersMorph(const int& morph_size, int kernel_shape)
{
	morphElement = cv::getStructuringElement(kernel_shape, cv::Size(2 * morph_size + 1, 2 * morph_size + 1));
}

void ImageProcessingMorphology::setProcessedImageMorph(const cv::Mat& inputImage, int morph_technique)
{
	openingMat = cv::Mat(inputImage.size(), inputImage.type());
	closingMat = cv::Mat(inputImage.size(), inputImage.type());
	dilateMat = cv::Mat(inputImage.size(), inputImage.type());
	erodeMat = cv::Mat(inputImage.size(), inputImage.type());
	mixedMat = cv::Mat(inputImage.size(), inputImage.type());

	switch (morph_technique)
	{
	case 0:
		// Opening
		cv::morphologyEx(inputImage, openingMat, cv::MORPH_OPEN, morphElement);
		break;
	case 1:
		// Closing
		cv::morphologyEx(inputImage, closingMat, cv::MORPH_CLOSE, morphElement);
		break;
	case 2:
		// Dilation
		cv::dilate(inputImage, dilateMat, morphElement);
		break;
	case 3:
		// Erosion
		cv::erode(inputImage, erodeMat, morphElement);
		break;
	case 4:
		// Mixed Case
		//cv::morphologyEx(inputImage, closingMat, cv::MORPH_CLOSE, morphElement);
		//cv::erode(inputImage, erodeMat, morphElement);
		//cv::morphologyEx(closingMat, mixedMat, cv::MORPH_OPEN, morphElement);
		cv::dilate(inputImage, dilateMat, morphElement);
		cv::erode(dilateMat, mixedMat, morphElement);
		break;
	default:
		std::cout << "No other morphological operation to perform !" << std::endl;
		break;
	}
}

cv::Mat ImageProcessingMorphology::getProcessedImageMorph(int morph_technique)
{
	switch (morph_technique)
	{
	case 0:
		return openingMat;
		break;
	case 1:
		return closingMat;
		break;
	case 2:
		return dilateMat;
		break;
	case 3:
		return erodeMat;
		break;
	case 4:
		return mixedMat;
		break;
	default:
		std::cout << "There's no other morphological filter that can be applied ! " << std::endl;
		break;
	}
}
