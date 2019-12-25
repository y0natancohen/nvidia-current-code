#pragma once
#ifndef CCHAUDHURI_H
#define CCHAUDHURI_H

//#define CUDA_OPTIM 1

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "structures.h"
#include "CIniReader.h"


struct SChaudhuriParams
{
	bool isNormalize;		// normalization flag
	bool isDisplayProgress;	// if true algo progress is exported to terminal
	bool isConvert;			// if true processed image will be converted to imDepth type
	int imDepth;			// convertion type for algorithm output - all opencv single channel types
	cv::Mat mask;			// holds image mask if needed
	double dispCutMin;		// after normalization all values below this value are set to it range -[0 1]
	double dispCutMax;		// after normalization all values above this value are set to it range -[0 1]
	double L;				// width of kernel
	double T;				// height of kernel
	double sigma;			// sigma - see article
	int numOfKerRotations;	// number of kernel rotations
	int imRows;				// image rows number
	int imCols;				// image colums number
	int cudaOptimRows;      // convolve optimization window 0: cuda optim
	int cudaOptimCols;      // convolve optimization window 0: cuda optim

	SChaudhuriParams() : isNormalize(false), isDisplayProgress(true),isConvert(false),
		imDepth(CV_16U), dispCutMin(0.0), dispCutMax(1.0), L(10.8), T(8.), sigma(1.9),
		numOfKerRotations(12), imRows(2080), imCols(3096), cudaOptimRows(0),
		cudaOptimCols(0) {};
};


//////////////////////////////////////////////////////////////////////////////////
//// Class - CChaudhuri
////	The class is containing all aspects of Chaudhuri et al. basic algorithm.
////	Generally speaking several kernels (private case of Gabor kernels) are
////	created and the image is filterred using them gathering all local responses
////	maxima (pixel-wise) of all filters.
////
////	The class is static singelton. All needed parameters are retrieved from *.ini
////	file.
////	At CUDA optimization all dynamic matrices will be initiated apriory at init stage
////	and will be class members.
///////////////////////////////////////////////////////////////////////////////////
class CChaudhuri
{
public:
	//methods
	//////////////////////////////////////////////////////////////////////////////
	///	The function getReference is used in order to define the class as singleton
	///		since the constructor and distructor are private.
	/////////////////////////////////////////////////////////////////////////////
	static CChaudhuri& getReference();

	/***************************************************************************
	*
	*	The method init() initialize class members as read from file, upon success
	*		the method modify the flag m_isInit (if this member is false, the main
	*		methods terminate with error code
	*	INPUT:
	*		filename - string - full path to *.ini file
	*
	*	OUTPUT:
	*		EReport error codes.
	*
	***************************************************************************/
	EReport init(std::string &filename);

	/***************************************************************************
	*
	*	The method calculateChaudhuri calculates chaudhuri blood vessels detection
	*		algorithm using a vector of predefined kernels aplied on the source image
	*		maximizing the responses.
	*	INPUT:
	*		src, dst - cv::Mat - single channels matrices (if not single channel the
	*							input image is converted to gray).
	*
	*	OUTPUT:
	*		EReport error codes.
	*
	***************************************************************************/
#if !CUDA_OPTIM //made this way in order to avoid too many changes
	EReport calculateChaudhuri(cv::Mat &src, cv::Mat &dst);
#endif
#if CUDA_OPTIM
	EReport calculateChaudhuri(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);
#endif // CUDA_OPTIM

	/***************************************************************************
	*
	*	General service get\set methods
	*
	***************************************************************************/
	void getKernels(std::vector<cv::Mat> &kerList);
	void setMask(cv::Mat &mask);
#if CUDA_OPTIM
	void setCudaMask(cv::cuda::GpuMat &mask);
#endif // CUDA_OPTIM

private:
	// methods

	/***************************************************************************
	*
	*	The method constructKernels creates Chaudhuri kernels
	*
	*	INPUT:
	*
	*	OUTPUT:
	*		EReport error codes.
	*
	***************************************************************************/
	EReport constructKernels();

	/***************************************************************************
	*
	*	The method getMeshgrid2D complies with matlab\python meshgrid
	*
	*	INPUT:
	*		xRange, yRange - cv::Range - vector of entries
	*		X,Y - cv::Mat - meshgrids
	*
	*	OUTPUT:
	*		EReport error codes.
	*
	***************************************************************************/
	EReport getMeshgrid2D(const cv::Range &xRange,
		const cv::Range &yRange,
		cv::Mat &X,
		cv::Mat &Y);

	/***************************************************************************
	*
	*	The method normalizeIm works in two phases. one, normalizing image values
	*		to range (0, max of unit8 or uint16), and then cuts the values relative
	*		to predefined range in [0,1] section.
	*
	*	INPUT:
	*		src, dst - cv::Mat - processed grayscale images
	*		params - SChaudhuriParams - we refer you above
	*
	*	OUTPUT:
	*		EReport error codes.
	*
	***************************************************************************/
#if! CUDA_OPTIM
	EReport normalizeIm(cv::Mat &src, cv::Mat &dst, SChaudhuriParams &params);
#endif
#if CUDA_OPTIM
	EReport normalizeIm(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, SChaudhuriParams &params);

    EReport createNormMask(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, double &compareVal, const std::string &oper);
#endif

	/***************************************************************************
	*
	*	The method linspace is similar to python linspace
	*
	*	INPUT:
	*		start_in - start number
	*		end_in - end number
	*		num_in - number of steps between start and end points
	*		linspaced - std::vector<double> - the output division of the segment to
	*			equivalent parts (including start and end points)
	*
	*	OUTPUT:
	*		EReport error codes.
	*
	***************************************************************************/
	template<typename T>
	EReport linspace(T start_in, T end_in, int num_in, std::vector<double> &linspaced);

	inline double round(double inNum, double numDigits)
	{
		double retNum(inNum);
		if (numDigits == 0)
		{
			retNum = std::lround(inNum);
		}
		else if (numDigits != 0)
		{
			retNum = static_cast<double>(std::lround(inNum * pow(10., numDigits)) / (pow(10., numDigits)));
		}
		return retNum;
	}

	CChaudhuri();
	~CChaudhuri();

	// members
	bool m_isInit;				// is algorithm initialized correctly
	bool m_isGPUOptimization;	// if there is GPU one should turn it to true using init file
	bool m_isInitParams;		// if parameters read correctly from ini file
	bool m_isCalculateLsigma;	// if true - the parameters L & sigma at m_chaudhuriParams will be updated relative to input image
	SChaudhuriParams m_chaudhuriParams;		// see ref above
	std::vector<double> m_rotationAngles;	//holds kernel rotation angles
	std::vector<cv::Mat> m_chadhuriKers;	//holds the lernels themselfs
#if CUDA_OPTIM
    cv::Ptr<cv::cuda::Convolution> cm_convolver;
    // std::vector<cv::Ptr<cv::cuda::Filter>> cm_filter2D; // tested option slower than CPU (one should check it)
	std::vector<cv::cuda::GpuMat> cm_chadhuriKers; // holds an upload of kernels to gpu
    cv::cuda::GpuMat cm_mask;   //gpu masking elements
    cv::cuda::GpuMat cm_gray, cm_conv, cm_bgr[3];    //gpu matrices for image process initialized to image size
    cv::cuda::GpuMat cm_srcCopy, cm_normIm, cm_normMask;   // used for image normaliztion
#endif
};

#endif // !CCHAUDHURI_H
