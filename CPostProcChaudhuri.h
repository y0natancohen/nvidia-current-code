#pragma once
#ifndef C_POSTPROCCHAUDHURI_H
#define C_POSTPROCCHAUDHURI_H

#include <iostream>
#include <vector>
#include <numeric>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "CIniReader.h"
#include "structures.h"

struct SPostProcParams
{
	bool isNormalize;		// normalization flag
	bool isDisplayProgress;
	bool isConvert;
	int imDepth;
	int morphKerLength;
	int numMorphKerRotations;
	double dispCutMin;
	double dispCutMax;
	bool isDilate;
	cv::Point circleCenter;
	int circlRadiusDilate;
	int circlRadiusClose;
	int adaptiveLowBlockSize;
	int adaptiveHighBlockSize;
	int adaptiveMeanShift;
	int adaptiveMaxValue;
	bool isAutoBrightness;
	double startCutValue;
	double threshCutValue;
	int numMorphItter;
	bool isCvDisplay;
	bool isUseDispCut;


	SPostProcParams() : isNormalize(false), isDisplayProgress(false), isConvert(false),
		imDepth(CV_16U), morphKerLength(17), numMorphKerRotations(36), dispCutMin(0.0),
		dispCutMax(1.0), isDilate(true), circleCenter(0,0), circlRadiusDilate(2),
		circlRadiusClose(3), adaptiveLowBlockSize(61), adaptiveHighBlockSize(201),
		adaptiveMeanShift(-4), adaptiveMaxValue(255), isAutoBrightness(true), startCutValue(0.25),
		threshCutValue(0.05), numMorphItter(2), isCvDisplay(false), isUseDispCut(false){};
};


class CPostProcChaudhuri
{
public:
	//methods
	//////////////////////////////////////////////////////////////////////////////
	///	The function getReference is used in order to define the class as singleton
	///		since the constructor and distructor are private.
	/////////////////////////////////////////////////////////////////////////////
	static CPostProcChaudhuri& getReference();
	EReport init(std::string &filename);
#if CUDA_OPTIM
	EReport postProcIm(cv::Mat &src, cv::Mat &dst);
#endif
#if CUDA_OPTIM
    EReport postProcIm(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);
#endif // CUDA_OPTIM
	/****************************************************************
	*
	*			set of service methods set\get - public
	*
	*****************************************************************/
	void getKernels(std::vector<cv::Mat> &kernels);
	// members

private:
	//methods
#if CUDA_OPTIM
	EReport morphChaudhuri(cv::Mat &src, cv::Mat &dst);
	EReport adaptiveThresholding(cv::Mat &src, cv::Mat &dst);
	EReport connectingComponents(cv::Mat &src, cv::Mat &dst);
#endif
#if CUDA_OPTIM
    EReport morphChaudhuri(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);
	EReport adaptiveThresholding(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);
	EReport connectingComponents(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);
#endif // CUDA_OPTIM
	EReport constructKernels();
	EReport strel2D(int length, double angle, cv::Mat &linKer);
	EReport createCircle(cv::Mat &retPattern, cv::Point center = cv::Point(0, 0), int radius = 2);
	EReport createCirclePattern();
	EReport modulu(double x, double y, double &mod);
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


	/***************************************************************************
	*
	*	The method normalizeIm works in two phases. one, normalizing image values
	*		to range (0, max of unit8 or uint16), and then cuts the values relative
	*		to predefined range in [0,1] section.
	*	OUTPUT:
	*		EReport error codes.
	*
	*
	*	INPUT:
	*		src, dst - cv::Mat - processed grayscale images
	*		params - SChaudhuriParams - we refer you above
	*
	***************************************************************************/
	EReport normalizeIm(cv::Mat &src, cv::Mat &dst, SPostProcParams params);
	EReport normalizeIm(cv::Mat &src, cv::Mat &dst, double dispCutMin, double dispCutMax);

	EReport setAutoBrightnessLevel(cv::Mat &src, double &dispCut);


	CPostProcChaudhuri();
	~CPostProcChaudhuri();


	//members
	bool m_isInit;
	bool m_isParamInit;
	SPostProcParams m_postProcParams;
	std::vector<cv::Mat> m_morphKernels;
	cv::Mat m_circleKerDilate;
	cv::Mat m_circleKerClose;
#if CUDA_OPTIM
    std::vector<cv::cuda::GpuMat> cm_morphKernels;
	cv::cuda::GpuMat cm_circleKerDilate;
	cv::cuda::GpuMat cm_circleKerClose;
#endif // CUDA_OPTIM

};

#endif // C_POSTPROCCHAUDHURI_H
