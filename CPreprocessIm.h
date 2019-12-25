#pragma once
#ifndef CPREPROCESS_IM_H
#define CPREPROCESS_IM_H

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "CIniReader.h"
#include "structures.h"

enum EBlurMethods
{
	EBM_GAUSSIAN = 0,
	EBM_BLUR = 1,
	EBM_MEDIAN = 2,
	EBM_BILATERAL = 3,

	EBM_COUNT
};

struct SThresolding
{
	int cvThresholdType;		// can get the following types: THRESH_BINARY, THRESH_BINARY_INV,
							// THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV, THRESH_OTSU, THRESH_TRIANGLE 
							// - combine all the abobe with OTSU (THRESH_BINARY )
	int otherThreshType;	// 0 - zero type, 1 - adaptive, -1 == do not use this flag. 
	int adaptiveThresType;	// can get the following: ADAPTIVE_THRESH_MEAN_C , ADAPTIVE_THRESH_GAUSSIAN_C 
	int blockSize;			// can get only odd values: 3,5,7,....
	double maxValue;		// value assigned to pixels that satisfies condition 
	double C;				// constant reduced from local environment 
	double thresh;			// value assigned for thresholding
	double threshFactor;	// factor to modify auto threshold

	SThresolding(): cvThresholdType(cv::THRESH_BINARY | cv::THRESH_OTSU), otherThreshType(-1),
		adaptiveThresType(cv::ADAPTIVE_THRESH_MEAN_C), blockSize(3),
		maxValue(1.0), C(0), thresh(0.0), threshFactor(0.5) {}
};

struct SBlurParameters
{
	bool isConvertToUint;	// if true after filtering result is converted to CV_U16 or CV_U8 depends on source image depth
	double sigmaX;		//in gaussian x direction - in bilateral color std
	double sigmaY;		//in gaussian y direction - if 0 as x, in bilateral std in coordinate space
	double diameter;	//in bilateral use only each pixel neighborhood diameter
	cv::Size winSize;	//filtering window size
	EBlurMethods method;//method one of the enum options

	SBlurParameters() : isConvertToUint(false),sigmaX(1.0), sigmaY(0.0), diameter(1.0), winSize(3, 3),
		method(EBM_GAUSSIAN) {}
};


struct SNiblackParameters
{
	cv::Size winSize;
	double kapa;
	double offset;

	SNiblackParameters(): winSize(100,100), kapa(0.2), offset(0.0) {}
};

class CPreprocessIm
{
public:
	//methods
	//////////////////////////////////////////////////////////////////////////////
	///	The function getReference is used in order to define the class as singleton
	///		since the constructor and distructor are private. 
	/////////////////////////////////////////////////////////////////////////////
	static CPreprocessIm& getReference();

	/***************************************************************************
	*
	*	The method init() initialize class members as read from file, upon success
	*		the method modify the flag m_isInit (if this member is false, the main
	*		methods terminate with error code - at all the project is it the same ini file
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
	*	The method preprocessIm preprocess the input image for chaudhuri algorithm,
	*		and creates masking image (if the proper flag is on).
	*	Preprocess parameters are defind as class members, and read from ini file.
	*	parameters properties modification can be done via ini file, reference is
	*		at ini file and hereinabove.
	*
	*	INPUT:
	*		src, dst, mask - cv::Mat - single channels matrices (if not single channel the
	*							input image is converted to gray).
	*		
	*
	*	OUTPUT:
	*		EReport error codes.
	*
	***************************************************************************/
	EReport preprocessIm(cv::Mat & src, cv::Mat &dst, cv::Mat &mask);

private:
	// methods

	/***************************************************************************
	*
	*	The method createMask uses opencv build-in thresholding methods in order to
	*		separate foreground from background i.e. masking operation. 
	*		We left an option for other methods masking implementations (not opencv). 
	*
	*	INPUT:
	*		src, mask - cv::Mat - single channels matrices (if not single channel the
	*							input image is converted to gray).
	*		params - SThresolding - see ref above.
	*		isBlur - bool - we are using heavy blurring prior to masking. we left an 
	*					option to avoid it only using hard coded option(!!!).
	*
	*	OUTPUT:
	*		EReport error codes.
	*
	***************************************************************************/
	EReport createMask(cv::Mat &src, cv::Mat &mask, SThresolding &params, bool isBlur = true);

	/***************************************************************************
	*
	*	The method blurImg uses opencv build-in bluring methods. In this implementation
	*		we give the user several bluring options: EBM_BLUR, EBM_MEDIAN, EBM_BILATERAL,
	*		and EBM_GAUSSIAN as default option. 
	*
	*	INPUT:
	*		src, dst - cv::Mat - single channels matrices (if not single channel the
	*							input image is converted to gray).
	*		params - SBlurParameters - see ref above.
	*
	*	OUTPUT:
	*		EReport error codes.
	*
	***************************************************************************/
	EReport blurImg(cv::Mat &src, cv::Mat &dst, SBlurParameters params);

	/***************************************************************************
	*
	*	The method niblack implements niblack algorithm. 
	*
	*	INPUT:
	*		src, outIm, outNiBl - cv::Mat - single channels matrices (if not single channel the
	*							input image is converted to gray). Where outIm is processed image
	*							while outNiBl - is binary image (can be used as mask).
	*		params - SNiblackParameters - see ref above.
	*
	*	OUTPUT:
	*		EReport error codes.
	*
	***************************************************************************/
	EReport niblack(cv::Mat &src, cv::Mat & outIm, cv::Mat &outNiBl, SNiblackParameters params);

	/***************************************************************************
	*
	*	The method illumEnhance implements CLAHE algorithm (adaptive histogram equalization).
	*	CLAHE parameters are defiend at project's *.INI file
	*
	*	INPUT:
	*		src, dst - cv::Mat - single channels matrices (if not single channel the
	*							input image is converted to gray). dst - enhanced im.
	*
	*	OUTPUT:
	*		EReport error codes.
	*
	***************************************************************************/
	EReport illumEnhance(cv::Mat &src, cv::Mat &dst);


	CPreprocessIm();
	~CPreprocessIm();	
	
	// members
	bool m_isInit;			// *.ini reading initialization verifier
	bool m_isBlur;			// preprocess bluring
	bool m_isNiBlack;		// if true, niblack algo is applied. 
	bool m_isMask;			// if true a mask is created. 
	bool m_isNormalizeClahe;	// if true image will be normalized before casting to uint for clahe algo
	int m_convert2type;		// convert ti type CV_8U or CV_16U for CLAHE algo
	int m_srcType;			// on run time, holds source image type
	double m_clipLimit;			// CLAHE parameter - see ref opencv 
	cv::Size m_tileGridSize;	// CLAHE parameter - see ref opencv
	SThresolding m_imProcThresh;			// foreground\background parameters
	SBlurParameters m_blurParams;			// bluring parameters
	SNiblackParameters m_niblackParams;		// niblack parameters
	cv::Ptr<cv::CLAHE> m_clahe;				// CLAHE object. 
};

#endif // !CPREPROCESS_IM_H