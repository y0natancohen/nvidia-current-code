#pragma once
#ifndef CIMPROCALGO_H
#define CIMPROCALGO_H

#include "CPreprocessIm.h"
#include "CChaudhuri.h"
#include "CPostProcChaudhuri.h"

class CImProcAlgo
{
public:
	//methods
	//////////////////////////////////////////////////////////////////////////////
	///	The function getReference is used in order to define the class as singleton
	///		since the constructor and distructor are private.
	/////////////////////////////////////////////////////////////////////////////
	static CImProcAlgo& getReference();


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
	*	The method springImProcAlgo is the main call for spring's blood vessels
	*		extraction algorithm. Getting an image we prepare the image for
	*		Chaudhuri basic algorithm. Post process then takes place inorder to
	*		resuce Chaudhuri algorithm build in noise.
	*
	*	REMARK: Please notice that ALL algorithm parameters are read from
	*			*.ini file
	*
	*	INPUT:
	*		src - cv::Mat - input image can be either color image or single channel
	*					we can handle all opencv supported image types.
	*		dst - cv::Mat - depends on user's definition output will set comply to
	*					*.ini file parameter. output can be any single channel opencv
	*					depth.
	*
	*	OUTPUT:
	*		EReport error codes.
	*
	***************************************************************************/
	EReport springImProcAlgo(cv::Mat &src, cv::Mat &dst);
	//members
private:
	//methods
	CImProcAlgo();
	~CImProcAlgo();

	//members
	bool m_isInit;	//indicates if parameters read properly from *.ini file and all
					//required initialization are properly performed
    cv::Mat m_mask; //holds image mask produced by preprocesssIm
    cv::Mat m_imPp; //holds preprocessed image - chaudhuri algo input
    cv::Mat m_chaud;    // holds chaudhuri algo output
#if CUDA_OPTIM
    cv::cuda::GpuMat cm_mask; //holds image mask produced by preprocesssIm
    cv::cuda::GpuMat cm_imPp; //holds preprocessed image - chaudhuri algo input
    cv::cuda::GpuMat cm_chaud;    // holds chaudhuri algo output
#endif // CUDA_OPTIM
};

#endif //CIMPROCALGO_H
