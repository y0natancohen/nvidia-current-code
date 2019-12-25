#include "CPreprocessIm.h"


/*///////////////////////////////////////////////////////////////
////////  Public methods implementation ////////////////////////
///////////////////////////////////////////////////////////////*/
CPreprocessIm& CPreprocessIm::getReference()
{
	static CPreprocessIm m_CPreprocessIm;
	return m_CPreprocessIm;
}


/****************************************************************
*
*			init - public
*
*****************************************************************/
EReport CPreprocessIm::init(std::string &filename)
{
	EReport er(ER_SUCCESS);
	EReadERR rv(EER_SUCCESS), retRead(EER_SUCCESS), rvWhileLoop(EER_SUCCESS);
	int lineNumber(0);
	std::string currLine, retName, retString;
	float retFloat(0);
	bool retBool(false), isHeader(false);
	cv::Rect2f retRect;

	while (EER_EOF != rvWhileLoop)
	{
		rvWhileLoop = CIniReader::getReference().readLine(filename, lineNumber, currLine);
		rv = rvWhileLoop;
		if (EER_UNDEFINED_FILE != rv)
		{
			//parse line
			rv = CIniReader::getReference().parseLine(currLine, retName, retString, retFloat, retBool, retRect, retRead, isHeader);
			if (EER_EMPTY_LINE != rv && EER_COMMENT != rv && !isHeader)
			{
				if (retName == "isBlur")
				{
					m_isBlur = retBool;
				}
				else if (retName == "isNiBlack")
				{
					m_isNiBlack = retBool;
				}
				else if (retName == "isMask")
				{
					m_isMask = retBool;
				}
				else if (retName == "clipLimit")
				{
					m_clipLimit = retFloat;
				}
				else if (retName == "tileGridSize")
				{
					cv::Rect tmp = retRect;
					m_tileGridSize.width = tmp.x;
					m_tileGridSize.height = tmp.y;
				}
				else if (retName == "cvThresholdType")
				{
					if (retString == "THRESH_OTSU")
					{
						m_imProcThresh.cvThresholdType = cv::THRESH_BINARY | cv::THRESH_OTSU;
					}
					else if (retString == "THRESH_TRIANGLE")
					{
						m_imProcThresh.cvThresholdType = cv::THRESH_BINARY | cv::THRESH_TRIANGLE;
					}
					else if (retString == "THRESH_BINARY")
					{
						m_imProcThresh.cvThresholdType = cv::THRESH_BINARY;
					}
					else if (retString == "THRESH_TRUNC")
					{
						m_imProcThresh.cvThresholdType = cv::THRESH_TRUNC;
					}
					else if (retString == "THRESH_TOZERO")
					{
						m_imProcThresh.cvThresholdType = cv::THRESH_TOZERO;
					}
				}
				else if (retName == "otherThreshType")
				{
					m_imProcThresh.otherThreshType = retFloat;
				}
				else if (retName == "adaptiveThresType")
				{
					if (retString == "ADAPTIVE_THRESH_MEAN_C")
					{
						m_imProcThresh.adaptiveThresType = cv::ADAPTIVE_THRESH_MEAN_C;
					}
					else if (retString == "ADAPTIVE_THRESH_GAUSSIAN_C")
					{
						m_imProcThresh.adaptiveThresType = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
					}
				}
				else if (retName == "blockSize")
				{
					m_imProcThresh.blockSize = retFloat;
				}
				else if (retName == "maxValue")
				{
					m_imProcThresh.maxValue = retFloat;
				}
				else if (retName == "C")
				{
					m_imProcThresh.C = retFloat;
				}
				else if (retName == "thresh")
				{
					m_imProcThresh.thresh = retFloat;
				}
				else if (retName == "threshFactor")
				{
					m_imProcThresh.threshFactor = retFloat;
				}
				else if (retName == "sigmaX")
				{
					m_blurParams.sigmaX = retFloat;
				}
				else if (retName == "sigmaY")
				{
					m_blurParams.sigmaY = retFloat;
				}
				else if (retName == "diameter")
				{
					m_blurParams.diameter = retFloat;
				}
				else if (retName == "winSize")
				{
					m_blurParams.winSize.width = retRect.x;
					m_blurParams.winSize.height = retRect.y;
				}
				else if (retName == "method")
				{
					if (retString == "EBM_GAUSSIAN")
					{
						m_blurParams.method = EBM_GAUSSIAN;
					}
					else if (retString == "EBM_BLUR")
					{
						m_blurParams.method = EBM_BLUR;
					}
					else if (retString == "EBM_MEDIAN")
					{
						m_blurParams.method = EBM_MEDIAN;
					}
					else if (retString == "EBM_BILATERAL")
					{
						m_blurParams.method = EBM_BILATERAL;
					}
				}
				else if (retName == "isConvertToUint")
				{
					m_blurParams.isConvertToUint = retBool;
				}
				else if (retName == "isNormalizeClahe")
				{
					m_isNormalizeClahe = retBool;
				}
				else if (retName == "convert2type")
				{
					if (retString == "CV_8U")
					{
						m_convert2type = CV_8U;
					}
					else if (retString == "CV_16U")
					{
						m_convert2type = CV_16U;
					}
				}
				else if (retName == "NBwinSize")
				{
					m_niblackParams.winSize.width = retRect.x;
					m_niblackParams.winSize.height = retRect.y;
				}
				else if (retName == "kapa")
				{
					m_niblackParams.kapa = retFloat;
				}
				else if (retName == "offset")
				{
					m_niblackParams.offset = retFloat;
				}

			}
			lineNumber++;
		}
		else
		{
			er = ER_INVALID_HANDLE;
		}
	}
	
	if (ER_SUCCESS == er)
	{
		m_isInit = true;
	}
	return er;
}

/****************************************************************
*
*			preprocessIm - public
*
*****************************************************************/
EReport CPreprocessIm::preprocessIm(cv::Mat & src, cv::Mat &dst, cv::Mat &mask)
{
	EReport rv(ER_SUCCESS);

	if (src.data)
	{
		if ((src.rows > 0) && (src.cols > 0) && (src.channels() == 1))
		{
			if (m_isInit)
			{
				m_srcType = src.type();
				cv::Mat blurIm, outNibl;
				if (m_isMask)
				{
					rv = createMask(src, mask, m_imProcThresh);
				}
				else
				{
					mask = cv::Mat::zeros(src.rows, src.cols, src.type());
				}
				if (ER_SUCCESS == rv)
				{
					if (m_isBlur)
					{
						rv = blurImg(src, blurIm, m_blurParams);
					}
					else
					{
						src.copyTo(blurIm);
					}
				}
				if (ER_SUCCESS == rv)
				{
					if (m_isNiBlack)
					{
						rv = niblack(blurIm, blurIm, outNibl, m_niblackParams);
					}
				}
				if (ER_SUCCESS == rv)
				{
					rv = illumEnhance(blurIm, blurIm);
				}
				if (ER_SUCCESS == rv)
				{
					if (m_isMask)
					{
						if (mask.type() != blurIm.type())
						{
							mask.convertTo(mask, blurIm.type());
						}
						cv::multiply(blurIm, mask, blurIm);
					}
					SBlurParameters params;
					params.winSize = cv::Size(5, 5);
					blurImg(blurIm, dst, params);
				}
			}
			else
			{
				rv = ER_INVALID_INIT;
			}
		}
		else
		{
			rv = ER_INVALID_DIMS;
		}
	}
	else
	{
		rv = ER_INVALID_INPUT;
	}

	return rv;
}

/*///////////////////////////////////////////////////////////////
////////  Private methods implementation ////////////////////////
///////////////////////////////////////////////////////////////*/

/****************************************************************
*
*			createMask - private
*
*****************************************************************/
EReport CPreprocessIm::createMask(cv::Mat &src, cv::Mat &mask, SThresolding &params, bool isBlur)
{
	EReport rv(ER_SUCCESS);

	if (src.data)
	{
		if ((src.rows > 0) && (src.cols > 0) && (src.channels() == 1))
		{
			mask = cv::Mat::zeros(src.rows, src.cols, src.type());
			cv::Mat img;
			src.copyTo(img);
			if (isBlur)
			{
				cv::GaussianBlur(img, img, cv::Size(11, 11), 1);
			}
			if (params.otherThreshType == -1)
			{
				cv::threshold(img, mask, params.thresh, 1., params.cvThresholdType);
				mask = cv::Mat::zeros(src.rows, src.cols, src.type());
				mask.setTo(1, img > params.thresh*params.threshFactor);
			}
			else // pick other threshold methods
			{
				if (params.otherThreshType == 0)
				{
					cv::threshold(img, mask, params.thresh, 1., cv::THRESH_BINARY);
				}
				else
				{
					cv::threshold(img, mask, params.thresh, 1., cv::THRESH_BINARY);
				}
			}
		}
		else
		{
			rv = ER_INVALID_DIMS;
		}
	}
	else
	{
		rv = ER_INVALID_INPUT;
	}

	return rv;
}

/****************************************************************
*
*			blurImg - private
*
*****************************************************************/
EReport CPreprocessIm::blurImg(cv::Mat &src, cv::Mat &dst, SBlurParameters params)
{
	EReport rv(ER_SUCCESS);
	if (src.data)
	{
		if ((src.rows > 0) && (src.cols > 0) && (src.channels() == 1))
		{
			double srcFactor(1.0);
			cv::Mat cvtSrc, cvtDst;
			if (params.method != EBM_BILATERAL)
			{
				src.convertTo(cvtSrc, CV_32F);
				if (src.type() == CV_8U)
				{
					srcFactor = 1. / 255.;
				}
				else if (src.type() == CV_16U)
				{
					srcFactor = 1. / 65535.;
				}
				else if (src.type() == CV_32F)
				{
					srcFactor = 1.0;
				}
				else
				{
					// convert source to 8bit and set compatible factor 
					double m(0.), M(0.);
					cv::minMaxLoc(cvtSrc, &m, &M);
					if (M > m)
					{
						cvtSrc = (cvtSrc - m) / (M - m);
						srcFactor = 1. / 255.;
					}
				}
				cvtSrc = cvtSrc * srcFactor;
			}
			else
			{
				src.copyTo(cvtSrc);
			}
			switch (params.method)
			{
			case EBM_GAUSSIAN:
				cv::GaussianBlur(cvtSrc, cvtDst, params.winSize, params.sigmaX, params.sigmaY);
				break;
			case EBM_BLUR:
				cv::blur(cvtSrc, cvtDst, params.winSize);
				break;
			case EBM_MEDIAN:
				cv::medianBlur(cvtSrc, cvtDst, params.winSize.width);
				break;
			case EBM_BILATERAL:
				cv::bilateralFilter(cvtSrc, cvtDst, params.diameter, params.sigmaX, params.sigmaY);
				break;
			default:
				cv::GaussianBlur(cvtSrc, cvtDst, params.winSize, params.sigmaX, params.sigmaY);
			}
			if (params.method != EBM_BILATERAL)
			{
				cvtDst = cvtDst * (1. / srcFactor);
				if (params.isConvertToUint)
				{
					cvtDst.convertTo(dst, src.type(), 1., 0.5);
				}
				else
				{
					cvtDst.copyTo(dst);
				}
			}
			else
			{
				cvtDst.copyTo(dst);
			}
		}
		else
		{
			rv = ER_INVALID_DIMS;
		}
	}
	else
	{
		rv = ER_INVALID_INPUT;
	}

	return rv;
}


/****************************************************************
*
*			niblack - public
*
*****************************************************************/
EReport CPreprocessIm::niblack(cv::Mat &src, cv::Mat & outIm, cv::Mat &outNiBl, SNiblackParameters params)
{
	EReport rv(ER_SUCCESS);

	if (src.data)
	{
		if ((src.rows > 0) && (src.cols > 0) && (src.channels() == 1))
		{
			cv::Mat img, avgImg, sqrImg, devImg, powImg;
			SBlurParameters blurParams;
			blurParams.winSize = params.winSize;
			blurParams.method = EBM_BLUR;
			src.copyTo(img);
			img.convertTo(img, CV_32F);
			blurImg(img, avgImg, blurParams);
			cv::multiply(img, img, powImg);
			blurImg(powImg, sqrImg, blurParams);
			cv::pow(avgImg, 2.0, powImg);
			cv::pow(sqrImg - powImg, 0.5, devImg);
			outNiBl = cv::Mat::zeros(src.rows, src.cols, img.type());
			outNiBl.setTo(1, img > (params.kapa*devImg - params.offset));
			outIm = 50. - avgImg + img;
		}
		else
		{
			rv = ER_INVALID_DIMS;
		}
	}
	else
	{
		rv = ER_INVALID_INPUT;
	}
	return rv;
}


/****************************************************************
*
*			illumEnhance - public
*
*****************************************************************/
EReport CPreprocessIm::illumEnhance(cv::Mat &src, cv::Mat &dst)
{
	EReport rv(ER_SUCCESS);

	if (src.data)
	{
		if ((src.rows > 0) && (src.cols > 0) && (src.channels() == 1))
		{
			cv::Mat tmpMat;
			if (m_isNormalizeClahe)
			{
				double m(0.), M(0.);
				cv::minMaxLoc(src, &m, &M);
				if (m != M) // if min is not max
				{
					tmpMat = (src - m) / (M - m);
				}
				else
				{
					src.copyTo(tmpMat);
				}
				if (m_convert2type == CV_8U)
				{
					tmpMat = 255 * tmpMat;
					tmpMat.convertTo(tmpMat, CV_8U);
				}
				else if (m_convert2type == CV_16U)
				{
					tmpMat = 65535 * tmpMat;
					tmpMat.convertTo(tmpMat, CV_16U);
				}
			}
			else
			{
				src.convertTo(tmpMat, m_srcType, 1., 0.5);
			}
			m_clahe->apply(tmpMat, dst);
		}
		else
		{
			rv = ER_INVALID_DIMS;
		}
	}
	else
	{
		rv = ER_INVALID_INPUT;
	}
	return rv;
}

CPreprocessIm::CPreprocessIm()
{
	m_isInit = false;
	m_isBlur = false;
	m_isNiBlack = true;
	m_isMask = true;
	m_isNormalizeClahe = false;
	m_srcType = 0;
	m_convert2type = CV_8U;
	m_clipLimit = 3.5;
	m_tileGridSize = cv::Size(4,4);
	m_clahe = cv::createCLAHE(m_clipLimit, m_tileGridSize);
	
	m_imProcThresh = {};
	m_blurParams = {};
	m_niblackParams = {};
}
CPreprocessIm::~CPreprocessIm()
{

}