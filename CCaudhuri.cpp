#include "CChaudhuri.h"
/*///////////////////////////////////////////////////////////////
////////  Public methods implementation ////////////////////////
///////////////////////////////////////////////////////////////*/

CChaudhuri& CChaudhuri::getReference()
{
	static CChaudhuri m_CChaudhuri;
	return m_CChaudhuri;
}


/****************************************************************
*
*			init - public
*
*****************************************************************/
EReport CChaudhuri::init(std::string &filename)
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
				if (retName == "isGPUOptimization")
				{
					m_isGPUOptimization = retBool;
				}
				else if (retName == "L")
				{
					m_chaudhuriParams.L = retFloat;
				}
				else if (retName == "T")
				{
					m_chaudhuriParams.T = retFloat;
				}
				else if (retName == "sigma")
				{
					m_chaudhuriParams.sigma = retFloat;
				}
				else if (retName == "numOfKerRotations")
				{
					m_chaudhuriParams.numOfKerRotations = retFloat;
				}
				else if (retName == "isNormalize")
				{
					m_chaudhuriParams.isNormalize = retBool;
				}
				else if (retName == "isDisplayProgress")
				{
					m_chaudhuriParams.isDisplayProgress = retBool;
				}
				else if (retName == "isConvert")
				{
					m_chaudhuriParams.isConvert = retBool;
				}
				else if (retName == "imDepth")
				{
					if (retString == "CV_16U")
					{
						m_chaudhuriParams.imDepth = CV_16U;
					}
					else
					{
						m_chaudhuriParams.imDepth = CV_8U;
					}
				}
				else if (retName == "dispCutMin")
				{
					m_chaudhuriParams.dispCutMin = retFloat;
				}
				else if (retName == "dispCutMax")
				{
					m_chaudhuriParams.dispCutMax = retFloat;
				}
				else if (retName == "isCalculateLsigma")
				{
					m_isCalculateLsigma = retBool;
				}
				else if (retName == "imRows")
				{
					m_chaudhuriParams.imRows = retFloat;
				}
				else if (retName == "imCols")
				{
					m_chaudhuriParams.imCols = retFloat;
				}
				else if (retName == "cudaConvOptim")
				{
                    m_chaudhuriParams.cudaOptimRows = retRect.x;
                    m_chaudhuriParams.cudaOptimCols = retRect.y;
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
		m_isInitParams = true;
		if (m_isCalculateLsigma)
		{
			m_chaudhuriParams.sigma = round(0.001*static_cast<double>(m_chaudhuriParams.imRows + m_chaudhuriParams.imCols) / 2. - 0.1, 1);
			m_chaudhuriParams.L = round(0.0052*static_cast<double>(m_chaudhuriParams.imRows + m_chaudhuriParams.imCols) / 2. + 0.34, 0);
		}
		er = constructKernels();
#if CUDA_OPTIM
		cv::cuda::GpuMat tmp;
		cv::Ptr<cv::cuda::Filter> tmpFilt;
		for (int i(0); i < m_chadhuriKers.size() ; i++)
		{
            cm_chadhuriKers.push_back(tmp);
            //cm_filter2D.push_back(tmpFilt);
            cm_chadhuriKers[i].upload(m_chadhuriKers[i]);
            //cm_filter2D[i] = cv::cuda::createLinearFilter(CV_32F, -1, m_chadhuriKers[i]); // tested works slower than CPU(???)
		}
		cm_convolver = cv::cuda::createConvolution(cv::Size(m_chaudhuriParams.cudaOptimRows, m_chaudhuriParams.cudaOptimCols)); // when set to size == 0 cv will optimize parameter
#endif // CUDA_OPTIM
	}
	if (ER_SUCCESS == er)
	{
		m_isInit = true;
	}
	return er;
}


/****************************************************************
*
*			calculateChaudhuri - public
*
*****************************************************************/
#if !CUDA_OPTIM
EReport CChaudhuri::calculateChaudhuri(cv::Mat &src, cv::Mat &dst)
{
	EReport rv(ER_SUCCESS);
	if (m_isInit)
	{
		if ((src.rows > 0) && (src.cols > 0))
		{
			cv::Mat gray, conv;
			if (src.channels() > 1)
			{
				cv::Mat bgr[3];
				cv::split(src, bgr);
				bgr[1].copyTo(gray);
				// should enter here warning at log in debug mode
			}
			else
			{
				src.copyTo(gray);
			}
			if ((gray.type() == CV_8U) || (gray.type() == CV_16U))
			{
				gray.convertTo(gray, CV_32F);
			}
			dst = cv::Mat::zeros(gray.rows, gray.cols, gray.type());
			conv = cv::Mat::zeros(gray.rows, gray.cols, gray.type());
			for (int i(0); i < m_chadhuriKers.size(); i++)
			{
				cv::filter2D(gray, conv, -1, m_chadhuriKers[i]);
				dst = cv::max(conv, dst);
				if (m_chaudhuriParams.isDisplayProgress)
				{
					std::cout << "Filter itteration No: " << i << std::endl;
				}
			}
			if (m_chaudhuriParams.isNormalize)
			{
				normalizeIm(dst, dst, m_chaudhuriParams);
			}
		}
		else
		{
			rv = ER_INVALID_DIMS;
		}
	}
	else
	{
		rv = ER_INVALID_INIT;
	}

	return rv;
}
#endif
#if CUDA_OPTIM

/****************************************************************
*
*			calculateChaudhuri - public - cuda overload
*
*****************************************************************/
EReport CChaudhuri::calculateChaudhuri(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst)
{
	EReport rv(ER_SUCCESS);
	if (m_isInit)
	{
		if ((src.rows > 0) && (src.cols > 0))
		{
			if (src.channels() > 1)
			{
				cv::cuda::split(src, cm_bgr);
				cm_bgr[1].copyTo(cm_gray);
				// should enter here warning at log in debug mode
			}
			else
			{
				src.copyTo(cm_gray);
			}
			if ((cm_gray.type() == CV_8U) || (cm_gray.type() == CV_16U))
			{
				cm_gray.convertTo(cm_gray, CV_32F);
			}
            cv::cuda::GpuMat conv;
            int top(0), left(0);
			dst = cv::cuda::GpuMat(cm_gray.rows, cm_gray.cols, cm_gray.type());
			dst.setTo(cv::Scalar::all(0));
			cm_conv = cv::cuda::GpuMat(cm_gray.rows, cm_gray.cols, cm_gray.type());
			cm_conv.setTo(cv::Scalar::all(0));
            top = static_cast<int>(cm_chadhuriKers[0].rows/2.) - 1;
            left = static_cast<int>(cm_chadhuriKers[0].cols/2.) - 1;
			for (int i(0); i < cm_chadhuriKers.size(); i++)
			{
				cm_convolver->convolve(cm_gray, cm_chadhuriKers[i], conv);
				//cm_filter2D[i]->apply(cm_gray, cm_conv); // skipped since slower than CPU
				conv.copyTo(cm_conv(cv::Rect(top,left, conv.cols, conv.rows))); // sinse cuda optim cut boundaries
				cv::cuda::max(cm_conv, dst, dst);
				if (m_chaudhuriParams.isDisplayProgress)
				{
					std::cout << "Filter itteration No: " << i << std::endl;
				}
			}
			if (m_chaudhuriParams.isNormalize)
			{
				normalizeIm(dst, dst, m_chaudhuriParams);
			}
		}
		else
		{
			rv = ER_INVALID_DIMS;
		}
	}
	else
	{
		rv = ER_INVALID_INIT;
	}

	return rv;
}

#endif // CUDA_OPTIM


/****************************************************************
*
*			set of service methods set\get - public
*
*****************************************************************/
void CChaudhuri::getKernels(std::vector<cv::Mat> &kerList)
{
	kerList = m_chadhuriKers;
}

void CChaudhuri::setMask(cv::Mat &mask)
{
	mask.copyTo(m_chaudhuriParams.mask);
}

#if CUDA_OPTIM
void CChaudhuri::setCudaMask(cv::cuda::GpuMat &mask)
{
    mask.copyTo(cm_mask);
}
#endif
/*///////////////////////////////////////////////////////////////
////////  Private methods implementation ////////////////////////
///////////////////////////////////////////////////////////////*/


/****************************************************************
*
*			constructKernels - private
*
*****************************************************************/
EReport CChaudhuri::constructKernels()
{
	EReport rv(ER_SUCCESS);
	if (m_isInitParams)
	{
		if ((m_chaudhuriParams.L > 0) && (m_chaudhuriParams.T > 0) && (m_chaudhuriParams.sigma > 0))
		{
			if (m_chadhuriKers.size() > 0)
			{
				m_chadhuriKers = {};
			}
			cv::Mat X, Y; // matrices that holds meshgrid
			cv::Mat U, V; // matrices holds partial calculation
			cv::Mat ker, mask;
			cv::Mat tmpKer;
			int m(0);	// m holds maximal boundary limits (will be used in range (-m,m))
			double theta(0.0);
			cv::Scalar kerMaskMean(0.0);
			cv::Scalar numEntries(0.0);
			// call meshgrid
			m = static_cast<int>(std::max(std::ceil((m_chaudhuriParams.sigma*3.0 + 5.0) / 2.0), (m_chaudhuriParams.L + 6.) / 2.));
			rv = getMeshgrid2D(cv::Range(-m, m), cv::Range(-m, m), X, Y);
			Y.convertTo(Y, CV_32F);
			X.convertTo(X, CV_32F);
			// prepare angles
			if (ER_SUCCESS == rv)
			{
				rv = linspace(0., 180. - (180. / m_chaudhuriParams.numOfKerRotations), m_chaudhuriParams.numOfKerRotations, m_rotationAngles);
			}
			// iterate over angles
			if (ER_SUCCESS == rv)
			{
				for (int i(0); i < m_rotationAngles.size(); i++)
				{
					// call calculateKernel
					theta = m_rotationAngles[i] * CV_PI / 180.0;
					U = X*std::cos(theta) - Y*std::sin(theta);
					V = Y*std::cos(theta) + X*std::sin(theta);
					cv::multiply(-U, U, tmpKer);
					cv::exp((tmpKer / (2 * std::pow(m_chaudhuriParams.sigma, 2))), ker);
					ker = 0.5 - ker;
					mask = ((cv::abs(U) <= m_chaudhuriParams.sigma*3) & (cv::abs(V) < m_chaudhuriParams.L / 2.)) / 255.;
					mask.convertTo(mask, CV_32F);
					cv::multiply(ker, mask, tmpKer);
					numEntries = cv::sum(mask);
					kerMaskMean = cv::sum(tmpKer) / numEntries;
					tmpKer = tmpKer - kerMaskMean;
					cv::multiply(tmpKer, mask, ker);
					// append kernels to kernels vector - we must use deep copy
					m_chadhuriKers.push_back(cv::Mat::zeros(ker.rows, ker.cols, ker.type()));
					ker.copyTo(m_chadhuriKers[i]);
				}
			}
		}
		else
		{
			rv = ER_INVALID_PARAM;
		}
	}
	else
	{
		rv = ER_INVALID_INIT;
	}

	return rv;
}

/****************************************************************
*
*			getMeshgrid2D - private
*
*****************************************************************/
EReport CChaudhuri::getMeshgrid2D(const cv::Range &xRange,
	const cv::Range &yRange,
	cv::Mat &X,
	cv::Mat &Y)
{
	EReport rv(ER_SUCCESS);
	if ((xRange.size() > 0) && (yRange.size() > 0))
	{
		cv::Mat xGrid, yGrid;
		std::vector<double> vecX, vecY;
		for (int i(xRange.start); i <= xRange.end; i++)
		{
			vecX.push_back(static_cast<double>(i));
		}
		for (int i(yRange.start); i <= yRange.end; i++)
		{
			vecY.push_back(static_cast<double>(i));
		}
		xGrid = cv::Mat(vecX);
		yGrid = cv::Mat(vecY);
		cv::repeat(xGrid.reshape(1, 1), yGrid.total(), 1, X);
		cv::repeat(yGrid.reshape(1, 1).t(), 1, xGrid.total(), Y);
	}
	else
	{
		rv = ER_INVALID_PARAM;
	}
	return rv;
}

#if !CUDA_OPTIM
/****************************************************************
*
*			normalizeIm - private
*
*****************************************************************/
EReport CChaudhuri::normalizeIm(cv::Mat &src, cv::Mat &dst, SChaudhuriParams &params)
{
	EReport rv(ER_SUCCESS);
	if (src.data)
	{
		if ((src.rows > 0) && (src.cols > 0))
		{
			cv::Mat srcCopy, normIm;
			double imMin(0.), imMax(0.);
			src.copyTo(srcCopy);
			// if mask is not empty and dimentions equivalent to src apply mask
			if (params.mask.data)
			{
				if ((params.mask.rows == srcCopy.rows) && (params.mask.cols == srcCopy.cols))
				{
					cv::multiply(srcCopy, params.mask, srcCopy);
				}
			}
			cv::minMaxLoc(srcCopy, &imMin, &imMax);
			if (imMin != imMax)
			{
				normIm = (srcCopy - imMin) / (imMax - imMin);
				normIm.setTo(params.dispCutMax, normIm > params.dispCutMax);
				normIm.setTo(params.dispCutMin, normIm < params.dispCutMin);
				if (params.isConvert)
				{
					if (params.imDepth == CV_8U)
					{
						normIm = 255. * normIm;
					}
					else if (params.imDepth == CV_16U)
					{
						normIm = 65535. * normIm;
					}
					else //default CV_8U
					{
						normIm = 255. * normIm;
					}
					normIm.convertTo(dst, params.imDepth);
				}
			}
			else
			{
				srcCopy.copyTo(dst);
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
#endif

#if CUDA_OPTIM

/****************************************************************
*
*			normalizeIm - private - cuda overload
*
*****************************************************************/
EReport CChaudhuri::normalizeIm(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, SChaudhuriParams &params)
{
	EReport rv(ER_SUCCESS);
	if (src.data)
	{
		if ((src.rows > 0) && (src.cols > 0))
		{
			double imMin(0.), imMax(0.);
			src.copyTo(cm_srcCopy);
			// if mask is not empty and dimentions equivalent to src apply mask
			if (params.mask.data)
			{
				if ((cm_mask.rows == cm_srcCopy.rows) && (cm_mask.cols == cm_srcCopy.cols))
				{
					cv::cuda::multiply(cm_srcCopy, cm_mask, cm_srcCopy);
				}
			}
			cv::cuda::minMax(cm_srcCopy, &imMin, &imMax);
			if (imMin != imMax)
			{
                cv::cuda::GpuMat cm_normIm(cm_srcCopy.rows, cm_srcCopy.cols, cm_srcCopy.type());
                cv::cuda::subtract(cm_srcCopy, imMin, cm_normIm);
                cv::cuda::multiply(cm_normIm, 1./(imMax - imMin), cm_normIm);
				rv = createNormMask(cm_normIm, cm_normMask, params.dispCutMax, ">");
				cm_normIm.setTo(cv::Scalar(params.dispCutMax), cm_normMask);
				rv = createNormMask(cm_normIm, cm_normMask, params.dispCutMax, "<");
				cm_normIm.setTo(cv::Scalar(params.dispCutMax), cm_normMask);
				if (params.isConvert)
				{
					if (params.imDepth == CV_8U)
					{
						cv::cuda::multiply(cm_normIm, 255. , cm_normIm);
					}
					else if (params.imDepth == CV_16U)
					{
						cv::cuda::multiply(cm_normIm, 65535. , cm_normIm);
					}
					else //default CV_8U
					{
						cv::cuda::multiply(cm_normIm, 255. , cm_normIm);
					}
					cm_normIm.convertTo(dst, params.imDepth);
				}
			}
			else
			{
				cm_srcCopy.copyTo(dst);
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
*			createNormMask - private - cuda optimization
*
*****************************************************************/
EReport CChaudhuri::createNormMask(cv::cuda::GpuMat &src, cv::cuda::GpuMat &normMask, double &compareVal, const std::string &oper)
{
    EReport rv(ER_SUCCESS);
    if (src.data)
	{
		if ((src.rows > 0) && (src.cols > 0))
		{
            normMask = cv::cuda::GpuMat(src.rows, src.cols, src.type());
            if (oper.size() > 0)
            {
                if (oper == ">")
                {
                    cv::cuda::compare(src, cv::Scalar(compareVal), normMask, cv::CMP_GT);
                }
                else if (oper == ">=")
                {
                    cv::cuda::compare(src, cv::Scalar(compareVal), normMask, cv::CMP_GE);
                }
                else if (oper == "==")
                {
                    cv::cuda::compare(src, cv::Scalar(compareVal), normMask, cv::CMP_EQ);
                }
                else if (oper == "<=")
                {
                    cv::cuda::compare(src, cv::Scalar(compareVal), normMask, cv::CMP_LE);
                }
                else if (oper == "<")
                {
                    cv::cuda::compare(src, cv::Scalar(compareVal), normMask, cv::CMP_LT);
                }
                else if (oper == "!=")
                {
                    cv::cuda::compare(src, cv::Scalar(compareVal), normMask, cv::CMP_NE);
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

#endif // CUDA_OPTIM

/****************************************************************
*
*			linspace - private
*
*****************************************************************/
template<typename T>
EReport CChaudhuri::linspace(T start_in, T end_in, int num_in, std::vector<double> &linspaced)
{

	EReport rv(ER_SUCCESS);
	double start = static_cast<double>(start_in);
	double end = static_cast<double>(end_in);
	double num = static_cast<double>(num_in);

	if (num > 1)
	{
		double delta = (end - start) / (num - 1);

		for (int i = 0; i < num - 1; ++i)
		{
			linspaced.push_back(start + delta * i);
		}
		linspaced.push_back(end); // I want to ensure that start and end
								  // are exactly the same as the input
	}
	else if (num == 1)
	{
		linspaced.push_back(start);
		rv = ER_INVALID_DIMS;
	}
	else if (num <= 0)
	{
		rv = ER_INVALID_PARAM;
	}

	return rv;
}


CChaudhuri::CChaudhuri()
{
	m_isInit = false;				// indicates if data is read properly from *.ini file
	m_isGPUOptimization = false;	// if true we'll use overload methods
	m_isInitParams = false;			// indicates while init process if needed parameters read properly
	m_isCalculateLsigma = false;	// if true automatic calculation of L & sigma is done relative to image size
#if CUDA_OPTIM
	cm_convolver = cv::cuda::createConvolution(cv::Size(m_chaudhuriParams.cudaOptimRows, m_chaudhuriParams.cudaOptimCols)); // when set to size == 0 cv will optimize parameter
#endif // CUDA_OPTIM
	m_chaudhuriParams = {};
	m_chadhuriKers = {};
}

CChaudhuri::~CChaudhuri()
{

}
