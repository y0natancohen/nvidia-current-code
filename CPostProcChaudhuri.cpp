#include "CPostProcChaudhuri.h"

/*///////////////////////////////////////////////////////////////
////////  Public methods implementation ////////////////////////
///////////////////////////////////////////////////////////////*/

CPostProcChaudhuri& CPostProcChaudhuri::getReference()
{
	static CPostProcChaudhuri m_CPostProcChaudhuri;
	return m_CPostProcChaudhuri;
}


/****************************************************************
*
*			init - public
*
*****************************************************************/
EReport CPostProcChaudhuri::init(std::string &filename)
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
				if (retName == "morphKerLength")
				{
					m_postProcParams.morphKerLength = retFloat;
				}
				else if (retName == "numMorphKerRotations")
				{
					m_postProcParams.numMorphKerRotations = retFloat;
				}
				else if (retName == "isDisplayMorphProgress")
				{
					m_postProcParams.isDisplayProgress = retBool;
				}
				else if (retName == "PPdisplatCutMin")
				{
					m_postProcParams.dispCutMin = retFloat;
				}
				else if (retName == "PPdisplayCutMax")
				{
					m_postProcParams.dispCutMax = retFloat;
				}
				else if (retName == "PPisConvert")
				{
					m_postProcParams.isConvert = retBool;
				}
				else if (retName == "PPimDepth")
				{
					if (retString == "CV_16U")
					{
						m_postProcParams.imDepth = CV_16U;
					}
					else
					{
						m_postProcParams.imDepth = CV_8U;
					}
				}
				else if (retName == "PPisNormalize")
				{
					m_postProcParams.isNormalize = retBool;
				}
				else if (retName == "PPisDilate")
				{
					m_postProcParams.isDilate = retBool;
				}
				else if (retName == "PPcirclePatternParams")
				{
					m_postProcParams.circleCenter.x = retRect.x;
					m_postProcParams.circleCenter.y = retRect.y;
					m_postProcParams.circlRadiusDilate = retRect.width;
					m_postProcParams.circlRadiusClose = retRect.height;
				}
				else if (retName == "PPadaptiveLowBlockSize")
				{
					m_postProcParams.adaptiveLowBlockSize = static_cast<int>(retFloat);
				}
				else if (retName == "PPadaptiveHighBlockSize")
				{
					m_postProcParams.adaptiveHighBlockSize = static_cast<int>(retFloat);
				}
				else if (retName == "PPadaptiveMeanShift")
				{
					m_postProcParams.adaptiveMeanShift = static_cast<int>(retFloat);
				}
				else if (retName == "PPadaptiveMaxValue")
				{
					m_postProcParams.adaptiveMaxValue = static_cast<int>(retFloat);
				}
				else if (retName == "PPisAutoBrightness")
				{
					m_postProcParams.isAutoBrightness = retBool;
				}
				else if (retName == "PPstartCutValue")
				{
					m_postProcParams.startCutValue = retFloat;
				}
				else if (retName == "PPthreshCutValue")
				{
					m_postProcParams.threshCutValue = retFloat;
				}
				else if (retName == "PPnumMorphItter")
				{
					m_postProcParams.numMorphItter = static_cast<int>(retFloat);
				}
				else if (retName == "PPisCvDisplay")
				{
					m_postProcParams.isCvDisplay = retBool;
				}
				else if (retName == "PPisUseDispCut")
				{
					m_postProcParams.isUseDispCut = retBool;
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
		m_isParamInit = true;
		er = constructKernels();
#if CUDA_OPTIM
        cv::cuda::GpuMat tmp;
        for (int i(0); i < m_morphKernels.size(); i++)
        {
            cm_morphKernels.push_back(tmp);
            cm_morphKernels[i].upload(m_morphKernels[i]);
        }
#endif // CUDA_OPTIM
	}
	if (ER_SUCCESS == er)
	{
		er = createCirclePattern();
#if CUDA_OPTIM
        if (ER_SUCCESS == er)
        {
            cm_circleKerClose.upload(m_circleKerClose);
            cm_circleKerDilate.upload(m_circleKerDilate);
        }
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
*			postProcIm - public
*
*****************************************************************/
EReport CPostProcChaudhuri::postProcIm(cv::Mat &src, cv::Mat &dst)
{
	EReport rv(ER_SUCCESS);

	if (m_isInit)
	{
		if ((src.rows > 0) && (src.cols > 0))
		{
			double dispCut(0.0);
			cv::Mat srcCopy, morphMat, threshRes, finalMask;
			src.copyTo(srcCopy);
			for (int i(0); i < m_postProcParams.numMorphItter; i++)
			{
				rv = morphChaudhuri(srcCopy, morphMat);
				morphMat.copyTo(srcCopy);
			}
			if (ER_SUCCESS == rv)
			{
				rv = normalizeIm(morphMat, morphMat, m_postProcParams);
			}
			if (ER_SUCCESS == rv)
			{
				rv = adaptiveThresholding(morphMat, threshRes);
			}
			if (ER_SUCCESS == rv)
			{
				if (m_postProcParams.isDisplayProgress)
				{
					std::cout << "Start connected components" << std::endl;
				}
				rv = connectingComponents(threshRes, finalMask);
				if (m_postProcParams.isDisplayProgress)
				{
					std::cout << "End connected components" << std::endl;
				}
			}
			if (ER_SUCCESS == rv)
			{
				finalMask.convertTo(finalMask, CV_32F);
				cv::multiply(morphMat, finalMask, dst);
				if (m_postProcParams.isAutoBrightness)
				{
					rv = setAutoBrightnessLevel(dst, dispCut);
				}
			}
			if (ER_SUCCESS == rv)
			{
				if (m_postProcParams.isCvDisplay)
				{
					if (m_postProcParams.isUseDispCut) // convert to uint8 for cv display
					{
						rv = normalizeIm(dst, dst, 0.0, dispCut);
					}
					else
					{
						rv = normalizeIm(dst, dst, 0.0, 1.0);
					}
					dst = 255 * dst;
					dst.convertTo(dst, CV_8U);
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
		rv = ER_INVALID_INIT;
	}

	return rv;
}

/****************************************************************
*
*			general service methods - public
*
*****************************************************************/
void CPostProcChaudhuri::getKernels(std::vector<cv::Mat> &kernels)
{
	kernels = m_morphKernels;
}

/*///////////////////////////////////////////////////////////////
////////  Private methods implementation ////////////////////////
///////////////////////////////////////////////////////////////*/
#if CUDA_OPTIM
/****************************************************************
*
*			morphChaudhuri - private
*
*****************************************************************/
EReport CPostProcChaudhuri::morphChaudhuri(cv::Mat &src, cv::Mat &dst)
{
	EReport rv(ER_SUCCESS);

	if (m_isInit)
	{
		if ((src.rows > 0) && (src.cols > 0))
		{
			cv::Mat morphMat;
			dst = cv::Mat::zeros(src.rows, src.cols, src.type());
			for (int i(0); i < m_morphKernels.size(); i++)
			{
				cv::morphologyEx(src, morphMat, cv::MORPH_ERODE, m_morphKernels[i]);
				cv::max(morphMat, dst, dst);
				if (m_postProcParams.isDisplayProgress)
				{
					std::cout << "Erode itteration No.: " << i << std::endl;
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
		rv = ER_INVALID_INIT;
	}
	return rv;
}
#endif

#if !CUDA_OPTIM
/****************************************************************
*
*			morphChaudhuri - private - cuda optimization overload
*
*****************************************************************/
EReport CPostProcChaudhuri::morphChaudhuri(cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst)
{
	EReport rv(ER_SUCCESS);

	if (m_isInit)
	{
		if ((src.rows > 0) && (src.cols > 0))
		{
			cv::Mat morphMat;
			dst = cv::Mat::zeros(src.rows, src.cols, src.type());
			for (int i(0); i < m_morphKernels.size(); i++)
			{
				cv::morphologyEx(src, morphMat, cv::MORPH_ERODE, m_morphKernels[i]);
				cv::max(morphMat, dst, dst);
				if (m_postProcParams.isDisplayProgress)
				{
					std::cout << "Erode itteration No.: " << i << std::endl;
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
		rv = ER_INVALID_INIT;
	}
	return rv;
}

#endif // CUDA_OPTIM

/****************************************************************
*
*			adaptiveThresholding - private
*
*****************************************************************/
EReport CPostProcChaudhuri::adaptiveThresholding(cv::Mat &src, cv::Mat &dst)
{
	EReport rv(ER_SUCCESS);

	if (m_isInit)
	{
		if ((src.rows > 0) && (src.cols > 0))
		{
			cv::Mat srcCopy,lowThresh, highThresh, finalThresh;
			src.copyTo(srcCopy);
			if (m_postProcParams.isDilate)
			{
				cv::morphologyEx(srcCopy, srcCopy, cv::MORPH_DILATE, m_circleKerDilate);
			}
			srcCopy.convertTo(srcCopy, CV_8U);
			cv::adaptiveThreshold(srcCopy, lowThresh, m_postProcParams.adaptiveMaxValue, cv::ADAPTIVE_THRESH_MEAN_C,
				cv::THRESH_BINARY, m_postProcParams.adaptiveLowBlockSize, m_postProcParams.adaptiveMeanShift);
			cv::adaptiveThreshold(srcCopy, highThresh, m_postProcParams.adaptiveMaxValue, cv::ADAPTIVE_THRESH_MEAN_C,
				cv::THRESH_BINARY, m_postProcParams.adaptiveHighBlockSize, m_postProcParams.adaptiveMeanShift);
			cv::bitwise_or(lowThresh, highThresh, finalThresh);
			cv::morphologyEx(finalThresh, dst, cv::MORPH_CLOSE, m_circleKerClose);
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


/****************************************************************
*
*			connectingComponents - private
*
*****************************************************************/
EReport CPostProcChaudhuri::connectingComponents(cv::Mat &src, cv::Mat &dst)
{
	EReport rv(ER_SUCCESS);

	if (m_isInit)
	{
		if ((src.rows > 0) && (src.cols > 0))
		{
			cv::Mat labels, stats, centroids, finMask;
			int nLabels = cv::connectedComponentsWithStats(src, labels, stats, centroids);
			labels.copyTo(finMask);
			for (int label(0); label < nLabels; label++)
			{
				if (stats.at<int>(label, cv::CC_STAT_AREA) < 500)
				{
					finMask.setTo(0, labels == label);
				}
				else
				{
					finMask.setTo(1, labels == label);
				}
			}
			finMask.copyTo(dst);
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

/****************************************************************
*
*			constructKernels - private
*
*****************************************************************/
EReport CPostProcChaudhuri::constructKernels()
{
	EReport rv(ER_SUCCESS);

	if (m_isParamInit)
	{
		std::vector<double> rotationAngles;
		rv = linspace(0., 180. - (180. / m_postProcParams.numMorphKerRotations),
			m_postProcParams.numMorphKerRotations, rotationAngles);
		if (ER_SUCCESS == rv)
		{
			for (int i(0); i < rotationAngles.size(); i++)
			{
				cv::Mat tmp, tmpUINT;
				rv = strel2D(m_postProcParams.morphKerLength, rotationAngles[i], tmp);
				tmp.convertTo(tmpUINT, CV_8U);
				m_morphKernels.push_back(tmpUINT);
			}
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
*			strel2D - private
*
*****************************************************************/
EReport CPostProcChaudhuri::strel2D(int length, double angle, cv::Mat &linKer)
{
	EReport rv(ER_SUCCESS);

	if (length > 2)
	{
		double deg90(0.0), alpha(0.0);
		double r(0.0), c(0.0); // r for rows, c for cols
		std::vector<int> rows, cols;
		double ray(0.0);
		double tanAlpha(0.0);
		rv = modulu(angle, 180., deg90);
		double sect = floor(deg90 / 45);
		cv::Mat line, linestrip, linerest, z, invLinerest, invLinestrip;
		cv::Mat subA, subB, subC, res;

		rv = modulu(angle, 90, deg90);
		if (deg90 > 45.)
		{
			alpha = CV_PI*(90 - deg90) / 180.;
		}
		else
		{
			alpha = CV_PI * deg90 / 180.;
		}
		ray = (length - 1.) / 2.;
		//We are interested only in the discrete rectangle which contains the diameter
		//However we focus our attention to the bottom left quarter of the circle,
		//because of the central symmetry.
		c = int(round(ray * cos(alpha)) + 1);
		r = int(round(ray * sin(alpha)) + 1);
		line = cv::Mat::zeros(r, c, CV_32F);
		tanAlpha = tan(alpha);
		for (int i(1); i < c + 1; i++)
		{
			cols.push_back(i);
			rows.push_back(float(r) - floor(tanAlpha * (cols[i - 1] - 0.5)));
			line.at<float>(rows[i - 1] - 1, cols[i - 1] - 1) = 1;
		}
		if ((r == 1) || (c == 1))
		{
			linestrip = line(cv::Range(0, 1), cv::Range(0, c - 1));
			cv::flip(linestrip, invLinestrip, 1); // flip linestrip horisontally
			cv::hconcat(linestrip, cv::Mat::ones(1, 1, linestrip.type()), subB);
			cv::hconcat(subB, invLinestrip, res);
		}
		else
		{
			linestrip = line(cv::Range(0, 1), cv::Range(0, c - 1));
			linerest = line(cv::Range(1, r), cv::Range(0, c - 1));
			z = cv::Mat::zeros(r - 1, c, CV_32F);
			cv::flip(linestrip, invLinestrip, 1); // flip linestrip horisontally
			cv::flip(linerest, invLinerest, 1);  // flip linerest horisontally
			cv::flip(invLinerest, invLinerest, 0); //flip unvlinerest vertically
			cv::hconcat(z, invLinerest, subA);
			cv::hconcat(linestrip, cv::Mat::ones(1, 1, linestrip.type()), subB);
			cv::hconcat(subB, invLinestrip, subB);
			cv::hconcat(linerest, z, subC);
			cv::vconcat(subA, subB, res);
			cv::vconcat(res, subC, res);
		}
		if (sect == 0)
		{
			res.copyTo(linKer);
		}
		else if (sect == 1) // second quarter
		{
			cv::transpose(res, linKer);
		}
		else if (sect == 2)
		{
			cv::transpose(res, res);
			cv::flip(res, linKer, 0);
		}
		else if (sect == 3)
		{
			cv::flip(res, linKer, 1);
		}
	}
	else
	{
		rv = ER_INVALID_DIMS;
	}
	return rv;
}


/****************************************************************
*
*			createCircle - private
*
*****************************************************************/
EReport CPostProcChaudhuri::createCircle(cv::Mat &retPattern, cv::Point center, int radius)
{
	EReport rv(ER_SUCCESS);

	if (radius > 0)
	{
		retPattern = cv::Mat::zeros(int(radius * 2. + 1), int(radius * 2. + 1), CV_8U);
		for (int i(center.x - radius); i < center.x + radius + 1; i++)
		{
			for (int j(center.y - radius); j < center.y + radius + 1; j++)
			{
				if ((std::pow(center.x + i, 2) + std::pow(center.y + j, 2)) < std::pow(radius, 2))
				{
					retPattern.at<char>(static_cast<int>( center.x + i + radius),
						static_cast<int>(center.y + j + radius)) = 1;
				}
			}
		}
	}
	else
	{
		rv = ER_INVALID_PARAM;
	}

	return rv;
}


/****************************************************************
*
*			createCirclePattern - private
*
*****************************************************************/
EReport CPostProcChaudhuri::createCirclePattern()
{
	EReport rv(ER_SUCCESS);

	if (m_isParamInit)
	{
		createCircle(m_circleKerDilate, m_postProcParams.circleCenter, m_postProcParams.circlRadiusDilate);
		createCircle(m_circleKerClose, m_postProcParams.circleCenter, m_postProcParams.circlRadiusClose);
	}
	else
	{
		rv = ER_INVALID_INIT;
	}
	return rv;
}


/****************************************************************
*
*			modulu - private
*
*****************************************************************/
EReport CPostProcChaudhuri::modulu(double x, double y, double &mod)
{
	EReport rv(ER_SUCCESS);

	if (y == 0)
	{
		mod = x;
	}
	else
	{
		double div = floor(x / y);
		double m = div * y;
		mod = x - m;
	}

	return rv;
}

/****************************************************************
*
*			linspace - private
*
*****************************************************************/
template<typename T>
EReport CPostProcChaudhuri::linspace(T start_in, T end_in, int num_in, std::vector<double> &linspaced)
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


/****************************************************************
*
*			normalizeIm - private
*
*****************************************************************/
EReport CPostProcChaudhuri::normalizeIm(cv::Mat &src, cv::Mat &dst, SPostProcParams params)
{
	EReport rv(ER_SUCCESS);
	if (src.data)
	{
		if ((src.rows > 0) && (src.cols > 0))
		{
			cv::Mat srcCopy, normIm;
			double imMin(0.), imMax(0.);
			src.copyTo(srcCopy);
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
				else
				{
					normIm.copyTo(dst);
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


/****************************************************************
*
*			normalizeIm - private
*
*****************************************************************/
EReport CPostProcChaudhuri::normalizeIm(cv::Mat &src, cv::Mat &dst, double dispCutMin, double dispCutMax)
{
	EReport rv(ER_SUCCESS);
	if (src.data)
	{
		if ((src.rows > 0) && (src.cols > 0))
		{
			cv::Mat srcCopy, normIm;
			double imMin(0.), imMax(0.);
			src.copyTo(srcCopy);
			cv::minMaxLoc(srcCopy, &imMin, &imMax);
			if (imMin != imMax)
			{
				normIm = (srcCopy - imMin) / (imMax - imMin);
				normIm.setTo(dispCutMax, normIm > dispCutMax);
				normIm.setTo(dispCutMin, normIm < dispCutMin);
				if (m_postProcParams.isConvert)
				{
					if (m_postProcParams.imDepth == CV_8U)
					{
						normIm = 255. * normIm;
					}
					else if (m_postProcParams.imDepth == CV_16U)
					{
						normIm = 65535. * normIm;
					}
					else //default CV_8U
					{
						normIm = 255. * normIm;
					}
					normIm.convertTo(dst, m_postProcParams.imDepth);
				}
				else
				{
					normIm.copyTo(dst);
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


/****************************************************************
*
*			setAutoBrightnessLevel - private
*
*****************************************************************/
EReport CPostProcChaudhuri::setAutoBrightnessLevel(cv::Mat &src, double &dispCut)
{
	EReport rv(ER_SUCCESS);

	if (m_isInit)
	{
		if ((src.rows > 0) && (src.cols > 0))
		{
			cv::Mat normIm, threshIm;
			double imSize(src.rows*src.cols);
			dispCut = m_postProcParams.startCutValue;
			int nonZeroCount(0);
			rv = normalizeIm(src, normIm, 0.0, 1.0);
			cv::threshold(normIm, threshIm, dispCut, 1, cv::THRESH_BINARY);
			nonZeroCount = cv::countNonZero(threshIm);
			if (static_cast<double>(nonZeroCount) / imSize < m_postProcParams.threshCutValue)
			{
				while (static_cast<double>(nonZeroCount) / imSize < m_postProcParams.threshCutValue)
				{
					dispCut -= 0.01;
					cv::threshold(normIm, threshIm, dispCut, 1, cv::THRESH_BINARY);
					nonZeroCount = cv::countNonZero(threshIm);
				}
			}
			else
			{
				while (static_cast<double>(nonZeroCount) / imSize > m_postProcParams.threshCutValue)
				{
					dispCut += 0.01;
					cv::threshold(normIm, threshIm, dispCut, 1, cv::THRESH_BINARY);
					nonZeroCount = cv::countNonZero(threshIm);
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
		rv = ER_INVALID_INIT;
	}

	return rv;
}

/****************************************************************
*
*			constructors - private
*
*****************************************************************/
CPostProcChaudhuri::CPostProcChaudhuri()
{
	m_isInit = false;
	m_isParamInit = false;
	m_postProcParams = {};
}

CPostProcChaudhuri::~CPostProcChaudhuri()
{

}
