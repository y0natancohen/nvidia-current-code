#include "CImProcAlgo.h"
//#define CUDA_OPTIM 1
/*///////////////////////////////////////////////////////////////
////////  Public methods implementation ////////////////////////
///////////////////////////////////////////////////////////////*/

CImProcAlgo& CImProcAlgo::getReference()
{
	static CImProcAlgo m_CImProcAlgo;
	return m_CImProcAlgo;
}


/****************************************************************
*
*			init - public
*
*****************************************************************/
EReport CImProcAlgo::init(std::string &filename)
{
	EReport rv(ER_SUCCESS);
	rv = CPreprocessIm::getReference().init(filename);
	if (ER_SUCCESS == rv)
	{
		rv = CChaudhuri::getReference().init(filename);
	}
	if (ER_SUCCESS == rv)
	{
		rv = CPostProcChaudhuri::getReference().init(filename);
	}

	if (ER_SUCCESS == rv)
	{
		m_isInit = true;
	}

	return rv;
}

/****************************************************************
*
*			springImProcAlgo - public
*
*****************************************************************/
EReport CImProcAlgo::springImProcAlgo(cv::Mat &src, cv::Mat &dst)
{
	EReport rv(ER_SUCCESS);

	if (m_isInit)
	{
		rv = CPreprocessIm::getReference().preprocessIm(src, m_imPp, m_mask);
		if (ER_SUCCESS == rv)
		{
# if !CUDA_OPTIM
			CChaudhuri::getReference().setMask(m_mask);
			rv = CChaudhuri::getReference().calculateChaudhuri(m_imPp, m_chaud);
#endif
#if CUDA_OPTIM
            cm_mask.upload(m_mask);
            cm_imPp.upload(m_imPp);
            CChaudhuri::getReference().setCudaMask(cm_mask);
            rv = CChaudhuri::getReference().calculateChaudhuri(cm_imPp, cm_chaud);
#endif // CUDA_OPTIM
		}
		if (ER_SUCCESS == rv)
		{
#if CUDA_OPTIM
            cm_chaud.download(m_chaud);
#endif // CUDA_OPTIM
			rv = CPostProcChaudhuri::getReference().postProcIm(m_chaud, dst);
		}
	}
	else
	{
		rv = ER_INVALID_INIT;
	}

	return rv;
}

/*///////////////////////////////////////////////////////////////
////////  Private methods implementation ////////////////////////
///////////////////////////////////////////////////////////////*/
CImProcAlgo::CImProcAlgo()
{
	m_isInit = false;
}

CImProcAlgo::~CImProcAlgo()
{

}
