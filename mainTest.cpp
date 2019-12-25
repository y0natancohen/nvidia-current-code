#include "structures.h"
//#include "CChaudhuri.h"
//#include "CPreprocessIm.h"
//#include "CPostProcChaudhuri.h"
#include "CImProcAlgo.h"
#include <numeric>


int main(int argc, char** argv)
{
	EReport rv(ER_SUCCESS);
	cv::Mat X, Y, im, dst, dst1;
	std::vector<cv::Mat> kernels;
	std::vector<cv::Mat> kers;
	std::string filename = "newINI.ini";
	im = cv::imread("L1.tif", cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
	if (!im.data)
	{
		std::cout << "could not open image file" << std::endl;
		return -1;
	}
	//cv::cvtColor(im, im, cv::COLOR_BGR2GRAY);
	cv::Mat bgr[3];
	cv::split(im, bgr);
	bgr[1].copyTo(im);

	std::cout << im.type() << std::endl;
	rv = CImProcAlgo::getReference().init(filename);
	/*rv = CChaudhuri::getReference().init(filename);
	rv = CPreprocessIm::getReference().init(filename);
	rv = CPostProcChaudhuri::getReference().init(filename);
	rv = CPreprocessIm::getReference().preprocessIm(im, dst, X);*/

	//file << "pp_im" << dst;
	//rv = CChaudhuri::getReference().calculateChaudhuri(dst, dst);
	//file << "chaud" << dst;
	//rv = CPostProcChaudhuri::getReference().postProcIm(dst, dst1);
	double e1, e2;
	//for (int i(0) ; i < 10 ; i++)
	//{
	e1 = cv::getTickCount();
	rv = CImProcAlgo::getReference().springImProcAlgo(im, dst);
	e2 = cv::getTickCount();
	std::cout << "performance: " << (e2 -e1)/cv::getTickFrequency() << std::endl;
	//}
	cv::FileStorage file("blur.xml", cv::FileStorage::APPEND);
	file << "final" << dst;
	file.release();
	cv::namedWindow("disp", cv::WINDOW_NORMAL);
	cv::imshow("disp", dst);
	cv::waitKey(0);
	cv::imwrite("finalIm.png", dst);
	std::cout << dst.type() << std::endl << std::endl;
	return 0;
}
