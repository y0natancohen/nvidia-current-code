#include <opencv2/core/core.hpp>

int main(){
    cv::Mat A, B;
    // where image type should be either CV_16UC3 or CV_8UC3

//    A = cv::Mat::zeros();
//    A = cv::Mat::zeros(480, 640, CV_8UC3);
//    A.data = ( unsigned char* ) pRequest->imageData.read();
//                    A.data = pImageData;
    // in order to verify that we are working with deep copy and not only header copy
//    A.copyTo(B);
    // converting from RGB to BGR (BGR is opencv format,
    // and you tald me that the input data is set as [rgb, rgb, rgb...] pixel by pixel)
//    cv::cvtColor(B, B, cv::COLOR_RGB2BGR);
//    cv::imwrite("")
    return 0;
};
