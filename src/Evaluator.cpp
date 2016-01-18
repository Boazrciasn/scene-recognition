//
// Created by Barış Özcan on 07/01/16.
//

#include "Evaluator.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


Evaluator::Evaluator(cv::Mat Response, cv::Mat GroundTruth,std::string FileName) {

    cv::Mat truepositives;
    cv::Mat falsepositives;
    cv::Mat falsenegatives;
    int tp;
    int fp = 0;
    int fn = 0;
    Response.convertTo(Response,CV_32S);
    GroundTruth.convertTo(GroundTruth,CV_32S);
    int cnt;
    FileName = "ConfusionMat"+FileName+".jpg";
    cv::Mat temp;
    cv::Mat ConfusionMatrixT = cv::Mat::zeros(15,15,CV_32S);
    cv::Mat ConfusionMatrix ;

    for (int i = 0; i < Response.rows ; ++i) {

        cnt = ConfusionMatrixT.at<int>(Response.at<int>(i),GroundTruth.at<int>(i));
        ConfusionMatrixT.at<int>(Response.at<int>(i),GroundTruth.at<int>(i)) = cnt+1;
    }



    for (int j = 0; j < ConfusionMatrixT.rows ; ++j) {

        for (int i = 0; i <ConfusionMatrixT.cols ; ++i) {

            if (i==j){
                tp = ConfusionMatrixT.at<int>(j,i);
                truepositives.push_back(tp);
            } else  {

                fp += ConfusionMatrixT.at<int>(j,i);
                fn += ConfusionMatrixT.at<int>(i,j);
            }
        }
        falsepositives.push_back(fp);
        falsenegatives.push_back(fn);
        fp = 0;
        fn = 0;
    }


    std::cout<< " TRUE POSITIVES : " << truepositives.t()<<std::endl;
    std::cout<< " FALSE POSITIVES : " << falsepositives.t()<<std::endl;
    std::cout<< " FALSE NEGATIVES : " << falsenegatives.t()<<std::endl;

cv::normalize(ConfusionMatrixT,ConfusionMatrix,0,255,CV_MINMAX,CV_32FC1);
cv::imwrite(FileName,ConfusionMatrix);


}

Evaluator::~Evaluator() {

}
