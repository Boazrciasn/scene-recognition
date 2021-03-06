//
// Created by Barış Özcan on 07/01/16.
//

#include "Evaluator.h"
#include "SVMAnalysis.h"
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

//    cv::imwrite(FileName,ConfusionMatrix);
    Evaluator::computeAccuracy(Response,GroundTruth);

    cv::Mat SVM_ConfusionMat;
    computeConfusionMat(GroundTruth,Response,SVM_ConfusionMat);
//    std::cout<<SVM_ConfusionMat<<std::endl;
    cv::namedWindow("SVM confusion mat",CV_WINDOW_NORMAL);
    cv::imshow("SVM confusion mat",SVM_ConfusionMat);
    cv::waitKey(0);

}

void Evaluator::computeConfusionMat(cv::Mat &expected, cv::Mat &predicted, cv::Mat &confMat)
{
    confMat = cv::Mat::zeros(15,15, CV_32FC1);

    std::cout<<expected.rows<<" "<<expected.cols<<std::endl;
    for (int i = 0; i < expected.rows; i++) {
        confMat.at<float>(expected.at<int>(i,0),predicted.at<int>(i,0))++;
    }

//    std::cout<<expected.t()<<std::endl;
//    std::cout<<predicted.t()<<std::endl;

    for (int i = 0; i < confMat.rows; i++)
        confMat.row(i) /= sum(confMat.row(i))[0];
}

void Evaluator::computeAccuracy(cv::Mat Response, cv::Mat GroundTruth)
{
    // 5 NUM_OF_QUADRANTS + 1 (one is qhole image itself)
    std::cout<<"\n computeAccuracy \n"<<std::endl;
    int numOfTestInputs = GroundTruth.rows/(1 + NUM_OF_QUADRANTS);
    cv::Mat GroundTruthLabels = GroundTruth(cv::Range(0,numOfTestInputs),cv::Range::all());
    cv::Mat ResponseLabels;

    for (int i = 0; i < numOfTestInputs; i++) {
        int count[15] = {};
        int countFourquad[15] = {};
        for (int j = i; j < Response.rows; j += numOfTestInputs) {
            count[Response.at<int>(j)]++;

            if(j >= 17*numOfTestInputs)
            countFourquad[Response.at<int>(j)]++;
        }



        // major vote count
        int max = count[0];
        int index = 0;

        for (int j = 1; j < 15; j++) {
            if(max < count[j])
            {
                max = count[j];
                index = j;
            }
        }

        // checkpoint
        if(max < 7)
            ResponseLabels.push_back(Response.at<int>(i,0));
        else
            ResponseLabels.push_back(index);

//        if(max < 8)
//        {
//            // major vote count
//            int max = countFourquad[0];
//            int index = 0;

//            for (int j = 1; j < 15; j++) {
//                if(max < countFourquad[j])
//                {
//                    max = countFourquad[j];
//                    index = j;
//                }
//            }

//            if(max < 3)
//                ResponseLabels.push_back(Response.at<int>(i,0));
//            else
//                ResponseLabels.push_back(index);
//        }
//        else
//            ResponseLabels.push_back(index);

    }

    ResponseLabels.convertTo(ResponseLabels, CV_32S);
    GroundTruthLabels.convertTo(GroundTruthLabels, CV_32S);
    // create 0,1 Mat (0: false, 1: true)
    cv::Mat out = (ResponseLabels == GroundTruthLabels)/255;
    // calculate accuracy
    this->accuracy = sum(out)[0]/out.rows;

    std::cout<< " ACCURACY : " << 100*this->accuracy <<"%"<< std::endl;
    std::cout<< " ACCURACY : " << ResponseLabels.rows << "  "<< numOfTestInputs<<std::endl;

}

Evaluator::~Evaluator() {

}
