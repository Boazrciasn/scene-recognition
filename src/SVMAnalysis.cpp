/*
Muhammet Pakyürek	S008284 Department of Electrical and Electronics
        Baris Ozcan	S010097	Department of Computer Science
*/

#include "SVMAnalysis.h"
#include<iostream>
#include <vector>
#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "ImageReader.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include "BagOfSIFT.h"
#include "Evaluator.h"

#define NUM_OF_CLASS  15


SVMAnalysis::SVMAnalysis(BagOfSIFT *BagOfSIFT) {

    std::cout<<"     "<<std::endl;
    std::cout<<"     "<<std::endl;
    std::cout<<"-------SVM Analysis RUNNING----------"<<std::endl;
    std::cout<<"     "<<std::endl;


    this->dataTestDescriptor = BagOfSIFT->dataTestDescriptor;
    this->dataTrainDescriptor = BagOfSIFT->dataTrainDescriptor;
    this->trainLabels = BagOfSIFT->TrainLabels;
    this->testLabels = BagOfSIFT->TestLabels;

    //Training Data type check
    int nType = this->dataTrainDescriptor.depth();
    if(nType != CV_32F)
    {
        dataTrainDescriptor.convertTo(dataTrainDescriptor, CV_32F);
        dataTestDescriptor.convertTo(dataTestDescriptor, CV_32F);
    }

    nType = this->trainLabels.depth();
    if(nType != CV_32S)
    {
        this->trainLabels.convertTo(this->trainLabels, CV_32S);
        this->trainLabels.convertTo(this->trainLabels, CV_32S);
    }


    SVMAnalysis::SVMTrainer();

    SVMAnalysis::Evaluation();

}



void SVMAnalysis::Evaluation() {
    float count=0;
    int consensus;
    this->Total_Response = this->Total_Response.t();
    std::cout<<this->Total_Response.rows<<"x"<<this->Total_Response.cols <<std::endl;

    for (int i = 0; i <this->Total_Response.rows ; ++i) {

        consensus = getMinIndex(this->Total_Response.row(i));

        if(consensus == this->testLabels.at<int>(i))
            count++;
        this->Consensus.push_back(consensus);
    }

    Evaluator Evaluator(this->Consensus,this->testLabels,"SVM");
    std::cout<<"classification rate : "<<count/this->testLabels.rows*100<< "%"<<std::endl;

}


int SVMAnalysis::getMinIndex(cv::Mat1f mat1) {

    if (mat1.rows != 1)
        std::invalid_argument("");

    int minIndex = 0;
    float minValue = mat1.at<float>(0);
    for (int i = 1; i < mat1.cols; ++i) {
        float value = mat1.at<float>(i);
        if (value < minValue) {
            minIndex = i;
            minValue = value;
           // std::cout<<"MINVAL : " <<minIndex<<std::endl;
        }
    }

    return minIndex;

}



void SVMAnalysis::SVMTester() {

    cv::Mat1f temp;
    cv::Mat1f resultsMat;
    this->svm_obj->predict(this->dataTestDescriptor,resultsMat,cv::ml::StatModel::RAW_OUTPUT);

    temp = resultsMat.t();
    this->Total_Response.push_back(temp);


}


void SVMAnalysis::SVMTrainer() {
    std::string labelstring;
    cv::Mat1i currentSVMlabels; // temporal label container for 15 SVM's

    for(int j=0; j<NUM_OF_CLASS;j++) {

        labelstring= std::to_string(j);
        std::string FileName("SVM_train_"+ labelstring);
        currentSVMlabels= SVMAnalysis::SVMDataCreate1vsALL(j)  ;
        CalculateSVM(currentSVMlabels, FileName);
        SVMAnalysis::SVMTester();

    }
    std::cout<<"C Value : "<< this->svm_obj->getC()  <<std::endl;

}


void SVMAnalysis::CalculateSVM(cv::Mat label, std::string FileName) {


    this->svm_obj = cv::ml::SVM::create();
    this->svm_obj->setC(300);
    setSVMParams(label, true );
    cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(this->dataTrainDescriptor, cv::ml::ROW_SAMPLE, label);
    this->svm_obj->train(tData);
    this->svm_obj->save(FileName);
}

void SVMAnalysis::setSVMParams(const cv::Mat &responses, bool balanceClasses) {

    int pos_ex = countNonZero(responses == 1);
    int neg_ex = countNonZero(responses == -1);
    this->svm_obj->setType(cv::ml::SVM::C_SVC);
    this->svm_obj->setKernel(cv::ml::SVM::LINEAR);
    if( balanceClasses )
    {
        cv::Mat class_wts( 2, 1, CV_32FC1 );
        // The first training sample determines the '+1' class internally, even if it is negative,
        // so store whether this is the case so that the class weights can be reversed accordingly.
        bool reversed_classes = (responses.at<int>(0) < 0.f);
        if( reversed_classes == false )
        {
            class_wts.at<float>(0) = static_cast<float>(pos_ex)/static_cast<float>(pos_ex+neg_ex); // weighting for costs of positive class + 1 (i.e. cost of false positive - larger gives greater cost)
            class_wts.at<float>(1) = static_cast<float>(neg_ex)/static_cast<float>(pos_ex+neg_ex); // weighting for costs of negative class - 1 (i.e. cost of false negative)
        }
        else
        {
            class_wts.at<float>(0) = static_cast<float>(neg_ex)/static_cast<float>(pos_ex+neg_ex);
            class_wts.at<float>(1) = static_cast<float>(pos_ex)/static_cast<float>(pos_ex+neg_ex);
            float temp = class_wts.at<float>(0);
            class_wts.at<float>(0) = class_wts.at<float>(1);
            class_wts.at<float>(1) = temp;
        }
        this->svm_obj->setClassWeights(class_wts);
    }
}


void SVMAnalysis::setSVMTrainAutoParams( cv::ml::ParamGrid &c_grid, cv::ml::ParamGrid &gamma_grid,
        cv::ml::ParamGrid &p_grid, cv::ml::ParamGrid &nu_grid,
        cv::ml::ParamGrid &coef_grid, cv::ml::ParamGrid &degree_grid ){
}

cv::Mat SVMAnalysis::SVMDataCreate1vsALL( int nClass) {

    cv::Mat SVM1vsAllTrainLabel = cv::Mat::ones(this->trainLabels.rows,this->trainLabels.cols,CV_32S)*-1;
    for(int i=0; i<this->trainLabels.rows;i++)
        if(this->trainLabels.at<int>(i) == nClass)
            SVM1vsAllTrainLabel.at<int>(i) = 1;
    return SVM1vsAllTrainLabel;
}

SVMAnalysis::~SVMAnalysis() {

}