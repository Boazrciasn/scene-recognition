//
// Created by Barış Özcan on 07/01/16.
//

#ifndef PRJ3_008284_010097_EVALUATOR_H
#define PRJ3_008284_010097_EVALUATOR_H


#include <opencv2/core/mat.hpp>

class Evaluator {

public:
    Evaluator(cv::Mat Response, cv::Mat GroundTruth, std::string FileName);


    ~Evaluator();

private:

};


#endif //PRJ3_008284_010097_EVALUATOR_H
