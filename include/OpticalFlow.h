//
// Created by TaozheLi on 11/29/23.
//
#include <opencv2/optflow.hpp>
#include <utils.h>
#ifndef OPTICALFLOWTEST_OPTICALFLOW_H
#define OPTICALFLOWTEST_OPTICALFLOW_H
std::vector<uchar> CalculateOpticalFlow(const cv::Mat &preGrayImg, const cv::Mat &currentGrayImg,const cv::Mat & prevImg, const cv::Mat &currentImg, bool visualize=false){
    std::vector<cv::Point2f> prevGoodFeatures;
    std::vector<cv::Point2f> currentGoodFeatures;
    // Detect good features to track
    std::vector<uchar> status;
    std::vector<float> err;
    cv::goodFeaturesToTrack(preGrayImg, prevGoodFeatures, 500, 0.01, 10);
//    cv::goodFeaturesToTrack(currentGrayImg, currentGoodFeatures, 100, 0.01, 10);
    cv::calcOpticalFlowPyrLK(preGrayImg, currentGrayImg, prevGoodFeatures, currentGoodFeatures, status, err, cv::Size(21, 21), 7);
    std::cout<<prevGoodFeatures.size()<<" "<<currentGoodFeatures.size()<<std::endl;
    if(visualize){
        int count = 0;
        float length = 0;
        float avg_length = 0;
        float std_length = 0;
        float max_length = 0;
        float min_length = 0;
        std::vector<float> lengths;
        cv::Mat showImg = currentImg.clone();
        cv::Mat showPrevImg = prevImg.clone();
        for(int i=0; i<prevGoodFeatures.size(); ++i){
            if(status[i]) {
                cv::circle(showPrevImg, cv::Point(prevGoodFeatures[i].x, prevGoodFeatures[i].y), 3, cv::Scalar(0, 0, 255));
                cv::circle(showImg, cv::Point(currentGoodFeatures[i].x, currentGoodFeatures[i].y), 3, cv::Scalar(0, 0, 255));
                cv::line(showImg, cv::Point(prevGoodFeatures[i].x, prevGoodFeatures[i].y),
                         cv::Point(currentGoodFeatures[i].x, currentGoodFeatures[i].y), cv::Scalar(0, 255, 0));
                count++;
                lengths.push_back(std::pow((currentGoodFeatures[i].x - prevGoodFeatures[i].x) * (currentGoodFeatures[i].x - prevGoodFeatures[i].x) +
                                           (currentGoodFeatures[i].y - prevGoodFeatures[i].y) * (currentGoodFeatures[i].y - prevGoodFeatures[i].y), 0.5));

                length += std::pow((currentGoodFeatures[i].x - prevGoodFeatures[i].x) * (currentGoodFeatures[i].x - prevGoodFeatures[i].x) +
                        (currentGoodFeatures[i].y - prevGoodFeatures[i].y) * (currentGoodFeatures[i].y - prevGoodFeatures[i].y), 0.5);
            }
        }
        cv::imshow("prev img", showPrevImg);
        cv::imshow("optical result", showImg);

        std::cout<<"found ratio: "<<float(count) / float(prevGoodFeatures.size())<<std::endl;
        std::cout<<"avg length: "<<length / float(prevGoodFeatures.size())<<std::endl;
        std::cout<<"std length: "<<std_(lengths)<<std::endl;
        std::cout<<"min length: "<<min_(lengths)<<std::endl;
        std::cout<<"max length: "<<max_(lengths)<<std::endl;

        std::cout<<"start to draw matches"<<std::endl;
        std::vector<cv::KeyPoint> prevKeyFeatures, currentKeyFeatures;
        for(int i=0; i<prevGoodFeatures.size(); ++i){
            prevKeyFeatures.push_back(cv::KeyPoint(prevGoodFeatures[i].x, prevGoodFeatures[i].y, 5));
            currentKeyFeatures.push_back(cv::KeyPoint(currentGoodFeatures[i].x, currentGoodFeatures[i].y, 5));
        }
        cv::Mat matchesImage;
        std::vector<cv::DMatch> Matches;
        for(int i=0; i<prevKeyFeatures.size(); ++i){
            cv::DMatch match;
            match.queryIdx = i;
            match.trainIdx = i;
            match.distance = 1;
        }
        cv::drawMatches(prevImg, prevKeyFeatures, currentImg, currentKeyFeatures, Matches, matchesImage);
        cv::imshow("matches", matchesImage);
    }
//    cv::waitKey(-1);
//    cv::destroyAllWindows();
    return status;

}
#endif //OPTICALFLOWTEST_OPTICALFLOW_H
