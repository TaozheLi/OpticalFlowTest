#include <iostream>
#include <opencv2/optflow.hpp>
#include <opencv2/opencv.hpp>
#include <OpticalFlow.h>
int main() {
    std::string imgPath1, imgPath2;
    cv::Mat prev, current, prevGray, currentGray;
    imgPath1 = "/home/rushmian/Datasets/Kitti/data_odometry_color/dataset/sequences/01/image_2/000155.png";
    imgPath2 = "/home/rushmian/Datasets/Kitti/data_odometry_color/dataset/sequences/01/image_2/000156.png";
    prev = cv::imread(imgPath1, cv::ImreadModes::IMREAD_UNCHANGED);
    current = cv::imread(imgPath2, cv::ImreadModes::IMREAD_UNCHANGED);
    std::cout<<"image size: "<<current.size<<std::endl;
    if(prev.channels() != 1){
        cv::cvtColor(prev, prevGray, cv::COLOR_BGR2GRAY);
    }
    if(current.channels()!=1){
        cv::cvtColor(current, currentGray, cv::COLOR_BGR2GRAY);
    }

    CalculateOpticalFlow(prevGray, currentGray, prev, current, true);
    std::vector<cv::Point2f> prevGoodFeatures;
    std::vector<cv::Point2f> currentGoodFeatures;
    cv::goodFeaturesToTrack(prevGray, prevGoodFeatures, 500, 0.01, 10);
//    cv::goodFeaturesToTrack(currentGray, currentGoodFeatures, 100, 0.01, 10);
    std::vector<double> depth; depth.reserve(prevGoodFeatures.size());
    std::cout<<"initialize depth size: "<<prevGoodFeatures.size()<<std::endl;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevGray, currentGray, prevGoodFeatures, currentGoodFeatures, status, err, cv::Size(21, 21), 1);
    for(int i=0; i<prevGoodFeatures.size(); ++i){
//        if(i%3 == 0) depth.push_back(2.0);
//        if(i%3 == 1) depth.push_back(10.0);
//        if(i%3 == 2) depth.push_back(15.0);
        depth.push_back(15.0);
    }
    cv::Mat currentCopy = current.clone();
    for(int i=0; i<status.size(); ++i){

        if(status[i]){
            if(cv::norm(prevGoodFeatures[i] - currentGoodFeatures[i]) < 5) continue;
            cv::circle(currentCopy, cv::Point(currentGoodFeatures[i].x, currentGoodFeatures[i].y), 3, cv::Scalar(0, 0, 255));
            cv::line(currentCopy, cv::Point(prevGoodFeatures[i].x, prevGoodFeatures[i].y),
                     cv::Point(currentGoodFeatures[i].x, currentGoodFeatures[i].y), cv::Scalar(0, 255, 0));
        }
//        if(i == 500) break;
    }
    cv::imshow("before process ", currentCopy);
    std::vector<bool> Vstatus(prevGoodFeatures.size(), true);
    for(int i=0; i<status.size(); ++i){
        if(!status[i]) Vstatus[i] = false;
    }
    std::vector<int> groups;
    int classes = 20;
    groups = classifyBasedOnDepth(classes, depth);
//    RemovePointsThroughDepth(classes, groups, prevGoodFeatures, currentGoodFeatures, 3, 3, Vstatus);
    double a = 1.5;
    double b = 1.5;
    ClassifyBasedOnXY(classes, a, b, groups, prevGoodFeatures, currentGoodFeatures, Vstatus);
    double length = 0;
    for(const auto & x: Vstatus){
        if(x) length++;
    }
    std::cout<<"original length: "<<Vstatus.size()<<" final length: "<<length;
    cv::Mat originalCurrentCopyImage = current.clone();
    for(int i=0; i<Vstatus.size(); ++i){

        if(Vstatus[i]){
            if(cv::norm(prevGoodFeatures[i] - currentGoodFeatures[i]) < 5) continue;
            cv::circle(originalCurrentCopyImage, cv::Point(currentGoodFeatures[i].x, currentGoodFeatures[i].y), 3, cv::Scalar(0, 0, 255));
            cv::line(originalCurrentCopyImage, cv::Point(prevGoodFeatures[i].x, prevGoodFeatures[i].y),
                     cv::Point(currentGoodFeatures[i].x, currentGoodFeatures[i].y), cv::Scalar(0, 255, 0));
        }
//        if(i == 100) break;
    }
    cv::imshow("after process", originalCurrentCopyImage);
    cv::waitKey(-1);
    cv::destroyAllWindows();

    const float xmax = 10;
    const float xmin = 5;
    const float ymax = 10;
    const float ymin = 5;
    //situation 1
    cv::Point2f p1(xmax, ymin);
    cv::Point2f p2(xmin, ymax);
    std::cout<<"\ncase 01: "<<ComputeAngleAtan2(p1, p2)<<std::endl;

    //situation 2
    cv::Point2f p3(xmin, ymin);
    cv::Point2f p4(xmax, ymax);
    std::cout<<"case 02: "<<ComputeAngleAtan2(p3, p4)<<std::endl;

    //situation 3
    cv::Point2f p5(xmin, ymax);
    cv::Point2f p6(xmax, ymin);
    std::cout<<"case 03: "<<ComputeAngleAtan2(p5, p6)<<std::endl;

    //situation 4
    cv::Point2f p7(xmax, ymax);
    cv::Point2f p8(xmin, ymin);
    std::cout<<"case 04: "<<ComputeAngleAtan2(p7, p8)<<std::endl;
}
