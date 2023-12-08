//
// Created by TaozheLi on 11/29/23.
//
#include <numeric>
#include <cmath>
#ifndef OPTICALFLOWTEST_UTILS_H
#define OPTICALFLOWTEST_UTILS_H
double avg_(const std::vector<float> &v){
    double sum;
    sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / double(v.size());
}

double avg_double(const std::vector<double> &v){
    double sum;
    sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / double(v.size());
}

double std_(const std::vector<float> & v){
    double mean = avg_(v);
    double sum = 0;
    for(const auto & x: v){
        sum += (x - mean) * (x- mean);
    }
    return sqrt(sum) / double(v.size());
}

double std_double(const std::vector<double> & v){
    double mean = avg_double(v);
    double sum = 0;
    for(const auto & x: v){
        sum += (x - mean) * (x- mean);
    }
    return sqrt(sum) / double(v.size());
}

float  max_(const std::vector<float> &v){
    float max = v[0];
    for(const auto &x:v){
        if(x > max){
            max = x;
        }
    }
    return max;
}

double  max_double(const std::vector<double> &v){
    double max = v[0];
    for(const auto &x:v){
        if(x > max){
            max = x;
        }
    }
    return max;
}

float min_(const std::vector<float> &v){
    float min = v[0];
    for(const auto & x: v){
        if(x < min){
            min = x;
        }
    }
    return min;
}

std::vector<int> classifyBasedOnDepth(const int & classes, const std::vector<double> &featureDepth){
    double max_depth = max_double(featureDepth);
    double segment = (max_depth + 1.0) / double(classes);
    std::vector<int> groups;
    for(int i=0; i<featureDepth.size(); ++i){
        //
        double depth = featureDepth[i];
        int group;
        for(int s = 0; s<classes; ++s){
            if(depth > (s + 1) * segment) continue;
            if(depth > s * segment) group = s;
        }
        groups.push_back(group);
    }
    return groups;
}



void ClassifyBasedOnXY(const int &classes, const double &a, const double &b,
                                   const std::vector<int> & groups, const std::vector<cv::Point2f> &featurePointsPrev, const std::vector<cv::Point2f> &featurePointsCurrent, std::vector<bool> &status){
    void ClassifyBasedOnXYAndRemovePoint(const double &a, const double &b,const std::vector<int> &IndexOfOneGroup, const std::vector<cv::Point2f> &featurePointPrev,
                                         const std::vector<cv::Point2f> &featurePointCurrent, std::vector<bool> &status, const std::vector<double> &orientation, const bool & useGlobalInformation);
    std::vector<double> ComputingGlobalOrientation(const std::vector<bool> & status, const std::vector<cv::Point2f> &featurePointsPrev, const std::vector<cv::Point2f> &featurePointsCurrent);
    std::vector<std::vector<int>> ClassifyThroughDepth;
    ClassifyThroughDepth.resize(classes);
    for(int i=0; i<groups.size(); ++i){
        ClassifyThroughDepth[groups[i]].push_back(i);
    }

    std::vector<double> globalOrientation = ComputingGlobalOrientation(status, featurePointsPrev, featurePointsCurrent);

    for(int i=0; i<ClassifyThroughDepth.size(); ++i){
        if(ClassifyThroughDepth[i].empty()) continue;
        std::cout<<"group :"<<i<<" size: "<<ClassifyThroughDepth[i].size()<<std::endl;
        ClassifyBasedOnXYAndRemovePoint(a, b, ClassifyThroughDepth[i], featurePointsPrev, featurePointsCurrent, status, globalOrientation, true);
    }

}

bool RemovedConditionOnlyLength(const double &opticalFlowLength, const double & mean, const double & std, const double & a, const double &b){
    if(opticalFlowLength > mean + a * std || opticalFlowLength < mean - b * std)
        return true;
    else{
        return false;
    }
}

bool RemovedConditionOnlyOrientation(const double & theta, const double &mainTheta) {
    double theta_threshold = 30.0;
    if(abs(theta - mainTheta) > theta_threshold ) return true;
    else
        return false;
}

bool RemovedCondition(const double &opticalFlowLength, const double & mean, const double & std, const double & a, const double &b,
                      const double & theta, const double &mainTheta, const double &globalOrientation) {
    double alpha = 0.0;
    double beta = 1 - alpha;
    if(RemovedConditionOnlyLength(opticalFlowLength, mean, std, a, b) || RemovedConditionOnlyOrientation(theta, alpha * mainTheta + beta * globalOrientation))
        return true;
    else
        return false;
}

double ComputeAngle(const cv::Point2f & pPrev, const cv::Point2f & pCurrent){
    return atan2((pCurrent.y - pPrev.y),  (pCurrent.x - pPrev.x) ) * 180.0 / M_PI;
}

std::vector<double> ComputingGlobalOrientation(const std::vector<bool> & status, const std::vector<cv::Point2f> &featurePointsPrev,
                                const std::vector<cv::Point2f> &featurePointsCurrent){
    const int width = 1232;
    const int parts = 3;
    double segment = double(width+1) / double(parts);
    std::vector<double> globalOrientation(parts, 0.0);
    std::vector<double> count(parts, 0.0);
    for(int i=0; i<status.size(); ++i){
        if(status[i]){
            int b;
            for(int k=0; k<parts; k++){
                if(featurePointsPrev[i].x > (k+1) * segment ) continue;
                b = k;
                break;
            }
            globalOrientation[b] += ComputeAngle(featurePointsCurrent[i], featurePointsPrev[i]);
//            if(b == 1)
//            std::cout<<ComputeAngle(featurePointsCurrent[i], featurePointsPrev[i])<<std::endl;
            count[b]++;
        }
    }
    for(int i=0; i<parts; ++i){
        globalOrientation[i] = (globalOrientation[i] / count[i]);
        std::cout<<"region: "<<i<<" orientation: "<<globalOrientation[i]<<std::endl;
    }
    return globalOrientation;
}
bool Converse(std::vector<double> &angles, const double & threshRatio){
    double total = double(angles.size());
    double count = 0.0;
    std::cout<<"start to run converse !!!! "<<std::endl;
    std::cout<<angles.size()<<std::endl;
    for(int i=0; i<angles.size(); ++i){
        std::cout<<"i:"<<i<<" angle: "<<angles[i]<<std::endl;
    }
    for(const auto & angle: angles){
        if(abs(angle) > 90)  count+=1;
    }
    std::cout<<"count: "<<count<<std::endl;

    // need to converse
    if((count / total) > threshRatio) {
        for(auto angle: angles){
            if(angle < 0) angle = -angle - 180;
            else{
                angle = -angle + 180;
            }
        }
        return true;
    }
    else
        return false;
}

float ComputeAngleAtan2(const cv::Point2f &p1, const cv::Point2f &p2){
    return atan2((p1.y - p2.y ), (p1.x - p2.x)) * 180 / M_PI;
}

void ClassifyBasedOnXYAndRemovePoint(const double &a, const double &b,const std::vector<int> &IndexOfOneGroup, const std::vector<cv::Point2f> &featurePointPrev,
                                     const std::vector<cv::Point2f> &featurePointCurrent, std::vector<bool> &status, const std::vector<double> &globalOrientation, const bool & useGlobalInformation = false){
    const int width = 1232;
    const int height = 371;
    const int minimumNumber = 5;
    const int nb = 10;
    const int mb = 3;
    const int totalClasses = nb * mb;
    const int mw = height / mb;
    const int nw = width / nb;
    const int parts = 3; int type;
    if(IndexOfOneGroup.size() <= 5) return;
    std::cout<<"start to based on x and y"<<std::endl;
    // if number of points is too small, don't remove points
    std::vector<std::vector<int>> newGroups;
    std::vector<std::vector<double>> length;
    std::vector<std::vector<double>> angle;
    length.resize(totalClasses);
    newGroups.resize(totalClasses);
    angle.resize(totalClasses);
    double x_cor = 0.0;
//    for(int i=0; i<totalClasses; ++i){
//        length[i].;
//        newGroups[i].reserve(10000);
//        angle[i].reserve(10000);
//
//    }
    std::vector<int> count_n(30, 0);
    for(int i=0; i<IndexOfOneGroup.size(); ++i){
        // it's good points
        int originalIndex = IndexOfOneGroup[i];
        if(status[originalIndex]){
            int _row = featurePointPrev[i].y / mw;
            int _col = featurePointPrev[i].x / nw;
            int group = _row * nb + _col;
            newGroups[group].push_back(originalIndex);
            length[group].push_back(cv::norm(featurePointCurrent[originalIndex] - featurePointPrev[originalIndex]));
//            if(group == 3){
//                std::cout<<"i: "<<ComputeAngle(featurePointPrev[originalIndex], featurePointCurrent[originalIndex])<<std::endl;
//            }
            double _ = ComputeAngle(featurePointPrev[originalIndex], featurePointCurrent[originalIndex]);
            if(_ < -180.0 || _ > 180.0){
                std::cout<<featurePointPrev[originalIndex]<<" "<<featurePointCurrent[originalIndex]<<std::endl;
                std::exit(-1);
            }
            angle[group].push_back(_);
            count_n[group] += 1;
            x_cor += featurePointPrev[i].x;
        }
    }
    for(int i=0; i<angle[3].size(); i++){
        std::cout<<"group 3 4 i: "<<i<<" angle: "<<angle[3][i]<<std::endl;
    }
    x_cor = x_cor / double(IndexOfOneGroup.size());
    for(int i=0; i<parts; ++i){
        if(x_cor > (i+1) * double(width + 1) / double(parts)) continue;
        type = i;
        break;
    }
    // check
    for(int i=0; i<newGroups.size(); ++i){
        if(newGroups[i].empty()) continue;
//        std::cout<<"xy group: "<<i<<" size: "<<newGroups[i].size()<<std::endl;
    }
    // check it again
    for(int i=0; i<angle[3].size(); i++){
        std::cout<<"group 3 i: "<<i<<" angle: "<<angle[3][i]<<std::endl;
    }
    // converse
    double threshRatio = 0.5;
    int count = 0;
    for(int i=0; i<angle.size(); ++i) {
        if (angle[i].empty()) continue;
        std::cout<<"not empty group: "<<i<<std::endl;
        if (Converse(angle[i], threshRatio)) count+=1;
    }
    std::cout<<"there are total "<<count<<" conversed group"<<std::endl;
    for(int i=0; i<totalClasses; ++i){
        // no element in these group
        if(newGroups[i].empty()) continue;
        if(newGroups[i].size() < minimumNumber) continue;
        double avg_optflow = avg_double(length[i]);
        double std_optflow = std_double(length[i]);
        double avg_angle = avg_double(angle[i]);
        double std_angle = std_double(angle[i]);
        std::cout<<" xy_group: "<<i<<" mean_optflow: "<<avg_optflow<<" std_optflow: "<<std_optflow;
        std::cout<<" mean_angle: "<<avg_angle<<std::endl;
        for(int each_idx=0; each_idx < newGroups[i].size(); ++each_idx){
            // detect angle rangle

            double optflow = length[i][each_idx];
            double theta = angle[i][each_idx];
            std::cout<<"optflow: "<<optflow<<" theta: "<<theta<<std::endl;
            if(RemovedCondition(optflow, avg_optflow, std_optflow, a, b, theta, avg_angle, globalOrientation[type])){
                int originalIndex = newGroups[i][each_idx];
                status[originalIndex] = false;
            }
        }
    }
}

void RemovePointsThroughDepth(const int & classes, const std::vector<int> & groups, const std::vector<cv::Point2f> &prevFeatures,
                                                          const std::vector<cv::Point2f> &currentFeatures, const double &a, const double & b, std::vector<bool> &status){
    assert(prevFeatures.size() == currentFeatures.size());
    std::vector<std::pair<double, double>> AvgAndStdArray;
    std::vector<double> Lengths;
    std::vector<std::vector<double>> LengthOfEachGroups(classes);
    for(int i=0; i<prevFeatures.size(); ++i){
        int group = groups[i];
        double length = std::pow((currentFeatures[i].x - prevFeatures[i].x) * (currentFeatures[i].x - prevFeatures[i].x) +
                                 (currentFeatures[i].y - prevFeatures[i].y) * (currentFeatures[i].y - prevFeatures[i].y), 0.5);
        LengthOfEachGroups[group].push_back(length);
        Lengths.push_back(length);
    }
    for(int i=0; i<classes; ++i){
        std::pair<double, double> _;
        if(LengthOfEachGroups[i].size() != 0) {
            _.first = avg_double(LengthOfEachGroups[i]);
            _.second = std_double(LengthOfEachGroups[i]);
        }
        else{
            _.first = 0;
            _.second = 0;
        }
        AvgAndStdArray.push_back(_);
    }
//    for(int i=0; i<classes; ++i){
//        std::cout<<"group :"<<i<<" avg :"<<AvgAndStdArray[i].first<<" std: "<<AvgAndStdArray[i].second<<std::endl;
//    }
    for(int i=0; i<prevFeatures.size(); ++i){
        int group = groups[i];
        double avg = AvgAndStdArray[group].first;
        double std = AvgAndStdArray[group].second;
        double length = Lengths[i];
        if(length < (avg - a*std) || length > (avg + b * std)){
            status[i] = false;
        }
    }

}
#endif //OPTICALFLOWTEST_UTILS_H
