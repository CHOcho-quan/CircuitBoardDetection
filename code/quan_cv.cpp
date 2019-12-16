/**
 * A method to detect the given circuit board from a picture
 * HSV space thresholding, SIFT feature extraction and match
 * Local Feature Examination - Hough Circle Detection
 * Distance-oriented search for the correct rect and NMS
 * Created by Quan 
**/
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <chrono>

using namespace std;
using namespace cv;

#define DEBUG 0

//TO DO : Image Pyramid Detection

//CBD Namespace, short for Circuit Board Detection
namespace CBD {
    //UTIL class
    class rectScore 
    {
        public:
            //Referencing the top corner point (x, y) and its total distance calculated by SIFT
            int rectX;
            int rectY;
            int rectDist;
            rectScore(int x, int y, int score)
            {
                rectX = x;
                rectY = y;
                rectDist = score;
            }

            rectScore(const rectScore &s)
            {
                rectX = s.rectX;
                rectY = s.rectY;
                rectDist = s.rectDist;
            }
    };

    //Clustering all the same rectangles
    vector<vector<rectScore> > checkCloseRect(vector<vector<rectScore> > rC, rectScore r, int sample_x, int sample_y)
    {
        if (rC.size() == 0)
        {
            vector<rectScore> tmp;
            tmp.push_back(r);
            rC.push_back(tmp);
            return rC;
        }

        //Check if they are probably the same rect
        bool inFlag = false;
        for (int i = 0;i < rC.size();i++)
        {
            //Getting main element in this singleRect
            sort(rC[i].begin(), rC[i].end(), 
                [](const rectScore & r1, const rectScore & r2)->bool {return r1.rectDist < r2.rectDist;});
            rectScore rect = rC[i][0];
            
            if (abs(r.rectX - rect.rectX) <= sample_x && abs(r.rectY - rect.rectY) <= sample_y) inFlag = true;
            if (inFlag) {
                rC[i].push_back(r);
                break;
            }
        }

        if (!inFlag)
        {
            vector<rectScore> tmp;
            tmp.push_back(r);
            rC.push_back(tmp);
        }

        return rC;
    }

    //DO NMS thresh to the boxes and get the final result
    Mat nmsBox(Mat result, vector<vector<rectScore> > rC, int sample_x, int sample_y)
    {
        for (auto singleRect : rC)
        {
            cout << "SIZE: " << (singleRect.size()) << endl;
            if (singleRect.size() == 1) {
                rectangle(result, Rect(singleRect[0].rectX, singleRect[0].rectY, sample_x, sample_y), Scalar(0, 255, 255), 3);
                continue;
            }

            //Now do NMS by voting
            auto minmax_x = minmax_element(singleRect.begin(), singleRect.end(),
                                    [](const rectScore &m1, const rectScore &m2) { return m1.rectX < m2.rectX; });
            auto minmax_y = minmax_element(singleRect.begin(), singleRect.end(),
                                    [](const rectScore &m1, const rectScore &m2) { return m1.rectY < m2.rectY; });
            int min_x = minmax_x.first->rectX, min_y = minmax_y.first->rectY, max_x = minmax_x.second->rectX, max_y = minmax_y.second->rectY;
            
            if (singleRect.size() == 2) {
                rectangle(result, Rect(min_x, min_y, max_x-min_x+sample_x, max_y-min_y+sample_y), Scalar(0, 255, 255), 3);
                continue;
            }

            //Simple Thresh of the rect's length of side
            if ((max_x-min_x+sample_x) >= 4.5 * sample_x / 3)
            {
                min_x += sample_x/3;
                max_x -= sample_x/3;
            }
            if (max_y-min_y+sample_y >= 4.5 * sample_y / 3)
            {
                min_y += sample_y/3;
                max_y -= sample_y/3;
            }
            rectangle(result, Rect(min_x, min_y, max_x-min_x+sample_x, max_y-min_y+sample_y), Scalar(0, 255, 255), 3);
            continue;

            //TODO : VOTING algorithm
            /**
             * First calculate the most-voted part of this part nms box
             * Secondly we calculate two side's voting
             * Last we predict which part of the central part it is and put bounding box
             * **/
            int num_x = (max_x - min_x) / (sample_x / 3) + 3, num_y = (max_y - min_y) / (sample_y / 3) + 3;
            if (DEBUG) cout << "num: " << num_x * num_y << endl;
            int *count;
            count = new int[num_x * num_y];
            memset(count, 0, sizeof(int)*num_x*num_y);

            //Counting the num of each rect by voting
            for (auto rect : singleRect)
            {
                int this_x = rect.rectX, this_y = rect.rectY;
                int coordinate_x = (this_x - min_x) / (sample_x / 3), coordinate_y = (this_y - min_y) / (sample_y / 3);
                // cout << coordinate_x << " :<->: " << coordinate_y << endl;

                //Calculating the count of this x and y box
                count[coordinate_y * num_x + coordinate_x] += 1;
                count[coordinate_y * num_x + coordinate_x + 1] += 1;
                count[coordinate_y * num_x + coordinate_x + 2] += 1;
                count[(coordinate_y + 1) * num_x + coordinate_x] += 1;
                count[(coordinate_y + 1) * num_x + coordinate_x + 1] += 1;
                count[(coordinate_y + 1) * num_x + coordinate_x + 2] += 1;
                count[(coordinate_y + 2) * num_x + coordinate_x] += 1;
                count[(coordinate_y + 2) * num_x + coordinate_x + 1] += 1;
                count[(coordinate_y + 2) * num_x + coordinate_x + 2] += 1;

                // cout << "DEBUG" << endl;
                // for (int i = 0;i < num_x * num_y;i++) cout << count[i] << ' ';
                // cout << endl;
            }

            // vector<int> vote_x, vote_y;
            // for (int i = 0;i < num_x * num_y;i++) {
            //     if (count[i] >= 3) {
            //         vote_x.push_back(i % num_x);
            //         vote_y.push_back(i / num_x);
            //     }
            // }

            // auto minmax_vx = minmax_element(vote_x.begin(), vote_x.end());
            // auto minmax_vy = minmax_element(vote_y.begin(), vote_y.end());
            // cout << minmax_vx.first[0] << ' ' << minmax_vx.second[0] << endl;
            // cout << minmax_vy.first[0] << ' ' << minmax_vy.second[0] << endl;
            // rectangle(result, Rect(min_x + minmax_vx.first[0] * sample_x / 3, 
            //           min_y + minmax_vy.first[0] * sample_y / 3,
            //          (minmax_vx.second[0]-minmax_vx.first[0])*sample_x/3+sample_x/3, 
            //          (minmax_vy.second[0]-minmax_vy.first[0])*sample_y/3+sample_y/3), 
            //          Scalar(0, 255, 255), 3);

            // for (int i = 0;i < num_x * num_y;i++) cout << count[i] << ' ';
        }

        return result;
    }

    //Selecting the correct rect with the circuit board
    Mat selectRectBoard(Mat result, vector<rectScore> rS, int sample_x, int sample_y, int threshold=180)
    {
        //First we get the Min & Max element of the distance to give us a reference
        sort(rS.begin(), rS.end(), 
                [](const rectScore & r1, const rectScore & r2)->bool {return r1.rectDist < r2.rectDist;});
        int min_dist = rS[0].rectDist;

        vector<vector<rectScore> > rectChoose;
        //Go through the vector from the min distance to max distance
        for (auto r : rS)
        {
            //To far away from our distance so it's not considered as corrct circuit board
            int dist = r.rectDist, x = r.rectX, y = r.rectY;
            if (dist > min_dist + threshold) break;
            
            //IF CLOSE THEN PUT INTO rectSingle
            rectChoose = checkCloseRect(rectChoose, r, sample_x, sample_y);
            rectangle(result, Rect(x, y, sample_x, sample_y), Scalar(0, 0, 255), 3);
        }

        //We choose the correct one to get the final result
        result = nmsBox(result, rectChoose, sample_x, sample_y);

        return result;
    }

    //HSV for blue to filter out ROI to find the true 
    Mat hsvThreshold(Mat detected)
    {
        Mat hsv, mask_detected;
        cvtColor(detected, hsv, COLOR_BGR2HSV);
        inRange(hsv, Scalar(100, 43, 46), Scalar(124, 255, 255), mask_detected);

        return mask_detected;
    }

    //Detect the circuit board by feature points like SIFT or something
    int detectBoardFeature(Mat sample_c, Mat roi, bool verbose=true) 
    {
        //Try : Hough Circle Detection
        vector<Vec3f> circles;
        Mat gray;
        cvtColor(roi, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(3, 3), 2 ,2);
        HoughCircles(gray, circles, HOUGH_GRADIENT, 1, 5, 100, 60, 0, 0);
        if (DEBUG) cout << circles.size() << endl;
        if (circles.size() == 0) return 2147483647;

        //We want the lower part of feature
        Mat sample = Mat(sample_c, Rect(0, sample_c.rows/2, sample_c.cols, sample_c.rows/2));

        //Getting Feature Point
        std::vector<KeyPoint> kp1, kp2;
        Mat descriptor1, descriptor2;
        Ptr<Feature2D> sift1 = xfeatures2d::SIFT::create(10);
        Ptr<Feature2D> sift2 = xfeatures2d::SIFT::create(100);
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        sift1->detectAndCompute(sample, Mat(), kp1, descriptor1);
        sift2->detectAndCompute(roi, Mat(), kp2, descriptor2);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        if (verbose) cout << "extract SIFT cost = " << time_used.count() << " seconds. " << endl;

        //Hamming Distance, BREIF descriptor
        vector<DMatch> matches;
        BFMatcher matcher = BFMatcher(NORM_L2);
        t1 = chrono::steady_clock::now();
        matcher.match(descriptor1, descriptor2, matches);
        t2 = chrono::steady_clock::now();
        time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        if (verbose) cout << "match SIFT cost = " << time_used.count() << " seconds. " << endl;

        //Selecting Matched points
        auto min_max = minmax_element(matches.begin(), matches.end(),
                                        [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
        double min_dist = min_max.first->distance;
        double max_dist = min_max.second->distance;

        double total_dist = 0;
        for (auto i : matches) total_dist += i.distance;

        if (DEBUG)
        {
            printf("-- Max dist : %f \n", max_dist);
            printf("-- Min dist : %f \n", min_dist);
            printf("-- Total dist : %f \n", total_dist);
        }

        //Dis(Describer) > 2 * normal - False with experienced value 30.0 as lower bound
        std::vector<DMatch> good_matches;
        for (int i = 0; i < descriptor1.rows; i++) {
            if (matches[i].distance <= max(2 * min_dist, 30.0)) {
                good_matches.push_back(matches[i]);
            }
        }

        //-- Draw Matches
        Mat img_match;
        Mat img_goodmatch;
        drawMatches(sample, kp1, roi, kp2, good_matches, img_goodmatch);
        // imshow("good matches", img_goodmatch);
        // waitKey(0);

        return total_dist;
    }

    //Main Algorithm to do Circuit Board Detection(CBD)
    Mat detectBoard(Mat sample, Mat detected, Mat mask_detected, int BlueThreshold=200000, int UpSampling=1)
    {
        //Set sample size and parameters
        Mat result = detected.clone();
        vector<rectScore> rS;
        int sample_size_y = sample.rows, sample_size_x = sample.cols;
        int BluePercent = 0, StepSize_x = sample_size_x / 3, StepSize_y = sample_size_y / 3;

        //Find the BlueD area and make sure here is the circuit board
        //Outer Loop : For Upsampling
        int mtotal = 2147483647, tx, ty;
        for (int i = 0;i < UpSampling;i++)
        {
            if (i) {
                pyrUp(detected, detected);
                pyrUp(mask_detected, mask_detected);
            }

            //Inner Loop : Find the BlueD part and go on to feature point or other detection
            for (int x = 0;x < detected.cols - sample_size_x;x+=StepSize_x)
            {
                for (int y = 0;y < detected.rows - sample_size_y;y+=StepSize_y)
                {
                    Mat detected_ROI = Mat(mask_detected, Rect(x, y, sample_size_x, sample_size_y));
                    Mat roi = Mat(detected, Rect(x, y, sample_size_x, sample_size_y));
                    BluePercent = sum(detected_ROI)[0] / sample_size_x*sample_size_y;
                    if (BluePercent > BlueThreshold)
                    {
                        //Detect which is the right circuit board
                        int tmpScore = detectBoardFeature(sample, roi);
                        //if (detectBoardFeature(sample, roi)) rectangle(result, Rect(x, y, sample_size_x, sample_size_y), Scalar(255, 0, 0), 3);
                        rS.push_back(rectScore(x, y, tmpScore));
                    }
                }
            }
        }

        //Select the correct rectangle from those rectScore
        result = selectRectBoard(result, rS, sample_size_x, sample_size_y);

        return result;
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        cout << "usage: feature_extraction img1 img2" << endl;
        return 1;
    }

    //Read Images 
    Mat sample = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    resize(sample, sample, Size(0, 0), 0.125, 0.125);
    Mat detected = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(sample.data != nullptr && detected.data != nullptr);

    //CBD
    Mat mask_detected = CBD::hsvThreshold(detected);
    Mat result = CBD::detectBoard(sample, detected, mask_detected);

    imshow("Blue", result);
    imwrite("blue.png", result);

    imshow("HSV", mask_detected);
    imwrite("hsv.png", mask_detected);
    waitKey(0);
}