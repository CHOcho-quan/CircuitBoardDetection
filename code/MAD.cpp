#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    if (argc != 3) {
        cout << "usage: feature_extraction img1 img2" << endl;
        return 1;
    }

    //Read Images 
    Mat sample = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    resize(sample, sample, Size(0, 0), 0.125, 0.125);
    sample = Mat(sample, Rect(0, sample.rows/2, sample.cols, sample.rows/2));
    Mat detected = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(sample.data != nullptr && detected.data != nullptr);

    //Set sample size and parameters
    int sample_size_y = sample.rows, sample_size_x = sample.cols;
    int Downsampling = 1;

    //MAD Algorithm
    Mat detected_tmp, sample_tmp;
    detected_tmp = detected.clone();
    cvtColor(sample, sample_tmp, cv::COLOR_BGR2GRAY);
    cvtColor(detected_tmp, detected_tmp, cv::COLOR_BGR2GRAY);
    imshow("Sample", sample_tmp);
    imshow("Detected", detected_tmp);
    vector<int> total_diff;
    int difference = 0, min_diff = 2147483647, top_x = 0, top_y = 0;

    //Outside Loop : For downsampling and gaussian process to make image pyramid
    for (int i = 0;i < Downsampling;i++)
    {
        if (i) pyrDown(detected, detected_tmp, Size());
        //Inner Loop : For sampling over the whole picture detected
        for (int j = 0;j < detected_tmp.cols - sample_size_x;j++)
        {
            for (int k = 0;k < detected_tmp.rows - sample_size_y;k++)
            {
                Mat detected_ROI = Mat(detected_tmp, Rect(j, k, sample_size_x, sample_size_y)), tmp_r;
                absdiff(sample_tmp, detected_ROI, tmp_r);
                difference = cv::sum(tmp_r)[0] / sample_size_x * sample_size_y;
                //cout << difference << endl;
                if (difference < min_diff) {
                    min_diff = difference;
                    top_y = k;
                    top_x = j;
                }
            }
        }
        total_diff.push_back(difference);
    }
    cout << "Best Difference: " << min_diff << endl;

    //Draw Result Rectangle
    rectangle(detected, Rect(top_x, top_y, sample_size_x,  sample_size_y), Scalar(255, 0, 0), 3);
    imwrite("./detected.png", detected);
    imshow("Result", detected);
    waitKey(0);
}