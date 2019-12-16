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
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    resize(img_1, img_1, Size(0, 0), 0.125, 0.125);
    img_1 = Mat(img_1, Rect(0, img_1.rows/2, img_1.cols, img_1.rows/2));
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    resize(img_2, img_2, Size(0, 0), 0.25, 0.25);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    //Getting Feature Point
    std::vector<KeyPoint> kp1, kp2;
    Mat descriptor1, descriptor2;
    Ptr<Feature2D> sift1 = xfeatures2d::SIFT::create(10);
    Ptr<Feature2D> sift2 = xfeatures2d::SIFT::create(50);
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    sift1->detectAndCompute(img_1, Mat(), kp1, descriptor1);
    sift2->detectAndCompute(img_2, Mat(), kp2, descriptor2);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "extract SIFT cost = " << time_used.count() << " seconds. " << endl;

    //Draw out key points
    // namedWindow("KPImage1", 0);
    // namedWindow("KPImage2", 0);
    // imshow("KPImage1", img_1);
    // imshow("KPImage2", img_2);
    // imshow("d1", descriptor1);
    // imshow("d2", descriptor2);
    // waitKey(0);

    //Hamming Distance, BREIF descriptor
    vector<DMatch> matches;
    BFMatcher matcher = BFMatcher(NORM_L2);
    t1 = chrono::steady_clock::now();
    matcher.match(descriptor1, descriptor2, matches);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "match SIFT cost = " << time_used.count() << " seconds. " << endl;

    //Selecting Matched points
    auto min_max = minmax_element(matches.begin(), matches.end(),
                                    [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

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
    drawMatches(img_1, kp1, img_2, kp2, matches, img_match);
    drawMatches(img_1, kp1, img_2, kp2, good_matches, img_goodmatch);
    imshow("all matches", img_match);
    imshow("good matches", img_goodmatch);
    imwrite("good_matches.png", img_goodmatch);
    waitKey(0);
}
