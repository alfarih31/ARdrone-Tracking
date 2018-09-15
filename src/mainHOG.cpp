#include "ardrone/ardrone.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/tracking/tracker.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

double blur_t=0, match_t=0;

// Init Header
void manual_control(ARDrone& ardrone);

static void help()
{
    // Instructions
    std::cout << "***************************************" << std::endl;
    std::cout << "*       CV Drone sample program       *" << std::endl;
    std::cout << "*           - How to play -           *" << std::endl;
    std::cout << "***************************************" << std::endl;
    std::cout << "*                                     *" << std::endl;
    std::cout << "* - Controls -                        *" << std::endl;
    std::cout << "*    'Space' -- Takeoff/Landing       *" << std::endl;
    std::cout << "*    'Up'    -- Move forward          *" << std::endl;
    std::cout << "*    'Down'  -- Move backward         *" << std::endl;
    std::cout << "*    'Left'  -- Turn left             *" << std::endl;
    std::cout << "*    'Right' -- Turn right            *" << std::endl;
    std::cout << "*    'Q'     -- Move upward           *" << std::endl;
    std::cout << "*    'A'     -- Move downward         *" << std::endl;
    std::cout << "*                                     *" << std::endl;
    std::cout << "* - Others -                          *" << std::endl;
    std::cout << "*    'C'     -- Change camera         *" << std::endl;
    std::cout << "*    'Esc'   -- Exit                  *" << std::endl;
    std::cout << "*                                     *" << std::endl;
    std::cout << "***************************************" << std::endl;
}

void crop_image(Mat src,  Mat &cropped,  Rect r)
{
    r.x +=  r.width*0.1;
    r.width =  r.width*0.8;
    r.y +=  r.height*0.07;
    r.height =  r.height*0.8;

    cropped = src(r);
}

double calculate_distance(int pixel)
{
    double distance = c_char/pixel;
    return distance;
}

void hog_detecting(HOGDescriptor &hog, Mat image, std::vector<Rect>* pfound)
{
    double t;
    GaussianBlur(image, image,  Size( 3, 3 ), 0, 0 );
    hog.detectMultiScale(image, *pfound, 0,  Size(4,4), Size(32,32), 1.05, 4);
}

void calibrate_camera(ARDrone &ardrone)
{
    Mat image, cropped;
    double distance;
    bool done = false;

    // Initiate HOG
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    cout << "INPUT CAMERA CONST :\n(input \'0\' if unknown)" << endl;
    cin >> c_char;
    if(c_char != 0) return;
    while(1)
    {
        image = ardrone.getImage();
        if( image.empty() ) break;
        //HOG Detecting
        vector<Rect> found;
        hog_detecting(hog, image, &found);
        //HOG Processing
        if (found.size() >= 1)
        {
            for(int i=0; i <= found.size(); i++)
            {
                Rect r = found[i];
                double w = r.width;
                double h = r.height;
                double ratio = h/w;
                if (ratio >= 1.61 && ratio <= 1.99) // Average Ratio of human body
                {
                    if(0 <= r.x &&
                        0 <= r.width &&
                        r.x + r.width <= image.cols &&
                        0 <= r.y &&
                        0 <= r.height &&
                        r.y + r.height <= image.rows) crop_image(image, cropped, r);
                    if(cropped.rows > 0 && cropped.cols > 0) imshow("TARGET", cropped);
                    cout << "Confirmed? y/n" << endl;
                    char cmd = (char)waitKey();
                    if(cmd == 'y')
                    {
                        cout << "Actual Distance (cm):" << endl;
                        cin >> distance;
                        c_char = r.width*0.8*distance;
                        printf("%.7g\n", c_char);
                        done = true;
                        break;
                    }
                }
            }
        }
        if(done) break;
    }
}

void maintain_alt(ARDrone& ardrone)
{
    String status = "maintain";
    double int_err = 0.0, prev_err = 0.0, temp_t = 0.0;
    int tc = 0;
    Mat image;
    while(1)
    {
        char c = (char)waitKey(33);
        double alt = ardrone.getAltitude();

        double dt = (getTickCount() - temp_t)/getTickFrequency();
        temp_t = getTickCount();
        double err = 1.0 - alt;

        int_err += err * dt;
        double der_err = (err - prev_err)/dt;
        double prev_err = err;

        if (fabs(err <= 0.01)) break;
        double vz = kp * err + ki * int_err + kd * der_err;
        ardrone.move3D(0.0, 0.0, vz, 0.0);
        // Get Image, Display Image, and Draw
            image = ardrone.getImage();
            putText(image, status, Point((image.rows/2),(image.cols/2)), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1, LINE_AA);
            imshow("LIVE", image);
        
        // Failsafe
            if (c == ' ') ardrone.landing(); //Super Panic
            else if (c == 'm') manual_control(ardrone);
    }
}

bool SURF_matching(Mat detected, Mat target)
{
    vector< DMatch > matches;
    vector<KeyPoint> keypoints_target, keypoints_detected;
    Mat descriptors_target, descriptors_detected;

    // Initate for FLANN Matcher
    if(detected.empty() || target.empty()) return false;
    cvtColor(target, target, CV_BGR2GRAY);
    cvtColor(detected, detected, CV_BGR2GRAY);

    FlannBasedMatcher matcher;
    Ptr<SURF> detector = SURF::create();
    detector -> setHessianThreshold(minHessian);

    detector -> detectAndCompute(target, Mat(), keypoints_target, descriptors_target);
    detector -> detectAndCompute(detected, Mat(), keypoints_detected, descriptors_detected);
    // CV_32 Type needed for Flann matcher
    if(descriptors_target.type() != CV_32F) descriptors_target.convertTo(descriptors_target, CV_32F);
    if(descriptors_detected.type() != CV_32F) descriptors_detected.convertTo(descriptors_detected, CV_32F);

    if(!descriptors_target.empty() && !descriptors_detected.empty()) matcher.match(descriptors_detected, descriptors_target, matches, 2);
    else return false;

    double min_dist = 100;
    for( int i = 0; i < matches.size(); i++ )
    {
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
    };

    int n = 0;
    for(int i = 0; i < matches.size(); i++)
    {
        double dist = matches[i].distance;
        if(dist <= 2*min_dist) n++;
    };

    // 25% Match;
    cout << "n:" << n << " matches: " << matches.size() << endl;
    if(n >= (match_t*matches.size())) return true;
    else return false;

}

double calculate_blur(Mat image)
{
    Scalar mean, dev;
    image.convertTo(image, CV_32F);
//    GaussianBlur(image, image, Size (3, 3), 0, 0);
    if(image.empty()) return 0.0;
    cvtColor(image, image, CV_BGR2GRAY);
    Laplacian(image, image, CV_32F);
    meanStdDev(image, mean, dev);
    return double(dev.val[0]*dev.val[0]);
}

void tracking_target(ARDrone &ardrone, Mat full_image, Mat target, Rect2d target_pos)
{
    // INIT
    Mat detected_target;
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    vector<Rect> found;

    while(1)
    {
        char c = (char)waitKey(5);
        full_image = ardrone.getImage();
        imshow("LIVE", full_image);

        double blurry = calculate_blur(full_image);
        if(blurry >= blur_t)
        {
            hog_detecting(hog, full_image, &found);
            cout <<"TRACKING" << "  |  Blurry: " << blurry << "  |  Found object: " << found.size() << endl;
            //HOG Processing
            for (int i = 0; i != found.size(); ++i)
            {
                Rect r = found[i];
                double h = r.height;
                double w = r.width;
                double ratio = h/w;
                if (ratio >= 1.61 && ratio <= 1.99)
                {
                    if( 0 <= r.x &&
                        0 <= r.width &&
                        r.x + r.width <= full_image.cols &&
                        0 <= r.y &&
                        0 <= r.height &&
                        r.y + r.height <= full_image.rows) crop_image(full_image, detected_target, r);
                    bool ok = SURF_matching(detected_target, target);
                    if(ok)
                    {
                        double distance = calculate_distance(r.width*0.8);
                        {
                            ostringstream buf;
                            buf << distance << " cm";
                            putText(full_image, buf.str(), Point(r.x, r.y), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 1, LINE_AA);
                        }
                        rectangle(full_image, r, Scalar(0, 255, 0), 2, 1 );
                        imshow("CURRENT_TARGET", detected_target);
                        imshow("TRACKING", full_image);
                        break;
                    }
                }
            }
        }
        found.clear();

        if(c == 'r') break;
    }
}

void get_target(ARDrone &ardrone, Mat &target_env, Mat &target, Rect2d &target_pos)
{
    Mat image, ff_image;
    double distance, delay =2.0;
    bool done = false;
    vector<Rect> found;

    // Initiate HOG
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    while(1){
        char c = (char)waitKey(delay);
        double t = (double)getTickCount();

        // Get Image
        image = ardrone.getImage();

        // Calcluate blur level
        double blurry = calculate_blur(image);
        cout << endl;
        cout << "GETTING" << "  |  Blurry: " << blurry;
        
        //HOG Detecting
        if(blurry >= blur_t) hog_detecting(hog, image, &found);
        cout << "  |  Found object: " << found.size();
        //HOG Processing
        if(found.size() != 0)
        {
            for (int i = 0; i <= found.size(); ++i)
            {
                Rect r = found[i];
                double h = r.height;
                double w = r.width;
                double ratio = h/w;
                if (ratio >= 1.71 && ratio <= 1.99) // Average Ratio of human body
                {
                    if(0 <= r.x &&
                        0 <= r.width &&
                        r.x + r.width <= image.cols &&
                        0 <= r.y &&
                        0 <= r.height &&
                        r.y + r.height <= image.rows) crop_image(image, target, r);
                    if(target.rows > 0 && target.cols > 0) imshow("TARGET", target);
                    double target_blur = calculate_blur(target);
                    cout << "  |  Target Blurry: "<< target_blur << endl;
                    if(target_blur*3 >= blur_t)
                    {
                        rectangle(image, r.tl(), r.br(),  Scalar(255, 0, 0), 2);
                        distance = calculate_distance(r.width);
                        {
                            ostringstream buf;
                            buf << distance << " cm";
                            putText(target, buf.str(), Point(target.rows/2, 0), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 0, 255), 1, LINE_AA);
                        }

                        target_env = image;
                        target_pos = r;
                        done = true;
                        break;
                    }
                };
            }
        }

        imshow("LIVE", image);

        if(done) break;

        found.clear();

        t = (((double)getTickCount() - t)*1000)/getTickFrequency();
        delay = 33.0 - t;
        if(delay < 0) delay = 2;

        // Failsafe
        if (c == 27)
        {
            target.release();
            break;
        };
    }        
    return;
}

void manual_control(ARDrone& ardrone)
{
    // Initialize Var
    Mat target, full_image, image;
    Rect2d target_pos;
    String status = "manual";
    bool new_key = false;

    while(1){
        // Key input
        int key = waitKey(33);
        if (key == 0x1b) break;

        // Get Image
        image = ardrone.getImage();
        
        // Hover and Takeof
            // Take off / Landing 
            if (key == ' ') {
                if (ardrone.onGround()) ardrone.takeoff();
                else ardrone.landing();
            };
            // Hover movement
                double vx = 0.0, vy = 0.0, vr = 0.0, vz = 0.0;
                if (key == 'i' || key == CV_VK_UP) vx = 1.0;
                if (key == 'k' || key == CV_VK_DOWN) vx = -1.0;
                if (key == 'u' || key == CV_VK_LEFT) vr =  1.0;
                if (key == 'o' || key == CV_VK_RIGHT) vr = -1.0;
                if (key == 'j') vy =  1.0;
                if (key == 'l') vy = -1.0;
                if (key == 'q') vz =  1.0;
                if (key == 'a') vz = -1.0;
                ardrone.move3D(vx, vy, vz, vr);

        // Change camera
            static int mode = 0;
            if (key == 'c') ardrone.setCamera(++mode % 4);

        // Display the image and Draw
            putText(image, status, Point((image.rows/2),(image.cols/2)), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1, LINE_AA);
            imshow("LIVE", image);

        // Switch Control
            if(key == 's')
            {
                get_target(ardrone, full_image, target, target_pos);
                if(!target.empty() && !full_image.empty()) tracking_target(ardrone, full_image, target, target_pos);
                else target.release(), full_image.release();
            }
            else if (key == 'n') maintain_alt(ardrone);
            else if (key == 27) break;
    }
    return;
}

int main( int argc, const char** argv )
{
    // Initialize Var
    Mat target, full_image;
    Rect2d target_pos;

    cout << argv[1] << endl;
    if (argc < 2) return 1;
    for(int i = 1; i < argc; i++)
    {
        string arg=argv[i];
        if (i+1 > argc) return 1;
        if (arg == "-c") c_char = atof(argv[i+1]);
        else if (arg == "-m") match_t = atof(argv[i+1]);
        else if (arg == "-b") blur_t = atof(argv[i+1]);
    }
    cout << "Received Params:\n" << "c_char: " << c_char << "  |  match_t: " << match_t << "  |  blur_t: " << blur_t << endl; 
    
    // Initiate Windows
    help();
    namedWindow("Control", 0);

    // Initiate Camera
    ARDrone ardrone;
    if( !ardrone.open() )
    {
        std::cout << "***Could not initialize capturing...***\n";
        std::cout << "Current parameter's value: \n";
        return -1;
    }

    // Battery
    std::cout << "Battery = " << ardrone.getBatteryPercentage() << "\%" << std::endl;
    calibrate_camera(ardrone);
    for(;;){
        char c =  (char)waitKey();
        if( c == 27 ) break;
        else if (c == 'm' && c_char != 0) manual_control(ardrone);
        else if(c == 's')
        {
            get_target(ardrone, full_image, target, target_pos);
            if(!target.empty() && !full_image.empty()) tracking_target(ardrone, full_image, target, target_pos);
            else target.release(), full_image.release();
        };
    }
    return 0;
}
