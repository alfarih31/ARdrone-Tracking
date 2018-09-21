#include "ardrone/ardrone.h"
#include "nms.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/tracking/tracker.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int minHessian = 600;
double kp = 0.1;
double ki = 0.05;
double kd = 0.05;
double blur_t=0, match_t=0;

// Camera Const
Mat distCoefs, cameraMat = Mat(3, 3, CV_32FC1);
double mp = -0.79;
double cp = 3232.83;
double md = -0.33;
double cd = 67.05;
double ms = 0.83;
double cs = 0.33;
double pxd = 3158.16;

// Arguments Flag
bool human = false, gray = false;

// Init Header
void manual_control(ARDrone& ardrone);
void filter_target(Mat &src, Rect &_r);

static void help()
{
    // Instructions
    std::cout << "***************************************" << std::endl;
    std::cout << "*       CV Drone manual control       *" << std::endl;
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

void crop_image(Mat src,  Mat &cropped,  Rect& r)
{
    double cx = r.width/2 + r.x;
    double cy = r.height/2 + r.y;

    double ratio = r.height/r.width;

    r.width =  r.width*0.7;
    r.height =  r.width * ratio;

    r.x = cx - r.width/2;
    r.y = cy - r.height/2;

    cropped = src(r);
}

void crop_image2d(Mat src,  Mat &cropped,  Rect2d &r)
{
    double cx = r.width/2 + r.x;
    double cy = r.height/2 + r.y;

    double ratio = r.height/r.width;

    r.width =  r.width*0.7;
    r.height =  r.width * ratio;

    r.x = cx - r.width/2;
    r.y = cy - r.height/2;

    cropped = src(r);
}

void calculate_distance(double a, double &pdpd, double &pdl, vector<double> &pdr, vector<double> x_target)
{
    // Calculate dpd
        pdpd = (mp + (cp/a))*0.67;
    // Calculate dl
        pdl = (md*a + cd)*0.67;
    // Calculate dr;
        if(human)
        {
            for(int i = 0; i < x_target.size(); i++)
            {
                double cdr = (((640*x_target[i])/a) - cs)/ms;
                pdr.push_back(cdr);
            }
        }
}

void hog_detecting(HOGDescriptor &hog, Mat image, std::vector<Rect>* pfound)
{

    GaussianBlur(image, image,  Size( 3, 3 ), 0, 0 );
    hog.detectMultiScale(image, *pfound, 0,  Size(4,4), Size(32,32), 1.05, 4);
}

void calibrate_camera(ARDrone &ardrone)
{
    Mat image, cropped;
    double distance, c_char;
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
            for(int i=0; i < found.size(); i++)
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
                        c_char = r.width*distance;
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

        double vz = kp * err + ki * int_err + kd * der_err;
        ardrone.move3D(0.0, 0.0, vz, 0.0);
        // Get Image, Display Image, and Draw
            image = ardrone.getImage();
            ostringstream buf;
            buf << alt;
            putText(image, status, Point((image.rows/2),(image.cols/2)), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1, LINE_AA);
            putText(image, buf.str(), Point((image.rows/4),(image.cols/4)), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1, LINE_AA);
            imshow("LIVE", image);

        // Failsafe
            if (c == ' ') ardrone.landing(); //Super Panic
            else if (c == 'm') manual_control(ardrone);
    }
}

void extract_target(Mat target, Mat& dsc)
{
    Mat *_dsc = new Mat;
    vector<KeyPoint> kp;
    try
    {
        if(gray) cvtColor(target, target, COLOR_BGR2GRAY);
        target.convertTo(target, CV_32F);
        Ptr<AKAZE> detector = AKAZE::create();
        detector -> setThreshold(3e-4);
        detector -> detectAndCompute(target, noArray(), kp, *_dsc);
        dsc = *_dsc;
    }
    catch(exception &e)
    {
        cout << "ERR: Extract Target:\n" << e.what() << endl;
        dsc.release();
    }
}

bool image_matching(Mat detected, Mat dsc_target)
{
    vector<vector< DMatch >> matches;
    vector<KeyPoint> keypoints_detected;
    Mat descriptors_detected;

    // Initate for FLANN Matcher
    try
    {
        if(gray) cvtColor(detected, detected, COLOR_BGR2GRAY);
        detected.convertTo(detected, CV_32F);

        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
        Ptr<AKAZE> detector = AKAZE::create();
        detector -> setThreshold(3e-4);

        // CV_32 Type needed for Flann matcher
        detector -> detectAndCompute(detected, noArray(), keypoints_detected, descriptors_detected);

        matcher -> knnMatch(descriptors_detected, dsc_target, matches, 2);
        cout << endl;
        cout << "matches: " << matches.size();
        if(matches.size() <= 2) throw " matches <= 2";
        int n = 0;
        for(size_t i = 0; i < matches.size(); i++) {
            double dist1 = matches[i][0].distance;
            double dist2 = matches[i][1].distance;
            if(dist1 < 0.8 * dist2) n++;
        }
        cout << " n: " << n;
        if(n >= match_t*matches.size()) return true;
        else throw "n < match_t";
    }
    catch(const char* msg)
    {
        cout << "ERR: Image_matching\n" << msg << endl;
        return false;
    }
    catch(exception &e)
    {
        cout << "ERR: Image_matching\n" << e.what() << endl;
        return false;
    }
}

double calculate_blur(Mat image)
{
    Scalar mean, dev;
    if(image.type() != CV_32F) image.convertTo(image, CV_32F);
//    GaussianBlur(image, image, Size (3, 3), 0, 0);
    if(image.empty()) return 0.0;
    cvtColor(image, image, CV_BGR2GRAY);
    Laplacian(image, image, CV_32F);
    meanStdDev(image, mean, dev);
    return double(dev.val[0]*dev.val[0]);
}

void reiniterror(double &e1, double &e2, double &e3, double &e4)
{
    e1 = 0.0, e2 = 0.0, e3 = 0.0, e4 = 0.0;
}

void tracking_target(ARDrone &ardrone, Mat full_image, Mat target, Rect2d target_pos)
{
    // INIT
        Ptr<TrackerMedianFlow> tracker = TrackerMedianFlow::create();
        tracker -> init(full_image, target_pos);

        //Extract Target Features
        Mat descriptors_target;
        extract_target(target, descriptors_target);
        if(descriptors_target.empty()) return;

        //Save initial target, and target background
        Rect2d last_target = target_pos;
        Mat last_full_image, detected_target, ff_image;
        full_image.copyTo(last_full_image);

        bool ok = false, first = true;
        int _ok = 0, n = 0;
        double  delay = 1,
                dpd, dl,
                sum = 0.0,
                dt = 0.0,
                temp_t = 0.0,
                ixe = 0.0,
                perx = 0.0,
                iye = 0.0,
                pery = 0.0;
        vector<double> x_target = {44.0, 46.5, 48.0}, dr;
    while(1)
    {
        char c = (char)waitKey(delay);
        full_image = ardrone.getImage();
        double t = (double)getTickCount();

        undistort(full_image, ff_image, cameraMat, distCoefs);
        imshow("LIVE", ff_image);
        double blurry = calculate_blur(ff_image);
        cout <<"\tTRACKING" << "  |  Blurry: " << blurry << endl;
        if(blurry >= blur_t)
        {
            try
            {
                ok = tracker -> update(ff_image, target_pos);
                if(ok)
                {
                    crop_image2d(ff_image, detected_target, target_pos);
                    bool show = image_matching(detected_target, descriptors_target);
                    if(show)
                    {
                        n++;
                        sum += target_pos.width;
                        if(n >= 0)
                        {
                            //Save last detected image
                            ff_image.copyTo(last_full_image);
                            last_target = target_pos;

                            double avg_width = sum/n;
                            calculate_distance(avg_width, dpd, dl, dr, x_target);

                            // PID Controller
                                //ardrone.move3D(0.0, 0.0, 0.0, 0.0);

                                double dt = ((getTickCount() - temp_t)*1000)/getTickFrequency();
                                temp_t = getTickCount();
                                if(!first)
                                {
                                    double cxt = target_pos.x + target_pos.width/2;
                                    double cyt = target_pos.y + target_pos.height/2;
                                    double erx = 1.0 - cxt/320; //320px is center x vertice of the image
                                    double ery = 1.0 - cyt/180; //180px is center y vertice of the image

                                    cout << "\nerx: " << erx << " ery: " << ery << " dt: " << dt;
                                    ixe += erx * dt;
                                    iye += ery * dt;
                                    double dye = (ery - pery)/dt;
                                    double dxe = (erx - perx)/dt;
                                    double perx = erx;
                                    double pery = ery;
                                    double vr = erx*kp + ixe*ki + dxe*kd;
                                    double vz = ery*kp + iye*ki + dye*kd;
                                    cout << "| vr: " << vr << " vz: " << vz << endl;
                                    ardrone.move3D(0.0, 0.0, vz, vr);
                                }
                                first = false;

                            double dpxd = pxd/avg_width;
                            {
                                ostringstream buf;
                                if(!human) buf  << "Dpd:" << setprecision(4) << dpd
                                                << " cm | Ddl:" << setprecision(4) << dl
                                                << " cm | Dpxd:" << setprecision(4) << dpxd << "cm";
                                else buf    << "D(" << x_target[0] << "): " << dr[0]
                                            << " | D(" << x_target[1] << "): " << dr[1]
                                            << " | D(" << x_target[2] << "): " << dr[2];
                                putText(ff_image, buf.str(), Point(1, ff_image.rows/2), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 0), 1, LINE_AA);
                                dr.clear();
                            }

                            rectangle(ff_image, target_pos, Scalar(0, 255, 0), 1, 1 );
                            imshow("TRACKING", ff_image);
                            n = 0;
                            sum = 0.0;
                        }
                    } else ardrone.move3D(0.0, 0.0, 0.0, 0.0);
                }
                else _ok++;
                if(_ok > 3)
                {
                    cout << "RE-INIT" << endl;
                    Mat last_detected = last_full_image(last_target);
                    extract_target(last_detected, descriptors_target);
                    if(descriptors_target.empty()) return;
                    tracker = TrackerMedianFlow::create();
                    tracker -> init(last_full_image, last_target);
                    _ok = 0;
                    first = true;
                }
            }
            catch (exception)
            {
                _ok++;
                continue;
            }
        }
        t = (((double)getTickCount() - t)*1000)/getTickFrequency();
        delay = 33.0 - t;

        if(delay < 0) delay = 1;
        if(c == 'r') break;
        else if(c == 27) break;
        else if(c == ' ') ardrone.landing();
    }
}

void get_target(ARDrone &ardrone, Mat &target_env, Mat &target, Rect2d &target_pos)
{
    Mat image, flat_image;
    int count = 0;
    double delay = 1;
    bool done = false;
    vector<Rect> found, found_nms;

    // Initiate HOG
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    while(1)
    {
        char c = (char)waitKey(delay);
        double t = (double)getTickCount();
        // Get Image, & Calculate Blur
            image = ardrone.getImage();
            undistort(image, flat_image, cameraMat, distCoefs);
            double blurry = calculate_blur(flat_image);
            cout << endl;
            cout << "GETTING" << "  |  Blurry: " << blurry;

        //HOG Detecting & Processing
            if(blurry >= blur_t) hog_detecting(hog, flat_image, &found);
            cout << "  |  Found object: " << found.size();
            nms(found, found_nms, 0.2f, 1); // NMS
            for(int i = 0; i < found_nms.size(); i++)
            {
                try
                {
                    Rect2d r = found_nms[i];
                    crop_image2d(flat_image, target, r);
                    double w = r.width;
                    double h = r.height;
                    double ratio = h/w;
                    target_env = flat_image;
                    target_pos = r;
                    rectangle(flat_image, r,  Scalar(255, 0, 0), 1, 1);
                    if (ratio >= 1.25 && ratio <= 1.99)// Average Ratio of human body
                    {
                        double target_blur = calculate_blur(target);
                        cout << "  |  Target Blurry: "<< target_blur << endl;
                        if(target_blur*3 >= blur_t)
                        {
                            imshow("TARGET", target);
                            done = true;
                            break;
                        }
                    };
                }
                catch(exception)
                {
                    continue;
                };
            }

        if(done) break;
        else count++;

        if(count > 30)
        {
            ardrone.move(0.0, 0.0, 1.0);
            waitKey(100);
            ardrone.move(0.0, 0.0, 0.0);
            count = 0;
        };
        imshow("LIVE", flat_image);

        found.clear();
        found_nms.clear();

        t = (((double)getTickCount() - t)*1000)/getTickFrequency();
        delay = 33.0 - t;
        if(delay < 0) delay = 1;

        // Failsafe
            if (c == 27)
            {
                target.release();
                break;
            }
            else if(c == ' ') ardrone.landing();
    }
    return;
}

void manual_control(ARDrone& ardrone)
{
    // Initialize Var
    Mat target, full_image, image;
    Rect2d target_pos;
    String status = "manual";
    bool new_key = false, sent = false;
    while(1){
        // Key input
        int key = (int)waitKey(33);
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
                if (key == 'i') vx = 1.0, new_key = true;
                else if (key == 'k') vx = -1.0, new_key = true;
                else if (key == 'u') vr =  1.0, new_key = true; //CCW
                else if (key == 'o') vr = -1.0, new_key = true; //CW
                else if (key == 'j') vy =  1.0, new_key = true;
                else if (key == 'l') vy = -1.0, new_key = true;
                else if (key == 'q') vz =  1.0, new_key = true;
                else if (key == 'a') vz = -1.0, new_key = true;
                else if (key == -1)
                {
                    if(sent) 
                    {
                        cout << "ZERO" << vx << vy << vz <<vr << endl;
                        ardrone.move3D(vx, vy, vz, vr);
                        sent = false;
                        new_key = false;
                    };
                    ardrone.update();
                };

                if(!sent && new_key)
                {
                    ardrone.move3D(vx, vy, vz, vr);
                    cout << "SENT"<< vx << vy << vz <<vr;
                    sent = true;
                };
        // Change camera
            static int mode = 0;
            if (key == 'c') ardrone.setCamera(++mode % 4);

        // Display the image and Draw
            putText(image, status, Point((image.rows/2),(image.cols/2)), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 0, 255), 1, LINE_AA);
            imshow("LIVE", image);

        // Switch Control
            if(key == 's')
            {
                while(1)
                {
                    get_target(ardrone, full_image, target, target_pos);
                    if(!target.empty() && !full_image.empty()) tracking_target(ardrone, full_image, target, target_pos);
                    else target.release(), full_image.release();
                    char k = (char)waitKey(33);
                    if(k == 27) break;
                }
            }
            else if (key == 'n') maintain_alt(ardrone);
            else if (key == 27) break;
    }
    return;
}

int main( int argc, char* argv[] )
{
    cout << "######\tARDRONE MF 6.1\t######\n";
    // Initialize Var
        Mat target, full_image;
        Rect2d target_pos;
        string calibfile = " ";
    // Passing Argument;
        if (argc < 2) return 1;
        for(int i = 1; i < argc; i++)
        {
            string arg=argv[i];
            if (i+1 > argc) return 1;
            if (arg == "-m") match_t = atof(argv[i+1]);
            else if (arg == "-b") blur_t = atof(argv[i+1]);
            else if (arg == "-h" || arg == "--human") human = true;
            else if (arg == "-i") calibfile = argv[i+1];
            else if (arg ==  "-g") gray = true;
        }
    // Initialize DistCoefs, and Camera Mat
        if(calibfile == " ") return 1;
        FileStorage fs;
        fs.open(calibfile, FileStorage::READ);
        fs["distortion_coefficients"] >> distCoefs;
        fs["camera_matrix"] >> cameraMat;

    // Initiate Windows
    namedWindow("Control", 0);

    // Initiate Camera
    ARDrone ardrone;
    if( !ardrone.open() )
    {
        std::cout << "***Could not initialize ARDRone...***\n";
        return -1;
    }
    help();
    cout    << "Received Params:"
            << "\n\tmatch_t: " << match_t
            << "\n\tblur_t: " << blur_t
            << "\n\tGS Proc: " << gray
            << "\n\tHuman: " << human
            << "\n\tCalibFile: " << calibfile << endl;
    Mat test = ardrone.getImage();
    Mat test_undist;
    undistort(test, test_undist, cameraMat, distCoefs);
    imshow("Control", test_undist);
    cout << "\nOriginal Spec:\t" << test.size() << test.type() << endl;
    cout << "\nUndistort Spec:\t" << test_undist.size() << test_undist.type() << endl;

    // Battery
    std::cout << "Battery = " << ardrone.getBatteryPercentage() << "\%" << std::endl;
    //calibrate_camera(ardrone);
    for(;;){
        char c =  (char)waitKey();
        if( c == 27 ) break;
        else if (c == 'm') manual_control(ardrone);
        else if(c == 's')
        {
            while(1)
            {
                get_target(ardrone, full_image, target, target_pos);
                if(!target.empty() && !full_image.empty()) tracking_target(ardrone, full_image, target, target_pos);
                else target.release(), full_image.release();
                char k = (char)waitKey(33);
                if(k == 27) break;
            }
        };
    }
    return 0;
}
