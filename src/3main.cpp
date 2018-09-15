#include "ardrone/ardrone.h"

using namespace cv;
using namespace std;

double c_char = 0;

// Init Header
void manual_control(ARDrone& ardrone, double *pvz);

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

void crop_image( Mat *src,  Mat *cropped,  Rect *r_temp){
     Rect temp = *r_temp;
     Mat src_temp = *src;
    temp.x +=  cvRound(temp.width*0.1);
    temp.width =  cvRound(temp.width*0.8);
    temp.y +=  cvRound(temp.height*0.07);
    temp.height =  cvRound(temp.height*0.8);

    *cropped = src_temp(temp);
}

double calculate_distance(int pixel, double c_char)
{
    double distance = c_char/pixel;
    return distance;
}

void hog_detecting(Mat *pimage, std::vector< Rect>* pfound)
{
     Mat temp;
    double t;

    // Initiate HOG
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    t =  (double)getTickCount();
     GaussianBlur( *pimage, temp,  Size( 3, 3 ), 0, 0 );
    hog.detectMultiScale(temp, *pfound, 0,  Size(8,8),  Size(32,32), 1.05, 2);
    t =  (double)getTickCount() - t;
    printf("Gaussian detection time = %gms\n", t*1000./ getTickFrequency());
}

void calibrate_camera(ARDrone& ardrone)
{
    Mat image, cropped;
    char cmd;
    double distance;

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
        hog_detecting(&image, &found);
        //HOG Processing
        if (found.size() == 1){
            Rect r = found[0];
            crop_image(&image, &cropped, &r);
            imshow("Cropped", cropped);
            cout << "Apakah ini gambarnya?(y/n)\n:";
            cmd = waitKey();
            if (cmd == 'y'){
                cout << "Input Actual Distance:";
                cin >> distance;
                c_char = distance * r.width;
                printf("%.7g\n", c_char);
                break;
            }else if (cmd == 'n') continue;
        }
    }
}

void goto_target(ARDrone& ardrone, Mat target, Rect pos, double c_char)
{
    double distance, vx = 0.0, vy = 0.0, vz = 0.0, vr = 0.0;
    // Target X pos = cols/2
    // Target Y pos = rows/2
    while(1)
    {
        distance = calculate_distance(pos.width, c_char);
        while(1)
        {
            double center_x = (pos.width/2) + pos.x;
            double center_y = (pos.height/2) + pos.y;
            if (center_x < (target.cols/2)) ardrone.move3D(vx, vy, vz, 1.0);
            if (center_x > (target.cols/2)) ardrone.move3D(vx, vy, vz, -1.0);
            if (center_y < (target.rows/2)) ardrone.move3D(vx, vy, 1.0, vr);
            if (center_y > (target.rows/2)) ardrone.move3D(vx, vy, -1.0, vr);
        }
    }
}

double maintain_alt(ARDrone& ardrone, double *pvz)
{
    double vx = 0.0, vy = 0.0, vr = 0.0, alt = 0;
    int tc = 0;
    bool sent = false;
    String status = "maintain";
    while(1)
    {
        Mat image = ardrone.getImage();
        alt = ardrone.getAltitude();
        printf("%.7g\n", alt);
        if (alt < 1)
        {
            if (tc >= 3 && *pvz <= 0.9)
            {
                tc = 0;
                *pvz += 0.1;
                ardrone.move3D(vx, vy, *pvz, vr);
            }
        }
        else if (alt >= 1)
        {
            if (tc >= 3 && *pvz >= -0.9)
            {
                tc = 0;
                *pvz -= 0.1;
                ardrone.move3D(vx, vy, *pvz, vr);
            }
        };

//        if ((1 - alt) <= 0.1 && tc >= 30) break; 

        // Display Image and Draw
            putText(image, status, Point((image.rows/2),(image.cols/2)), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1, LINE_AA);
            imshow("LIVE", image);

        // Failsafe
        char c = (char)waitKey(33);
        if (c == ' ') ardrone.landing(); //Super Panic
        else if (c == 'm') manual_control(ardrone, pvz);
        ++tc;
    }
    return alt;
}

void start_program(ARDrone& ardrone, double *pvz)
{
    Mat image, cropped;
    Rect target_pos;
    double distance, vx = 0.0, vy = 0.0;
    bool done = false;

    while(1){
        // Initial Check
        if (ardrone.onGround()) ardrone.takeoff();
        
        // Get Image
        image = ardrone.getImage();

        // 1. Check Altitude
        double alt = maintain_alt(ardrone, pvz);
        if (alt >= 1.0)
        {
            //HOG Detecting    
            vector<Rect> found;
            hog_detecting(&image, &found);
            cout << "WITH gauss found object=" << found.size() << endl;

            //HOG Processing
            vector<Rect>::const_iterator it;
            for (it = found.begin(); it != found.end(); ++it)
            {
                maintain_alt(ardrone, pvz);

                Rect r = *it;
                double h = r.height;
                double w = r.width;
                double ratio = h/w;
                if (ratio >= 1.71 && ratio <= 1.99) // Average Ratio of human body
                {
                    target_pos = r;
                    crop_image(&image, &cropped, &r);
                    imshow("TARGET", cropped);
                    rectangle(image, r.tl(), r.br(),  Scalar(255, 0, 0), 2);
                    distance = calculate_distance(r.width, c_char);
                    {
                        ostringstream buf;
                        buf << distance << " cm";
                        putText(image, buf.str(), Point(r.x, r.y), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1, LINE_AA);
                    }
                    done = true;
                    break;
                } else 
                {
                    distance = calculate_distance(r.width, c_char);
                    rectangle(image, r.tl(), r.br(),  Scalar(255, 0, 0), 2);
                    {
                        ostringstream buf;
                        buf << distance << " cm";
                        putText(image, buf.str(), Point(r.x, r.y), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1, LINE_AA);
                    }
                    done = true;
                };
            }
            imshow("LIVE", image);
            if (done) goto_target(ardrone, image, target_pos, c_char);
            else if (!done) ardrone.move(vx, vy, 1.0);
            done = false;
        }else if (alt < 1.0) maintain_alt(ardrone, pvz);

        // Failsafe
        char c = (char)waitKey(33);
        if (c == ' ') ardrone.landing(); //Super Panic
        else if (c == 'm') manual_control(ardrone, pvz);;
    }        
    return;
}

void manual_control(ARDrone& ardrone, double *pvz)
{
    Mat image;
    String status = "manual";
    int tick_count, push_count, cmd_sent = 0, z_cmd = 0;
    bool new_key = false, key_pressed = false;

    double vx = 0.0, vy = 0.0, vr = 0.0;
    while(1){
        // Key input
        char key = (char)waitKey(33);
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
            if (key == 'i' || key == CV_VK_UP) vx = 1.0, new_key = true;
            if (key == 'k' || key == CV_VK_DOWN) vx = -1.0, new_key = true;
            if (key == 'u' || key == CV_VK_LEFT) vr =  1.0, new_key = true;
            if (key == 'o' || key == CV_VK_RIGHT) vr = -1.0, new_key = true;
            if (key == 'j') vy =  1.0, new_key = true;
            if (key == 'l') vy = -1.0, new_key = true;
            if (key == 'q' && *pvz <= 0.9)
            {
                *pvz +=  0.1;
                cout << "Z speed :\n" << *pvz << endl;
                key_pressed = true;
                ++push_count;
            };
            if (key == 'a' && *pvz >= -0.9)
            {
                *pvz -= 0.1;
                cout << "Z speed :" << *pvz << endl;
                key_pressed = true;
                ++push_count;
            };
            // Reset Z movement
            if (key == 'r') *pvz = 0;

        // Anti Multiple Sequence
            if(key_pressed)
            {
                ++tick_count;
                if ((tick_count - push_count) >= 30)
                {
                    cout << "Sent Z Speed:\n\t" << *pvz << endl;
                    ardrone.move3D(vx, vy, *pvz, vr);
                    z_cmd += 1;
                    key_pressed = false;
                    push_count = 0, tick_count = 0;
                }
            }
            else if (!key_pressed && z_cmd > 0)
            {
                ardrone.move3D(0, 0, 0, 0);
                cout << "3D normalized\n";
                z_cmd -= 1;
            };

            if (new_key)
            {
                ardrone.move(vx, vy, vr);
                new_key = false;
                vx = 0; vy = 0; vr = 0;
                cout << "Command sent\n";
                cmd_sent += 1;
            } 
            else if(!new_key && cmd_sent > 0)
            {
                ardrone.move(0, 0, 0);
                cout << "2D normalized\n";
                cmd_sent -= 1;
            };

        // Change camera
            static int mode = 0;
            if (key == 'c') ardrone.setCamera(++mode % 4);

        // Display the image and Draw
            putText(image, status, Point((image.rows/2),(image.cols/2)), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1, LINE_AA);
            imshow("LIVE", image);

        // Switch Control
            if (key == 's') start_program(ardrone, pvz);
            else if (key == 'n') maintain_alt(ardrone, pvz);
            else if (key == 27) break;
    }
    return;
}

int main( int argc, const char** argv )
{
    // Initialize variable
    Mat blank;
    double vz = 0;

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
        else if (c == 'm' && c_char != 0) manual_control(ardrone, &vz);
        else if (c == 's' && c_char != 0) start_program(ardrone, &vz);
    }
    return 0;
}
