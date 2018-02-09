#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>
#include <string>
#include "opencv2/videoio.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <math.h>
#include <iomanip>
using namespace cv;
using namespace std;

long getMatches(const Mat& Car1, const Mat& Car2);
double getPSNR(const Mat& I1, const Mat& I2);
Point GetWrappedPoint(Mat M, const Point& p);
void draw_locations(Mat & img, vector< Rect > & locations, const Scalar & color,string text);


#define VIDEO_FILE_NAME "video2.avi"
#define CASCADE_FILE_NAME "cars3.xml"
#define CASCADE4_FILE_NAME "left-sign.xml"
#define CASCADE5_FILE_NAME "right-sign.xml"

#define CAR_IMAGE "car.png"
#define LEFT_SIGN_IMAGE "left.png"
#define RIGHT_SIGN_IMAGE "right.png"

#define WINDOW_NAME_1 "WINDOW1"
#define WINDOW_NAME_2 "WINDOW2"

int main()
{
    cout << CV_VERSION <<endl;
	VideoCapture cap;
	Mat mFrame, mGray, mCanny, imageROI,mGray1, mGray2, carTrack , mask, IPM_ROI, IPM, IPM_Gray, IPM1, IPM2 ,IPM_Gray2, mFrame2;
	CascadeClassifier cars, traffic_light, stop_sign, pedestrian,sign, sign2;//级联分类器
	vector<Rect> cars_found, traffic_light_found, stop_sign_found ,pedestrian_found ,sign_found, sign_found2, cars_tracking;
    vector<Mat> cars_tracking_img;
    vector<int> car_timer;
    
    

	cars.load(CASCADE_FILE_NAME);//调用cars分类器文件"cars3.xml"
    // traffic_light.load(CASCADE1_FILE_NAME);
    // stop_sign.load(CASCADE2_FILE_NAME);
    // pedestrian.load(CASCADE3_FILE_NAME);
    sign.load(CASCADE4_FILE_NAME);//同上
    sign2.load(CASCADE5_FILE_NAME);
    
	cap.open(0);


    
    double fps = 0;
    int level=0,a=mFrame.rows;
    
    // Number of frames to capture
    int num_frames = 60;//60帧抓取一次
    int started_frames = 0;
    
    // Start and end times
    time_t start, end;//起止时间
    
    // Variable for storing video frames
    Mat frame;
    
    cout << "Capturing " << num_frames << " frames" << endl ;
    
    // Start time
    time(&start);//?


	while (cap.read(mFrame))//videocapture类读取下一帧， mFrame用来承载每一帧图像
	{
        
        
        started_frames++;
        
        if (started_frames==num_frames){
        // End Time
        time(&end);
            
        
        // Time elapsed
        double seconds = difftime (end, start);//返回60帧相差秒数
        cout << "Time taken : " << seconds << " seconds" << endl;
        
        // Calculate frames per second
        fps  = num_frames / seconds;
        
        cout <<"fps : "<<fps<<endl;
        started_frames=0;
        time(&start);
        }
        
        
		// Apply the classifier to the frame
        
        mFrame2 = mFrame.clone();
        imageROI = mFrame(Rect(0,mFrame.rows/2,mFrame.cols,mFrame.rows/2));//选定ROI
        IPM_ROI = imageROI(Rect(0,65,imageROI.cols,(imageROI.rows-65)));
        IPM_ROI = IPM_ROI.clone();
	

		cvtColor(imageROI, mGray, COLOR_BGR2GRAY);//imageROI>>灰度图，mGray承载
        cvtColor(mFrame, mGray2, COLOR_BGR2GRAY); //mGray2承载mFrame形成的灰度图像
        
        //imshow("before", mGray);
        mGray.copyTo(mGray1);
		//equalizeHist(mGray, mGray);
        //imshow("after", mGray);
        
        //cars cascade
		cars.detectMultiScale(mGray, 	cars_found, 1.1, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));//对每一帧用分类器？？
		draw_locations(mFrame, cars_found, Scalar(0, 255, 0),"Car");
        draw_locations(mFrame2, cars_found, Scalar(0, 255, 0),"Car");
        

        
      
		//imshow(WINDOW_NAME_1, mFrame);//车道检测
        imshow(WINDOW_NAME_2, mFrame2);
        

		waitKey(10);
	}

	return 0;
}



//Point GetWrappedPoint(Mat M, const Point& p)
// {
//     cv::Mat_<double> src(3/*rows*/,1 /* cols */);//用一个矩阵保存P点坐标
    
//     src(0,0)=p.x;
 //    src(1,0)=p.y;
//     src(2,0)=1.0;
    
//     cv::Mat_<double> dst = M*src;
//     dst(0,0) /= dst(2,0);
//         dst(1,0) /= dst(2,0);
//     return Point(dst(0,0),dst(1,0));
// }

void draw_locations(Mat & img, vector< Rect > &locations, const Scalar & color, string text)
{

    Mat img1, car, carMask ,carMaskInv,car1,roi1, LeftArrow , LeftMask, RightArrow,RightMask;


    img.copyTo(img1);
    string dis;

	if (!locations.empty())
	{

        double distance= 0;
        
        for( int i = 0 ; i < locations.size() ; ++i){
            
            if (text=="Car"){  //如果是车
                car = imread(CAR_IMAGE);//读车小图标
                carMask = car.clone();
                cvtColor(carMask, carMask, CV_BGR2GRAY);//变灰度图像
                locations[i].y = locations[i].y + img.rows/2; // 移动框
                distance = (0.0397*2)/((locations[i].width)*0.00007);// 2 为车辆平均宽度
                Size size(locations[i].width/1.5, locations[i].height/3);//使小图标随方框变小
                resize(car,car,size, INTER_NEAREST);//resize
                resize(carMask,carMask,size, INTER_NEAREST);
                Mat roi = img.rowRange(locations[i].y-size.height, (locations[i].y+locations[i].height/3)-size.height).colRange(locations[i].x, (locations[i].x  +locations[i].width/1.5));
                bitwise_and(car, roi, car);
                car.setTo(color, carMask);
                add(roi,car,car);
                car.copyTo(img1.rowRange(locations[i].y-size.height, (locations[i].y+locations[i].height/3)-size.height).colRange(locations[i].x, (locations[i].x  +locations[i].width/1.5)));
                
            }
	    
	   
	   
	    
            stringstream stream;
            stream << fixed << setprecision(2) << distance;
            dis = stream.str() + "m";
            rectangle(img,locations[i], color, -1);
        }
        addWeighted(img1, 0.8, img, 0.2, 0, img);
        
        for( int i = 0 ; i < locations.size() ; ++i)//图像上绘制文字(car,distance)
	{
        
            rectangle(img,locations[i],color,1.8);
            
            putText(img, text, Point(locations[i].x+1,locations[i].y+8), FONT_HERSHEY_DUPLEX, 0.3, color, 1);
            putText(img, dis, Point(locations[i].x,locations[i].y+locations[i].height-5), FONT_HERSHEY_DUPLEX, 0.3, Scalar(255, 255, 255), 1);
            
            if (text=="Car"){
                locations[i].y = locations[i].y - img.rows/2; 
            }
        
        }
        
	}
}




