// labosi.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cmath>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <fstream>

#include <iostream>
#include <fstream>
#include <array>
using namespace std;
using namespace cv;

vector<Point> imagePoints;
bool cam = true;
bool roiTaken = false;
Mat frame, img1, roi;

void onMouse(int event, int x, int y, int flags, void* param) {

	if (event == CV_EVENT_LBUTTONDOWN) {
		imagePoints.clear();
		imagePoints.push_back(Point(x, y));
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		imagePoints.push_back(Point(x, y));
		roi = img1(Rect(imagePoints.at(0), imagePoints.at(1)));
		imshow("roi", roi);
		roiTaken = true;

		cout << "Roi Selected! Press 'c' for calculation. " << endl;
	}
}

/** Inicijalno sparivanje, model se sparije sa scenom, svih 128 brojave se zborji i usporeðuje */
//array<vector<Point2f>, 2> 

vector<DMatch> L2paring(Mat modelDesc, Mat sceneDesc)
{
	//SPORIJE RADI, DISTANCE JE U PREVELIKO I U FLOAT. treba popraviti racunanje distance varijable
	vector<DMatch> matches;
	int M = sceneDesc.rows;
	int N = sceneDesc.cols;

	Mat differenceOfMats(M, N, DataType<double>::type, double(0));
	Mat l2Norm(M, 1, DataType<double>::type, double(0));

	/* rows == broj modekeysa */
	for (int row = 0; row < modelDesc.rows; ++row)
	{

		for (int rowS = 0; rowS < sceneDesc.rows; ++rowS)
		{
			uchar* ps = sceneDesc.ptr(rowS);
			uchar* pm = modelDesc.ptr(row); //pokazivac na prvi broj u nizu

			for (int colsS = 0; colsS < sceneDesc.cols; ++colsS)
			{
				differenceOfMats.at<double>(rowS, colsS) = *ps++ - *pm++;
			}
		}

		/* L2 norma */

		for (int rowDOM = 0; rowDOM < differenceOfMats.rows; ++rowDOM)
		{
			int *pdom = differenceOfMats.ptr<int>(rowDOM);

			for (int colDOM = 0; colDOM < differenceOfMats.cols; ++colDOM)
			{
				l2Norm.at<double>(rowDOM) += pow(*pdom++, 2); //provjeriti
			}
			l2Norm.at<double>(rowDOM) = sqrt(l2Norm.at<double>(rowDOM));
		}

		/* naði najmanjega u L2 normi */
		float min = l2Norm.at<double>(0, 0);
		int pair = 0;
		for (int rowL2 = 0; rowL2 < l2Norm.rows; ++rowL2)
		{
			double* pl2 = l2Norm.ptr<double>(rowL2);
			if (*pl2 < min)
			{
				min = *pl2;
				pair = rowL2;
			}
		}

		matches.push_back(DMatch(row, pair, l2Norm.at<double>(pair)));
	}

	differenceOfMats.release();
	l2Norm.release();

	return matches;
}

/* rezultat koordiante modela na sceni */
void ransac(vector<DMatch> matches, vector<KeyPoint> modelKeypoints, vector<KeyPoint> sceneKeypoints, Mat* ita, double* alf, double* sig, vector<DMatch> *ransacMatches)
{
	vector<DMatch> bestMatch, tempBestMatches;

	int size = matches.size(), r1, r2;
	int nOfIteration = round(log(1 - 0.99)/log(1 - pow( 1 - 0.7, 2))); //0.99 - željena vjerojatnost odabira ispravnih parova; 0.7 - procjenjeni postotak neispravnih parova u matches
	double epsilon = 2; //koliko piksela je pomaknuto
	
	Mat mj(2, 1, DataType<double>::type), mi(2, 1, DataType<double>::type), mk(2, 1, DataType<double>::type), ml(2, 1, DataType<double>::type);
	double alfa, sigma, sigmaStar;

	Mat cd2(2, 1, DataType<double>::type), cd4(2, 2, DataType<double>::type);
	Mat matDeltaUV(2, 2, DataType<double>::type);
	Mat vecDeltaUV(2, 1, DataType<double>::type);

	Mat ItA(1, 2, DataType<double>::type);
	Mat Ralfa(2, 2, DataType<double>::type);
	Mat wstar(1, 3, DataType<double>::type);


	int model1, model2;
	int scena1, scena2;
	if (size > 0)
	{
		for (int i = 0; i < nOfIteration; i++)
		{
		
			//parovi model - scena, definirani brojem reda u deskriptoru
			do{
				r1 = rand() % size;
				r2 = rand() % size;
			} while (r1 == r2);
			

			model1 = matches.at(r1).queryIdx;
			scena1 = matches.at(r1).trainIdx;

			model2 = matches.at(r2).queryIdx;
			scena2 = matches.at(r2).trainIdx;


			// prvi par - i model ,j scena
			mi.at<double>(0, 0) = modelKeypoints.at(model1).pt.x;
			mi.at<double>(1, 0) = modelKeypoints.at(model1).pt.y;

			mj.at<double>(0, 0) = sceneKeypoints.at(scena1).pt.x;
			mj.at<double>(1, 0) = sceneKeypoints.at(scena1).pt.y;

			// drugi par - k,l
			mk.at<double>(0, 0) = modelKeypoints.at(model2).pt.x;
			mk.at<double>(1, 0) = modelKeypoints.at(model2).pt.y;

			ml.at<double>(0, 0) = sceneKeypoints.at(scena2).pt.x;
			ml.at<double>(1, 0) = sceneKeypoints.at(scena2).pt.y;

			if (abs(norm(mi - mk) - norm(mj - ml)) <= epsilon)
			{

				vecDeltaUV = ml - mj;
				matDeltaUV.col(0) = mk - mi;
				matDeltaUV.at<double>(0, 1) = -matDeltaUV.at<double>(1, 0);
				matDeltaUV.at<double>(1, 1) = matDeltaUV.at<double>(0, 0);

				cd4.col(0) = matDeltaUV.inv()*vecDeltaUV;
				cd4.at<double>(0, 1) = -cd4.at<double>(1, 0);
				cd4.at<double>(1, 1) = cd4.at<double>(0, 0);

				cd2 = cd4.col(0);
				alfa = atan2(cd2.at<double>(1, 0), cd2.at<double>(0, 0));
				sigma = sqrtl(powl(cd2.at<double>(1, 0), 2) + powl(cd2.at<double>(0, 0), 2));

				ItA = mj - cd4*mi;

				Ralfa.at<double>(0, 0) = cosh(alfa);
				Ralfa.at<double>(1, 0) = sinh(alfa);
				Ralfa.at<double>(0, 1) = -Ralfa.at<double>(1, 0);
				Ralfa.at<double>(1, 1) = Ralfa.at<double>(0, 0);

				// provjerava tocnost sparenih tocaka i odbacije loše sparene
				for (int i = 0; i < size; i++)
				{
					int modeli = matches.at(i).queryIdx;
					int scenei = matches.at(i).trainIdx;
					mi.at<double>(0, 0) = modelKeypoints.at(modeli).pt.x;
					mi.at<double>(1, 0) = modelKeypoints.at(modeli).pt.y;

					mj.at<double>(0, 0) = sceneKeypoints.at(scenei).pt.x;
					mj.at<double>(1, 0) = sceneKeypoints.at(scenei).pt.y;

					if (norm(mj - (sigma*Ralfa*mi + ItA)) <= epsilon)
					{
						tempBestMatches.push_back(matches.at(i));
					}
					
				}

				if (tempBestMatches.size() > bestMatch.size())
				{
					bestMatch = tempBestMatches;
					tempBestMatches.clear();

					*ita = ItA;
					*alf = alfa;
					*sig = sigma;
				}
			}
		}
	}
	
	*ransacMatches = bestMatch;

	cd2.release();
	cd4.release();
	matDeltaUV.release();
	vecDeltaUV.release();

	ItA.release();
	Ralfa.release();
	wstar.release();

	mi.release();
	mj.release();
	ml.release();
	mk.release();
}

int _tmain(int argc, _TCHAR* argv[])
{
	BFMatcher bfMatcher(NORM_L2); /* brute force Matcher */
	SiftFeatureDetector detector;
	SiftDescriptorExtractor extractor;

	Mat modelDescriptors, sceneDescriptors;
	vector<KeyPoint> modelKeypoints, sceneKeypoints;
	vector<DMatch> matches, matchesL2pair, ransacMatches;
	Mat ItA(1, 2, DataType<double>::type);
	double alfa, sigma;

	Mat showTask, imgMatches;

	char c;
	VideoCapture cap(1);
	bool calculation = false;
	int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);


	if (!cap.isOpened())
		return -1;

	namedWindow("Frame Image");
	setMouseCallback("Frame Image", onMouse, NULL);

	namedWindow("Cam");

	cout << "Press 'c' (capture) and take a frame!" << endl;

	while (cam)
	{
		cap >> frame;

		c = waitKey(1);
		if (c == 'c' && !roiTaken) //c kao capture 
		{
			roiTaken = false;
			calculation = false;

			frame.copyTo(img1);
			imshow("Frame Image", frame);
			cout << "Select Roi!" << endl;
		}
		else if (c == 'c' && roiTaken) //kao calculate
		{
			frame.copyTo(showTask);
			/* detektira keypointse*/
			detector.detect(showTask, sceneKeypoints);
			detector.detect(roi, modelKeypoints);
			
			/* trazenje deskriptora za keypointse */
			extractor.compute(roi, modelKeypoints, modelDescriptors);
			extractor.compute(showTask, sceneKeypoints, sceneDescriptors);

			//matchesL2pair = L2paring(modelDescriptors, sceneDescriptors);
			bfMatcher.match(modelDescriptors, sceneDescriptors, matchesL2pair); // pronalazi parove
			ransac(matchesL2pair, modelKeypoints, sceneKeypoints, &ItA, &alfa, &sigma, &ransacMatches);

			// varijable za preslikavanje model koordinata u scenu
			if (ransacMatches.size() > 0)
			{
				Mat ralfa(2, 2, DataType<double>::type);
				Mat p1(2, 1, DataType<double>::type), p2(2, 1, DataType<double>::type);

				ralfa.at<double>(0, 0) = cosh(alfa);
				ralfa.at<double>(1, 0) = sinh(alfa);
				ralfa.at<double>(0, 1) = -ralfa.at<double>(1, 0);
				ralfa.at<double>(1, 1) = ralfa.at<double>(0, 0);

				p1.at<double>(0, 0) = 0;
				p1.at<double>(1, 0) = 0;
				p2.at<double>(0, 0) = imagePoints.at(1).x - imagePoints.at(0).x;
				p2.at<double>(1, 0) = imagePoints.at(1).y - imagePoints.at(0).y;

				Mat r1 = sigma*ralfa*p1 + ItA;
				Mat r2 = sigma*ralfa*p2 + ItA;

				rectangle(showTask,
					Rect(Point(r1.at<double>(0, 0), r1.at<double>(1, 0)),
					Point(r2.at<double>(0, 0), r2.at<double>(1, 0))),
					Scalar(0, 0, 255),
					1);

				/* crtanje sparenih tocaka sa ransacom */
				drawMatches(roi, modelKeypoints, showTask, sceneKeypoints, ransacMatches, imgMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
				imshow("RANSAC", imgMatches);

				calculation = true;
				cout << "Calculation finished! \nPress 'B' for task b) or 'C' for task c).\nPress 'c' and try again.\n" << endl;
			}
			else 
			{
				cout << "Can't find matches!" << endl;
			}
			
			
		}
		else if (c == 'B' && calculation)
		{
			cout << "Task b." << endl;

			/* crtanje sparenih tocaka bez ransaca */
			drawMatches(roi, modelKeypoints, showTask, sceneKeypoints, matchesL2pair, imgMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			imshow("B", imgMatches);

			roiTaken = false;
		}

		imshow("Cam", frame);
	}

	waitKey(0);
	return 0;
}

