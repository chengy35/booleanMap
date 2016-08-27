#include "DenseTrackStab.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"

#include <time.h>

using namespace cv;


#include <iostream>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "BMS.h"


#define MAX_IMG_DIM 400

using namespace cv;
using namespace std;

int show_track = 0; // set show_track = 1, if you want to visualize the trajectories
float threshold_s = 1.5;

Mat getSaliencyMap(Mat src,float *saliency_All)
{
	int sample_step = 8;
	int dilation_width_1 = 3 ;
	int dilation_width_2 = 4;
	float blur_std = 20;
	bool use_normalize = 1;
	bool handle_border = 0;
	int colorSpace = 2;
	bool whitening = 1;
	float max_dimension = 400;  

	Mat src_small;
	char *temp = new char[10];
	float w = (float)src.cols, h = (float)src.rows;
	float maxD = max(w,h);
	if (max_dimension < 0)
		resize(src,src_small,Size((int)(MAX_IMG_DIM*w/maxD),(int)(MAX_IMG_DIM*h/maxD)),0.0,0.0,INTER_AREA);// standard: width: 600 pixel
	else
		resize(src, src_small, Size((int)(max_dimension*w / maxD), (int)(max_dimension*h / maxD)), 0.0, 0.0, INTER_AREA);

	BMS bms(src_small, dilation_width_1, use_normalize, handle_border, colorSpace, whitening);
	bms.computeSaliency((double)sample_step);
	
	Mat result=bms.getSaliencyMap();

	/* Post-processing */

	if (dilation_width_2 > 0)
		dilate(result, result, Mat(), Point(-1, -1), dilation_width_2);
	if (blur_std > 0)
	{
		int blur_width = (int)MIN(floor(blur_std) * 4 + 1, 51);
		GaussianBlur(result, result, Size(blur_width, blur_width), blur_std, blur_std);
	}

	resize(result,result,src.size());

	*saliency_All = getSum(result); // in the last of descriptors.h
	return result;
}

int main(int argc, char** argv)
{
	VideoCapture capture;
	char* video = argv[1];
	int flag = arg_parse(argc, argv);
	capture.open(video);

	if(!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return -1;
	}

	clock_t ttt;
	double avg_time=0;

	
	int frame_num = 0;
	TrackInfo trackInfo;
	DescInfo hogInfo, hofInfo, mbhInfo;
	
	// kernel matrix of densely sampled angles
	CvMat* kernelMatrix = cvCreateMat(3600, 8, CV_32FC1); 
	InitKernelMatrix(kernelMatrix, 5);


	InitTrackInfo(&trackInfo, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);

	SeqInfo seqInfo;
	InitSeqInfo(&seqInfo, video);

	std::vector<Frame> bb_list;
	if(bb_file) {
		LoadBoundBox(bb_file, bb_list);
		assert(bb_list.size() == seqInfo.length);
	}

	if(flag)
		seqInfo.length = end_frame - start_frame + 1;

//	fprintf(stderr, "video size, length: %d, width: %d, height: %d\n", seqInfo.length, seqInfo.width, seqInfo.height);

	if(show_track == 1)
		namedWindow("DenseTrackStab", 0);

	SurfFeatureDetector detector_surf(200);
	SurfDescriptorExtractor extractor_surf(true, true);

	std::vector<Point2f> prev_pts_flow, pts_flow;
	std::vector<Point2f> prev_pts_surf, pts_surf;
	std::vector<Point2f> prev_pts_all, pts_all;

	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
	Mat prev_desc_surf, desc_surf;
	Mat flow, human_mask;

	Mat image, prev_grey, grey;

	std::vector<float> fscales(0);
	std::vector<Size> sizes(0);

	std::vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0), flow_warp_pyr(0),current_pyr(0);
	std::vector<Mat> prev_poly_pyr(0), poly_pyr(0), poly_warp_pyr(0);

	std::vector<std::list<Track> > xyScaleTracks;
	int init_counter = 0; // indicate when to detect new feature points

	/* Computing salient trajectory */
	ttt=clock();


	while(true) {
		Mat frame;
		int i, j, c;

		// get a new frame
		capture >> frame;
		if(frame.empty())
			break;

		if(frame_num < start_frame || frame_num > end_frame) {
			frame_num++;
			continue;
		}

		if(frame_num == start_frame) {
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_grey.create(frame.size(), CV_8UC1);

			InitPry(frame, fscales, sizes);

			BuildPry(sizes, CV_8UC3, current_pyr);
			BuildPry(sizes, CV_8UC1, prev_grey_pyr);
			BuildPry(sizes, CV_8UC1, grey_pyr);
			BuildPry(sizes, CV_32FC2, flow_pyr);
			BuildPry(sizes, CV_32FC2, flow_warp_pyr);

			BuildPry(sizes, CV_32FC(5), prev_poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_warp_pyr);

			xyScaleTracks.resize(scale_num);

			frame.copyTo(image);
			cvtColor(image, prev_grey, CV_BGR2GRAY);

			for(int iScale = 0; iScale < scale_num; iScale++) {
				if(iScale == 0)
				{
					prev_grey.copyTo(prev_grey_pyr[0]);
					image.copyTo(current_pyr[0]);
				}	
				else
				{
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);
					resize(current_pyr[iScale-1], current_pyr[iScale], current_pyr[iScale].size(), 0, 0, INTER_LINEAR);
				}

				// dense sampling feature points
				std::vector<Point2f> points(0);
				DenseSample(prev_grey_pyr[iScale], points, quality, min_distance);

				// save the feature points
				std::list<Track>& tracks = xyScaleTracks[iScale];
				for(i = 0; i < points.size(); i++)
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
			}

			// compute polynomial expansion
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);

			human_mask = Mat::ones(frame.size(), CV_8UC1);
			if(bb_file)
				InitMaskWithBox(human_mask, bb_list[frame_num].BBs);

			detector_surf.detect(prev_grey, prev_kpts_surf, human_mask);
			extractor_surf.compute(prev_grey, prev_kpts_surf, prev_desc_surf);

			frame_num++;
			continue;
		}

		init_counter++;
		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);

		// match surf features
		if(bb_file)
			InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
		detector_surf.detect(grey, kpts_surf, human_mask);
		extractor_surf.compute(grey, kpts_surf, desc_surf);
		ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);

		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

		MatchFromFlow(prev_grey, flow_pyr[0], prev_pts_flow, pts_flow, human_mask);
		MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);

		Mat H = Mat::eye(3, 3, CV_64FC1);
		if(pts_all.size() > 50) {
			std::vector<unsigned char> match_mask;
			Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
			if(countNonZero(Mat(match_mask)) > 25)
				H = temp;
		}

		Mat H_inv = H.inv();
		Mat grey_warp = Mat::zeros(grey.size(), CV_8UC1);
		MyWarpPerspective(prev_grey, grey, grey_warp, H_inv); // warp the second frame

		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey_warp, poly_warp_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_warp_pyr, flow_warp_pyr, 10, 2);

		for(int iScale = 0; iScale < scale_num; iScale++)
		{
			
			/*===========================================================*/
			std::vector<CvPoint2D32f> points_in(0);
			std::list<Track>& tracks = xyScaleTracks[iScale];
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); ++iTrack)
			{
				CvPoint2D32f point = iTrack->point[iTrack->index];
				points_in.push_back(point); // collect all the feature points
			}
			int count = points_in.size();
			/*===========================================================*/


			if(iScale == 0)
			{
				grey.copyTo(grey_pyr[0]);
				image.copyTo(current_pyr[0]);
			}
			else
			{
				resize(grey_pyr[iScale-1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);
				resize(current_pyr[iScale-1], current_pyr[iScale], current_pyr[iScale].size(), 0, 0, INTER_LINEAR);
			}

			int width = grey_pyr[iScale].cols;
			int height = grey_pyr[iScale].rows;

			// compute the integral histograms
			DescMat* hogMat = InitDescMat(height+1, width+1, hogInfo.nBins);
			HogComp(prev_grey_pyr[iScale], hogMat->desc, hogInfo,kernelMatrix);

			DescMat* hofMat = InitDescMat(height+1, width+1, hofInfo.nBins);
			HofComp(flow_warp_pyr[iScale], hofMat->desc, hofInfo,kernelMatrix);

			DescMat* mbhMatX = InitDescMat(height+1, width+1, mbhInfo.nBins);
			DescMat* mbhMatY = InitDescMat(height+1, width+1, mbhInfo.nBins);
			MbhComp(flow_warp_pyr[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo,kernelMatrix);

			/*=======================================================================*/
			float saliency_All = 0;
			Mat salMap = getSaliencyMap(current_pyr[iScale],&saliency_All); 
			std::vector<int> status(count);
			std::vector<CvPoint2D32f> points_out(count);
			std::vector<float> saliency(count);

			// track feature points by median filtering

			OpticalFlowTracker(flow_warp_pyr[iScale], salMap, points_in, points_out, status, saliency);
			/*=======================================================================*/

			// track feature points in each scale separately
			j = 0;
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();++j)
			{
					if( status[j] == 1 )
					{ // if the feature point is successfully tracked
						int index = iTrack->index;
						Point2f prev_point = iTrack->point[index];
						int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
						int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);
						Point2f point;
						point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];
						point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];
						iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];
						iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];
						iTrack->saliency[index] = saliency[j];
						iTrack->averageSaliency[index] = saliency_All;
						
						// get the descriptors for the feature point
						RectInfo rect;
						GetRect(prev_point, rect, width, height, hogInfo);
						GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
						GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
						GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
						GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
						iTrack->addPoint(point);
					}
					else
					{
						iTrack = tracks.erase(iTrack);
						continue;
					}
					// draw the trajectories at the first scale
					if(show_track == 1 && iScale == 0)
					{ //save the saliency image;
						DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);
						//salMap
						char str[20];
						sprintf(str, "saliency/%d.jpg", frame_num);
						imwrite(str,salMap);
					}

					// if the trajectory achieves the maximal length
					if(iTrack->index >= trackInfo.length) 
					{
						std::vector<Point2f> trajectory(trackInfo.length+1);
						std::vector<float> saliency(trackInfo.length+1);
						std::vector<float> averageSaliency(trackInfo.length+1);

						for (int i = 0; i <= trackInfo.length; ++i) {
							trajectory[i] = iTrack->point[i]*fscales[iScale];
							saliency[i] = iTrack->saliency[i];
							averageSaliency[i] = iTrack->averageSaliency[i];
						}
						std::vector<Point2f> displacement(trackInfo.length);
						for (int i = 0; i < trackInfo.length; ++i)
							displacement[i] = iTrack->disp[i]*fscales[iScale];

						float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
						if(( isValid(trajectory, saliency, averageSaliency, threshold_s) == 1 ) && IsCameraMotion(displacement))
						{
							// output the trajectory
							printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t", frame_num, mean_x, mean_y, var_x, var_y, length, fscales[iScale]);

							// for spatio-temporal pyramid
							printf("%f\t", std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999));
							printf("%f\t", std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999));
							printf("%f\t", std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999));
						
							// output the trajectory
							for (int i = 0; i < trackInfo.length; ++i)
								printf("%f\t%f\t", displacement[i].x, displacement[i].y);
			
							PrintDesc(iTrack->hog, hogInfo, trackInfo);
							PrintDesc(iTrack->hof, hofInfo, trackInfo);
							PrintDesc(iTrack->mbhX, mbhInfo, trackInfo);
							PrintDesc(iTrack->mbhY, mbhInfo, trackInfo);
							printf("\n");
						}

						iTrack = tracks.erase(iTrack);
						continue;
					}
					++iTrack;
				}
			ReleDescMat(hogMat);
			ReleDescMat(hofMat);
			ReleDescMat(mbhMatX);
			ReleDescMat(mbhMatY);

			if(init_counter != trackInfo.gap)
				continue;

			// detect new feature points every gap frames
			std::vector<Point2f> points(0);
			for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
				points.push_back(iTrack->point[iTrack->index]);

			DenseSample(grey_pyr[iScale], points, quality, min_distance);
			// save the new feature points
			for(i = 0; i < points.size(); i++)
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
		}

		init_counter = 0;
		grey.copyTo(prev_grey);
		for(i = 0; i < scale_num; i++) {
			grey_pyr[i].copyTo(prev_grey_pyr[i]);
			poly_pyr[i].copyTo(prev_poly_pyr[i]);
		}

		prev_kpts_surf = kpts_surf;
		desc_surf.copyTo(prev_desc_surf);

		frame_num++;

		if( show_track == 1 ) {
			imshow( "DenseTrackStab", image);
			c = cvWaitKey(3);
			if((char)c == 27) break;
		}
	}

	if( show_track == 1 )
		destroyWindow("DenseTrackStab");

	ttt=clock()-ttt;

	float process_time=(float)ttt/CLOCKS_PER_SEC;
	avg_time+=process_time;
	cvReleaseMat(&kernelMatrix);
	cout<<"average_time: "<<avg_time<<endl;

	return 0;
}
