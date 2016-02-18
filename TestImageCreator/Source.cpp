#include "helpers2.h"
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <omp.h>

void img_resize(cv::Mat& src, cv::Mat& dst, int new_width, int new_height);
void rotate(cv::Mat& src, double angle, cv::Mat& dst);
void rot90(cv::Mat &matImage, int rotflag);
std::string NameBeforeExt(std::string const& s);
void RotatedImages(std::string destFoldPath, cv::Mat Im);
void MirrorEffects(std::string destFoldPath, cv::Mat Im);
void SaltAndPepperNoise(std::string destFoldPath, cv::Mat Im);
void GaussianBluring(std::string destFoldPath, cv::Mat Im);
void ContrastHL(std::string destFoldPath, cv::Mat Im);
void ScalesWH(std::string destFoldPath, cv::Mat Im);
void TextAdd(std::string destFoldPath, cv::Mat Im);
void CropIm(std::string destFoldPath, cv::Mat Im, double s);
void JpegComp(std::string destFoldPath, cv::Mat Im, const int JPEG_QUALITY);

cv::RNG rng(0xFFFFFFFF);

int main(int argc, char** argv)
{
	std::string folderPath = argv[1];
	std::string destMainFolder = argv[2];
	std::vector<std::string> fileList;

	GET_DirectoryImages(folderPath.c_str(), fileList);

	omp_set_dynamic(0);     // Explicitly disable dynamic teams
	omp_set_num_threads(8); // Use 4 threads for all consecutive parallel regions
	#pragma omp parallel for
	for (int i = 0; i < fileList.size(); i++)
	{
		std::string fileName = NameBeforeExt(fileList[i]);
		std::string filePath = folderPath + "\\" + fileList[i];
		std::string destFoldPath = destMainFolder + "\\" + fileName;
		PathControl(destFoldPath);
		destFoldPath += "\\" + fileName;
		cv::Mat Im = cv::imread(filePath, cv::IMREAD_COLOR);

		//1.Rotated
		RotatedImages(destFoldPath, Im);
		//2.Mirror Effect
		MirrorEffects(destFoldPath, Im);
		//3.Salt and Pepper Noise
		SaltAndPepperNoise(destFoldPath, Im);
		//4.Gaussian Blur Image
		GaussianBluring(destFoldPath, Im);
		//5.Contrast High and Low
		ContrastHL(destFoldPath, Im);
		//6.Scales Change
		ScalesWH(destFoldPath, Im);
		//7.Text Adding
		TextAdd(destFoldPath, Im);
		//8.Cropped Images
		CropIm(destFoldPath, Im, 0.6);
		CropIm(destFoldPath, Im, 0.7);
		CropIm(destFoldPath, Im, 0.8);
		CropIm(destFoldPath, Im, 0.9);
		//9.JPEG Compression
		JpegComp(destFoldPath, Im, 60);
		JpegComp(destFoldPath, Im, 40);
		JpegComp(destFoldPath, Im, 20);
		JpegComp(destFoldPath, Im, 10);
		JpegComp(destFoldPath, Im, 5);
	}
	return 0;
}

/**
* Rotate an image
*/
//void rotate(cv::Mat& src, double angle, cv::Mat& dst)
//{
//	int len = max(src.cols, src.rows);
//	cv::Point2f pt(len / 2., len / 2.);
//	cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
//	cv::warpAffine(src, dst, r, cv::Size(src.rows, src.cols));
//}



void rot90(cv::Mat &matImage, int rotflag){
	//1=CW, 2=CCW, 3=180
	if (rotflag == 1){
		transpose(matImage, matImage);
		flip(matImage, matImage, 1); //transpose+flip(1)=CW
	}
	else if (rotflag == 2) {
		transpose(matImage, matImage);
		flip(matImage, matImage, 0); //transpose+flip(0)=CCW     
	}
	else if (rotflag == 3){
		flip(matImage, matImage, -1);    //flip(-1)=180          
	}
	else if (rotflag != 0){ //if not 0,1,2,3:
		std::cout << "Unknown rotation flag(" << rotflag << ")" << std::endl;
	}
}

void rotate(cv::Mat& src, double angle, cv::Mat& dst){
	//cout << RANDCOL << "R O T A T I N G" << endlr;
	//int len = std::max(src.cols, src.rows);
	cv::Point2f ptCp(src.cols*0.5, src.rows*0.5);
	//cv::Point2f pt(len/2., len/2.);
	cv::Mat M = cv::getRotationMatrix2D(ptCp, angle, 1.0);
	cv::warpAffine(src, dst, M, src.size(), cv::INTER_CUBIC); //Nearest is too rough, 
}

void img_resize(cv::Mat& src, cv::Mat& dst, int new_width, int new_height)
{
	dst = cv::Mat(cv::Size(new_width, new_height), CV_8UC3);
	cv::resize(src, dst, dst.size(), 0, 0, CV_INTER_LINEAR);
}

std::string NameBeforeExt(std::string const& s)
{
	std::string result = s.substr(0, s.length() - 4);
	return result;
}


void RotatedImages(std::string destFoldPath, cv::Mat Im)
{
	cv::Mat R90, L90, R180;
	R90 = Im.clone();
	R180 = Im.clone();
	L90 = Im.clone();
	rot90(R90, 1);
	rot90(L90, 2);
	rot90(R180, 3);
	cv::imwrite(destFoldPath + "_R90.jpg", R90);
	cv::imwrite(destFoldPath + "_R180.jpg", R180);
	cv::imwrite(destFoldPath + "_L90.jpg", L90);
}

void MirrorEffects(std::string destFoldPath, cv::Mat Im)
{
	cv::Mat mirrorV, mirrorH;
	//cv::Mat mirrorV, mirrorH, mirrorB;
	cv::flip(Im, mirrorV, 0);
	cv::flip(Im, mirrorH, 1);
	//cv::flip(Im, mirrorB, -1);
	cv::imwrite(destFoldPath + "_mirrorV.jpg", mirrorV);
	cv::imwrite(destFoldPath + "_mirrorH.jpg", mirrorH);
	//cv::imwrite(destFoldPath + "_mirrorB.jpg", mirrorB);
}

void SaltAndPepperNoise(std::string destFoldPath, cv::Mat Im)
{
	cv::Mat saltpepper_noise = cv::Mat::zeros(Im.rows, Im.cols, CV_8U);
	randu(saltpepper_noise, 0, 255);

	cv::Mat black = saltpepper_noise < 5;
	cv::Mat white = saltpepper_noise > 250;

	cv::Mat salt_img = Im.clone();
	cv::Mat pepper_img = Im.clone();
	cv::Mat saltpepper_img = Im.clone();
	saltpepper_img.setTo(255, white);
	saltpepper_img.setTo(0, black);
	salt_img.setTo(255, white);
	pepper_img.setTo(0, black);

	cv::imwrite(destFoldPath + "_salt.jpg", salt_img);
	cv::imwrite(destFoldPath + "_pepper.jpg", pepper_img);
	cv::imwrite(destFoldPath + "_saltpepper.jpg", saltpepper_img);
}

void GaussianBluring(std::string destFoldPath, cv::Mat Im)
{
	cv::Mat GBlur, GBlur2, GBlur3;
	cv::GaussianBlur(Im, GBlur, cv::Size(3, 3), 2);
	cv::GaussianBlur(Im, GBlur2, cv::Size(5, 5), 4);
	cv::GaussianBlur(Im, GBlur3, cv::Size(7, 7), 6);
	cv::imwrite(destFoldPath + "_GBlur.jpg", GBlur);
	cv::imwrite(destFoldPath + "_GBlur2.jpg", GBlur2);
	cv::imwrite(destFoldPath + "_GBlur3.jpg", GBlur3);
}

void ContrastHL(std::string destFoldPath, cv::Mat Im)
{
	cv::Mat imgH;
	Im.convertTo(imgH, -1, 0.5, 0); //increase the contrast (double)

	cv::Mat imgL;
	Im.convertTo(imgL, -1, 0.5, 0); //decrease the contrast (halve)

	cv::imwrite(destFoldPath + "_HCont.jpg", imgH);
	cv::imwrite(destFoldPath + "_LCont.jpg", imgL);
}

void ScalesWH(std::string destFoldPath, cv::Mat Im)
{
	cv::Mat Res = cv::Mat(cv::Size(Im.cols * 1.25, Im.rows), CV_8UC3);
	cv::resize(Im, Res, Res.size(), 0, 0, CV_INTER_LINEAR);

	cv::Mat Res2 = cv::Mat(cv::Size(Im.cols, Im.rows * 1.25), CV_8UC3);
	cv::resize(Im, Res2, Res2.size(), 0, 0, CV_INTER_LINEAR);

	cv::imwrite(destFoldPath + "_ScaleW.jpg", Res);
	cv::imwrite(destFoldPath + "_ScaleH.jpg", Res2);
}

void TextAdd(std::string destFoldPath, cv::Mat Im)
{
	cv::Mat textIm = Im.clone();

	cv::putText(textIm, "THIS TEXT IS ADDED TO CREATE IMAGE WITH TEXT.", cvPoint(30, 30),
		cv::FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(200, 0, 0), 2, CV_AA);

	cv::putText(textIm, "THIS TEXT IS ADDED TO CREATE IMAGE WITH TEXT.", cvPoint(30, 60),
		cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cvScalar(0, 0, 200), 1, CV_AA);

	cv::putText(textIm, "THIS TEXT IS ADDED TO CREATE IMAGE WITH TEXT.", cvPoint(30, 70),
		cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cvScalar(0, 0, 0), 1, CV_AA);

	cv::putText(textIm, "THIS TEXT IS ADDED TO CREATE IMAGE WITH TEXT.", cvPoint(30, 80),
		cv::FONT_HERSHEY_COMPLEX_SMALL, 0.3, cvScalar(255, 255, 255), 1, CV_AA);
	cv::imwrite(destFoldPath + "_Text.jpg", textIm);
}

void CropIm(std::string destFoldPath, cv::Mat Im, double s)
{
	int w = Im.rows;
	int h = Im.cols;

	int wStart;
	int hStart;
	
	wStart = rng.uniform(0.0, w*(1-s));
	hStart = rng.uniform(0.0, h*(1-s));

	cv::Rect myROI(hStart, wStart, h*s, w*s);

	cv::Mat croppedImage = Im(myROI);
	cv::imwrite(destFoldPath + "_crop" + int2string(int(s * 100)) + ".jpg", croppedImage);
}

void JpegComp(std::string destFoldPath, cv::Mat Im, const int JPEG_QUALITY)
{
	std::vector<int> params;
	params.push_back(CV_IMWRITE_JPEG_QUALITY);
	params.push_back(JPEG_QUALITY);
	cv::imwrite(destFoldPath + "_J_Comp" + int2string(JPEG_QUALITY) + ".jpg", Im, params);
}
