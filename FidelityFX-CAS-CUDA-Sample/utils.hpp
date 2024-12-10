#pragma once
#define cimg_use_png
#include <chrono>
#include <CImg.h>
#include <string>

std::string addSuffixBeforeExtension(const std::string& file, const std::string& suffix);
void saveCImgAsImage(const std::string& imagePath, const std::string& suffix, const cimg_library::CImg<float>& cimg);
void exitProgram(const int exitCode);
std::string executionTime(const bool showFps, const double seconds);

namespace timer
{
	static std::chrono::time_point<std::chrono::steady_clock> startTime, currentTime;
	void start();
	void end();
	double elapsedSeconds();
}
