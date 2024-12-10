#include "utils.hpp"
#include <chrono>
#include <CImg.h>
#include <cstdlib>
#include <string>

using std::string;
using namespace cimg_library;

string addSuffixBeforeExtension(const string& file, const string& suffix)
{
	auto dot = file.find_last_of('.');
	return dot == string::npos ? file + suffix : file.substr(0, dot) + suffix + file.substr(dot);
}

void saveCImgAsImage(const string& imagePath, const string& suffix, const CImg<float>& cimg)
{
	const string newFileName = addSuffixBeforeExtension(imagePath, suffix);
	cimg.save_png(newFileName.c_str());
}

//calculate execution time in seconds, or show FPS value
string executionTime(const bool showFps, const double seconds)
{
	std::stringstream ss;
	if (showFps)
		ss << "FPS: " << std::fixed << std::setprecision(2) << 1.0 / seconds << " FPS";
	else
		ss << std::fixed << std::setprecision(6) << seconds << " seconds";
	return ss.str();
}

//exits the program with the provided exit code
void exitProgram(const int exitCode)
{
	std::system("pause");
	std::exit(exitCode);
}

namespace timer
{
	void start()
	{
		startTime = std::chrono::high_resolution_clock::now();
	}
	void end()
	{
		currentTime = std::chrono::high_resolution_clock::now();
	}
	double elapsedSeconds()
	{
		return (double)std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTime).count() / 1000000;
	}
}