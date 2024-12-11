#define cimg_use_png
#include "CASLibWrapper.h"
#include "utils.hpp"
#include <CImg.h>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <INIReader.h>
#include <iostream>
#include <memory>
#include <string>

using namespace cimg_library;
using std::cout;
using std::string;

/*!
 *  \brief  Main starting point of the FidelityFX-CAS CUDA Sample usage implementation project.
 *          A simple working example of applying CAS on an input image. Provides benchmarks
 *			for measuring total execution time, in order to compare with a CPU implementation.
 *	
 *  \author Dimitris Karatzas
 */
int main() 
{
	double totalExecutionTime = 0, cudaInitExecutionTime = 0;
	//measure cuda context initialization
	timer::start();
	cudaFree(0); 
	timer::end();
	cudaInitExecutionTime += timer::elapsedSeconds();
	totalExecutionTime += timer::elapsedSeconds();

	//load parameters from file
	const INIReader inir("settings.ini");
	if (inir.ParseError() < 0)
	{
		cout << "Could not load configuration file, exiting..";
		exitProgram(EXIT_FAILURE);
	}
	const string imagePath = inir.Get("paths", "image", "NO_IMAGE");
	const float sharpenStrength = inir.GetFloat("parameters", "sharpen_strength", -1);
	const float contrastAdaption = inir.GetFloat("parameters", "contrast_adaption", -1);
	const bool showFps = inir.GetBoolean("options", "execution_time_in_fps", false);
	int loops = inir.GetInteger("parameters", "loops_for_test", 5);
	loops = loops <= 0 ? 5 : loops;

	//load image from disk
	timer::start();
	CImg<unsigned char> diskImage(imagePath.c_str());
	const int rows = diskImage.height();
	const int cols = diskImage.width();
	if (diskImage.spectrum() != 3)
	{
		cout << "Image is not RGB\n";
		exitProgram(EXIT_FAILURE);
	}
	// Add empty "A" channel (RGBA), CUDA cannot work with RGB, padding is required
	diskImage.append(CImg<unsigned char>(cols, rows, 1, 1, 0), 'c');
	//interleave data [R1, G1, B1, 0, R2, B2, G2, 0....]
	diskImage.permute_axes("cxyz");
	timer::end();
	totalExecutionTime += timer::elapsedSeconds();

	if (cols <= 16 || rows <= 16 || rows >= 16384 || cols >= 16384)
	{
		cout << "Image dimensions too low or too high\n";
		exitProgram(EXIT_FAILURE);
	}
	if (sharpenStrength < 0 || sharpenStrength > 1)
	{
		cout << "sharpen_strength should be in the range [0,1]\n";
		exitProgram(EXIT_FAILURE);
	}
	if (contrastAdaption < 0 || contrastAdaption > 1)
	{
		cout << "contrast adaption should be in the range [0,1]\n";
		exitProgram(EXIT_FAILURE);
	}
	cout << "Each test will be executed " << loops << " times. Average time will be shown below\n";
	cout << "Image size is: " << rows << " rows and " << cols << " columns\n\n";
	cout << "Time to initialize CUDA context: " << cudaInitExecutionTime << " seconds\n";
	cout << "Time to load image from disk and initialize CImg object: " << timer::elapsedSeconds() << " seconds\n";

	//initialize CAS algorithm class and execute sharpening
	timer::start();
	void* cas = CAS_initialize(rows, cols);
	timer::end();
	totalExecutionTime += timer::elapsedSeconds();
	cout << "Time to initialize CUDA memory: " << timer::elapsedSeconds() << " seconds\n\n";

	//execute sharpening and instantiate CImg to use the pinned memory returned by CAS
	double copyAndKernelSecs = 0.0;
	std::unique_ptr<CImg<unsigned char>> sharpImage;
	for (int i = 0; i < loops; i++)
	{
		timer::start();
		const unsigned char *sharpedRgbBuffer = CAS_sharpenImage(cas, diskImage.data(), 0, sharpenStrength, contrastAdaption);
		sharpImage = std::make_unique<CImg<unsigned char>>(sharpedRgbBuffer, cols, rows, 1, 3, true);
		timer::end();
		copyAndKernelSecs += timer::elapsedSeconds();
	}
	totalExecutionTime += copyAndKernelSecs / loops;
	cout << "Image sharpening using CAS. Image size is " << rows << " rows and " << cols << " columns and parameters:\nSharpen Strength = " << sharpenStrength << "\nContrast Adaption = " <<contrastAdaption << "\n" << executionTime(showFps, copyAndKernelSecs / loops) << "\n\n";
	cout << "Total execution time: " << totalExecutionTime << " seconds\n";

	//save watermarked images to disk
	if (inir.GetBoolean("options", "save_sharpened_file_to_disk", false))
	{
		cout << "\nSaving watermarked files to disk...\n";
		saveCImgAsImage(imagePath, "_CAS", *sharpImage.get());
		cout << "Successully saved to disk\n";
	}
	CAS_destroy(cas);
	return EXIT_SUCCESS;
}