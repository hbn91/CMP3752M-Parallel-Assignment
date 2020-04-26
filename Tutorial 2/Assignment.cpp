#include <iostream>
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

int main()
{
	int platform_id = 0;
	int device_id = 0;

	try
	{
		cl::Context context = GetContext(platform_id, device_id);

		std::cout << "Platform: " << GetPlatformName(platform_id) << "\nDevice: " << GetDeviceName(platform_id, device_id) << std::endl;

		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		// Build and debug kernel code
		try
		{
			program.build();
		}

		catch (const cl::Error& err)
		{
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		std::string image_file = "test.pgm";
		std::string image_fileOut = "test_out.pgm";

		CImg<unsigned char> image_input(image_file.c_str());
		CImg<unsigned char> image_output(image_fileOut.c_str());

		vector<int> H(256, 0), CH(256, 0), LUT(256, 0);

		const size_t H_size = H.size() * sizeof(int);
		const size_t CH_size = CH.size() * sizeof(int);
		const size_t LUT_size = LUT.size() * sizeof(int);
		const size_t LUT_elem = LUT.size();
		const size_t image_size = image_input.size();

		cl::Buffer device_image_input(context, CL_MEM_READ_ONLY, image_size);
		cl::Buffer histogram(context, CL_MEM_READ_WRITE, H_size);
		cl::Buffer cumHistogram(context, CL_MEM_READ_WRITE, CH_size);
		cl::Buffer LUTkern(context, CL_MEM_READ_WRITE, LUT_size);
		cl::Buffer device_image_output(context, CL_MEM_READ_WRITE, image_size);//

		queue.enqueueWriteBuffer(device_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);
		queue.enqueueWriteBuffer(histogram, CL_TRUE, 0, H_size, &H.data()[0]);
		queue.enqueueWriteBuffer(cumHistogram, CL_TRUE, 0, CH_size, &CH.data()[0]);
		queue.enqueueWriteBuffer(LUTkern, CL_TRUE, 0, LUT_elem, &LUT.data()[0]);
		queue.enqueueWriteBuffer(device_image_output, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);//

		cl::Kernel kernel_hist = cl::Kernel(program, "hist");
		kernel_hist.setArg(0, device_image_input);
		kernel_hist.setArg(1, histogram);
		cl::Kernel kernel_cumHist = cl::Kernel(program, "cumHist");
		kernel_cumHist.setArg(0, histogram);
		kernel_cumHist.setArg(1, cumHistogram);
		cl::Kernel kernel_LUT = cl::Kernel(program, "lutScale");
		kernel_LUT.setArg(0, cumHistogram);
		kernel_LUT.setArg(1, LUTkern);

		//newkern
		cl::Kernel kernel_reproj = cl::Kernel(program, "reproj");
		kernel_reproj.setArg(0, device_image_input);
		kernel_reproj.setArg(1, LUTkern);
		kernel_reproj.setArg(2, device_image_output);
		//

		queue.enqueueNDRangeKernel(kernel_hist, cl::NullRange, cl::NDRange(image_size), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel_cumHist, cl::NullRange, cl::NDRange(H_size), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel_LUT, cl::NullRange, cl::NDRange(LUT_elem), cl::NullRange);
		queue.enqueueNDRangeKernel(kernel_reproj, cl::NullRange, cl::NDRange(image_size), cl::NullRange);//

		queue.enqueueReadBuffer(histogram, CL_TRUE, 0, H_size, &H.data()[0]);
		queue.enqueueReadBuffer(cumHistogram, CL_TRUE, 0, CH_size, &CH.data()[0]);
		queue.enqueueReadBuffer(LUTkern, CL_TRUE, 0, LUT_size, &LUT.data()[0]);
		queue.enqueueReadBuffer(device_image_output, CL_TRUE, image_output.size(), &image_output.data()[0]); //

		std::cout << "H:   " << H << std::endl << std::endl;
		std::cout << "CH:  " << CH << std::endl << std::endl;
		std::cout << "LUT: " << LUT << std::endl << std::endl;

		CImgDisplay disp_input(image_input, "input");
		CImgDisplay disp_output(image_output, "output");

		while (!disp_input.is_closed() && !disp_input.is_keyESC())
		{
			disp_input.wait(1);
		}
	}

	catch (const cl::Error& err)
	{
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}