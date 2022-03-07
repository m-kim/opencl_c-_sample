#include <iostream>
#include <sstream>
#include <CL/cl.hpp>
#include <tuple>
#include <random>

std::istringstream code {R"(__kernel void test1(__global int* a, __global int* b) {
	int i = get_global_id(0);
	b[i] = a[i] * a[i];
}
)"};

std::tuple<cl::Context, cl::CommandQueue, cl::Program> setup()
{
	cl_int err = 0;
	auto platform = cl::Platform::get(&err);
	std::cout << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	std::cout << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

	std::vector<cl::Device> devices;
	err = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	auto context = cl::Context(devices, 0, 0, 0, &err);
	auto queue = cl::CommandQueue(context, devices[0], 0, &err);
	auto program = cl::Program(context, devices, code.str(), &err);
	err = program.build();
	
	return std::make_tuple(context, queue, program);
}

int main()
{
	cl_int err = 0;
	auto [context, queue, program] = setup();
	auto kernel = cl::Kernel(program, "test1", &err);
	std::vector<int> idxs(10), ret_idxs(10);
	std::vector<float> vals(10), ret_vals(10);
	//std::random_device dev;
	std::mt19937 rng(42);
	std::uniform_real_distribution<float> distVal(0.0, 1.0);
	for (auto& val : vals) {
		val = distVal(rng);
		std::cout << val << " ";
	}
	std::cout << std::endl;

	
	auto dev_val = cl::Buffer(CL_MEM_READ_ONLY | CL_MEM_HOST_PTR, sizeof(float) * vals.size(), vals.data(), &err);
	auto dev_ret_val = cl::Buffer(CL_MEM_WRITE_ONLY | CL_MEM_HOST_PTR, sizeof(float) * vals.size(), vals.data(), &err);
	int kernelIdx = 0;
	kernel.setArg(kernelIdx++, sizeof(cl_mem), &dev_val);
	kernel.setArg(kernelIdx++, sizeof(cl_mem), &dev_ret_val);
	//kernel.setArg(kernelIdx++, sizeof(cl_mem), &dev_ret_val);
	//kernel.setArg(kernelIdx++, sizeof(cl_mem), &dev_ret_idx);
	queue.enqueueNDRangeKernel(kernel, 0, 64, 32);
	queue.finish();
	queue.enqueueReadBuffer(dev_ret_val, CL_TRUE, 0, sizeof(float) * ret_vals.size(), ret_vals.data());

	for (auto val : vals)
		std::cout << val << " ";
	std::cout << std::endl;
}
