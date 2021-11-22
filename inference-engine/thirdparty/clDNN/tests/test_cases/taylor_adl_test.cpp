#include <gtest/gtest.h>
#include <cl2_wrapper.h>
#include "../../src/gpu/kernels_cache.h"
#include <api/topology.hpp>
#include <api/program.hpp>
#include "program_impl.h"
#include "api/engine.hpp"
#include "gpu/ocl_toolkit.h"
#include "../../src/include/kernel_selector_helper.h"
#include "foo.cl"
#include <fstream>
#include <sstream>
static void checkStatus(int status, const char *message)
{
    if (status != 0)
    {
        std::string str_message(message + std::string(": "));
        std::string str_number(std::to_string(status));
        throw std::runtime_error(str_message + str_number);
    }
}
struct OpenCL
{
    cl::Context _context;
    cl::Device _device;
    cl::CommandQueue _queue;
    OpenCL()
    {
        // get Intel iGPU OCL device, create context and queue
        {
            static constexpr auto INTEL_PLATFORM_VENDOR = "Intel(R) Corporation";
            const uint32_t device_type = CL_DEVICE_TYPE_GPU;  // only gpu devices
            const uint32_t device_vendor = 0x8086;  // Intel vendor
            cl_uint n = 0;
            cl_int err = clGetPlatformIDs(0, NULL, &n);
            checkStatus(err, "clGetPlatformIDs");
            // Get platform list
            std::vector<cl_platform_id> platform_ids(n);
            err = clGetPlatformIDs(n, platform_ids.data(), NULL);
            checkStatus(err, "clGetPlatformIDs");
            for (auto& id : platform_ids) {
                cl::Platform platform = cl::Platform(id);

                auto platform_name = platform.getInfo<CL_PLATFORM_NAME>();
                std::cout << "pltform" << platform_name << std::endl;
                auto vendor_id = platform.getInfo<CL_PLATFORM_VENDOR>();
                if (vendor_id != INTEL_PLATFORM_VENDOR)
                    continue;

                std::vector<cl::Device> devices;
                platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
                for (auto& d : devices) {
                    std::cout << "device " << d.getInfo<CL_DEVICE_NAME>() << std::endl;
                }
                for (auto& d : devices) {
                    if (d.getInfo<CL_DEVICE_TYPE>() == device_type &&
                        d.getInfo<CL_DEVICE_VENDOR_ID>() == device_vendor) {
                        _device = d;
                        _context = cl::Context(_device);
                        return;
                    }
                }
            }
        }
    }
    void releaseOclImage(std::shared_ptr<cl_mem> image)
    {
        checkStatus(clReleaseMemObject(*image), "clReleaseMemObject");
    }
};
#if 0
TEST(kernel_cache_test, ssd_resnet34) {
    auto ocl_instance = std::make_shared<OpenCL>();
#if 0
    auto& ctx = ocl_instance->_context;
    cldnn::gpu::configuration context_config;
    context_config.compiler_options = std::string();
    context_config.enable_profiling = false;
    context_config.meaningful_kernels_names = false;
    context_config.dump_custom_program = false;
    context_config.single_kernel_name = std::string();
    context_config.host_out_of_order = true;
    context_config.use_unifed_shared_memory = true;  // Switch on/off USM.
    context_config.log = std::string();
    context_config.ocl_sources_dumps_dir = "./cl_dumps";
    context_config.priority_mode = cldnn::priority_mode_types::disabled;
    context_config.throttle_mode = cldnn::throttle_mode_types::disabled;
    context_config.queues_num = 2;
    context_config.kernels_cache_path = "";
    context_config.tuning_cache_path = "";
    context_config.n_threads = 8;
    auto context = cldnn::gpu::gpu_toolkit::create(dev, context_config);
    auto cache = cldnn::gpu::kernels_cache(ctx, 0);
#endif
    device_query query(static_cast<void*>(ocl_instance->_context.get()));
    auto devices = query.get_available_devices();
    auto engine_config = cldnn::engine_configuration();
    engine_config.n_threads = 8;
    engine_config.n_streams = 2;
    engine engine(devices.begin()->second, engine_config);
//    engine.get_context()->add_program(0);

    topology topo;
    program prog(engine, topo, build_options());

    auto cache = cldnn::gpu::kernels_cache(*prog.get()->get_engine().get_context(), 0);
}
#endif
TEST(taylor_test, test) {
    auto ocl_instance = std::make_shared<OpenCL>();

    const int n_threads = 8;
    const int n_kernels = 80;
    cldnn::custom::task_arena arena{n_threads};
    std::vector<std::string> kernel_code_copied;
    for (auto i = 0; i < n_kernels; ++i) {
        std::string kernel_code = "";
        for (auto ss : kernel_codes) {
            kernel_code += ss;
        }
        kernel_code_copied.push_back(kernel_code);
    }

    arena.execute([&] {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n_kernels), [&](const tbb::blocked_range<size_t>& r) {
            for (auto i = r.begin(); i != r.end(); ++i) {
                cl::vector<cl::Kernel> kernels;
                auto& ctx = ocl_instance->_context;
                auto& device = ocl_instance->_device;
                std::ofstream dump_file;
                dump_file.open("dump_code_" + std::to_string(i) + ".cl");
                if (dump_file.good()) {
                    dump_file << kernel_code_copied[i];
                }
                try {
                    cl::Program program(ctx, kernel_code_copied[i]);
                    checkStatus(program.build({device}, ""), "build");
                    program.createKernels(&kernels);
                } catch (const cl::BuildError& err) {
                    std::string err_log;
                    for (auto& p : err.getBuildLog()) {
                        err_log += p.second + '\n';
                    }
                    std::cout << err_log << std::endl;
                    assert(false);
                }
            }
        });
    });
}

