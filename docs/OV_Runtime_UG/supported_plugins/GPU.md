# GPU Device {#openvino_docs_OV_UG_supported_plugins_GPU}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_supported_plugins_GPU_RemoteTensor_API


The GPU plugin is an OpenCL based plugin for inference of deep neural networks on Intel GPUs, both integrated and discrete ones.
For an in-depth description of the GPU plugin, see:

- `GPU plugin developers documentation <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/README.md>`__
- `OpenVINO Runtime GPU plugin source files <https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_gpu/>`__
- `Accelerate Deep Learning Inference with Intel® Processor Graphics <https://software.intel.com/en-us/articles/accelerating-deep-learning-inference-with-intel-processor-graphics>`__

The GPU plugin is a part of the Intel® Distribution of OpenVINO™ toolkit. For more information on how to configure a system to use it, see the :doc:`GPU configuration <openvino_docs_install_guides_configurations_for_intel_gpu>`.

Device Naming Convention
#######################################

* Devices are enumerated as ``GPU.X``, where ``X={0, 1, 2,...}`` (only Intel® GPU devices are considered).
* If the system has an integrated GPU, its ``id`` is always 0 (``GPU.0``).
* The order of other GPUs is not predefined and depends on the GPU driver.
* The ``GPU`` is an alias for ``GPU.0``.
* If the system does not have an integrated GPU, devices are enumerated, starting from 0.
* For GPUs with multi-tile architecture (multiple sub-devices in OpenCL terms), a specific tile may be addressed as ``GPU.X.Y``, where ``X,Y={0, 1, 2,...}``, ``X`` - id of the GPU device, ``Y`` - id of the tile within device ``X``

For demonstration purposes, see the :doc:`Hello Query Device C++ Sample <openvino_inference_engine_samples_hello_query_device_README>` that can print out the list of available devices with associated indices. Below is an example output (truncated to the device names only):

.. code-block:: sh
   
   ./hello_query_device
   Available devices:
       Device: CPU
   ...
       Device: GPU.0
   ...
       Device: GPU.1
   ...
       Device: GNA


Then, the device name can be passed to the ``ov::Core::compile_model()`` method, running on:

.. tab-set::
   
   .. tab-item:: default device

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/gpu/compile_model.cpp
               :language: cpp
               :fragment: compile_model_default_gpu

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/snippets/gpu/compile_model.py
               :language: Python
               :fragment: compile_model_default_gpu

   .. tab-item:: specific GPU

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/gpu/compile_model.cpp
               :language: cpp
               :fragment: compile_model_gpu_with_id

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/snippets/gpu/compile_model.py
               :language: Python
               :fragment: compile_model_gpu_with_id

   .. tab-item:: specific tile

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/gpu/compile_model.cpp
               :language: cpp
               :fragment: compile_model_gpu_with_id_and_tile

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/snippets/gpu/compile_model.py
               :language: Python
               :fragment: compile_model_gpu_with_id_and_tile

Supported Inference Data Types
#######################################

The GPU plugin supports the following data types as inference precision of internal primitives:

- Floating-point data types:

  - f32
  - f16

- Quantized data types:

  - u8
  - i8
  - u1

Selected precision of each primitive depends on the operation precision in IR, quantization primitives, and available hardware capabilities.
The ``u1``/``u8``/``i8`` data types are used for quantized operations only, which means that they are not selected automatically for non-quantized operations.
For more details on how to get a quantized model, refer to the :doc:`Model Optimization guide <openvino_docs_model_optimization_guide>`.

Floating-point precision of a GPU primitive is selected based on operation precision in the OpenVINO IR, except for the :doc:`<compressed f16 OpenVINO IR form <openvino_docs_MO_DG_FP16_Compression>`, which is executed in the ``f16`` precision.

.. note::

   Hardware acceleration for ``i8``/``u8`` precision may be unavailable on some platforms. In such cases, a model is executed in the floating-point precision taken from IR. 
   Hardware support of ``u8``/``i8`` acceleration can be queried via the `ov::device::capabilities` property.

:doc:`Hello Query Device C++ Sample<openvino_inference_engine_samples_hello_query_device_README>` can be used to print out the supported data types for all detected devices.


Supported Features
#######################################

The GPU plugin supports the following features:

Multi-device Execution
+++++++++++++++++++++++++++++++++++++++

If a system has multiple GPUs (for example, an integrated and a discrete Intel GPU), then any supported model can be executed on all GPUs simultaneously.
It is done by specifying ``MULTI:GPU.1,GPU.0`` as a target device.

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/gpu/compile_model.cpp
         :language: cpp
         :fragment: compile_model_multi

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/gpu/compile_model.py
         :language: Python
         :fragment: compile_model_multi


For more details, see the :doc:`Multi-device execution <openvino_docs_OV_UG_Running_on_multiple_devices>`.

Automatic Batching
+++++++++++++++++++++++++++++++++++++++

The GPU plugin is capable of reporting ``ov::max_batch_size`` and ``ov::optimal_batch_size`` metrics with respect to the current hardware
platform and model. Therefore, automatic batching is enabled by default when ``ov::optimal_batch_size`` is ``> 1`` and ``ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)`` is set.
Alternatively, it can be enabled explicitly via the device notion, for example ``BATCH:GPU``.


.. tab-set::
   
   .. tab-item:: Batching via BATCH plugin

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/gpu/compile_model.cpp
               :language: cpp
               :fragment: compile_model_batch_plugin

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/snippets/gpu/compile_model.py
               :language: Python
               :fragment: compile_model_batch_plugin

   .. tab-item:: Batching via throughput hint

      .. tab-set::

         .. tab-item:: C++
            :sync: cpp

            .. doxygensnippet:: docs/snippets/gpu/compile_model.cpp
               :language: cpp
               :fragment: compile_model_auto_batch

         .. tab-item:: Python
            :sync: py

            .. doxygensnippet:: docs/snippets/gpu/compile_model.py
               :language: Python
               :fragment: compile_model_auto_batch


For more details, see the :doc:`Automatic batching<openvino_docs_OV_UG_Automatic_Batching>`.

Multi-stream Execution
+++++++++++++++++++++++++++++++++++++++

If either the ``ov::num_streams(n_streams)`` with ``n_streams > 1`` or the ``ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)`` property is set for the GPU plugin,
multiple streams are created for the model. In the case of GPU plugin each stream has its own host thread and an associated OpenCL queue
which means that the incoming infer requests can be processed simultaneously.

.. note:: 

   Simultaneous scheduling of kernels to different queues does not mean that the kernels are actually executed in parallel on the GPU device. 
   The actual behavior depends on the hardware architecture and in some cases the execution may be serialized inside the GPU driver.

When multiple inferences of the same model need to be executed in parallel, the multi-stream feature is preferred to multiple instances of the model or application.
The reason for this is that the implementation of streams in the GPU plugin supports weight memory sharing across streams, thus, memory consumption may be lower, compared to the other approaches.

For more details, see the :doc:`optimization guide<openvino_docs_deployment_optimization_guide_dldt_optimization_guide>`.

Dynamic Shapes
+++++++++++++++++++++++++++++++++++++++

The support for dynamic shapes by GPU plugin is different dipending on the OpenVINO™ API version : pre or post API 2.0. 
Prior to OpenVINO ™ API 2.0, only dynamic batch was available, where dynamism on the batch dimension was supported only.
Also there was a limitation that an upper bound of the batch dimension should be provided. 
This method will be referred as 'legacy dynamic batch" in the following statemetns.
From the API 2.0, general dynamic shape introduced in :doc:`dynamic shapes guide<openvino_docs_OV_UG_DynamicShapes>` is available. 

#### Legacy dynamic batch (< API 2.0)

In this mode, GPU plugin supports dynamic shapes for batch dimension only (specified as ``N`` in the :doc:`layouts terms<openvino_docs_OV_UG_Layout_Overview>`) with a fixed upper bound. 
Any other dynamic dimensions are unsupported in this version. Internally, GPU plugin creates ``log2(N)`` (``N`` - is an upper bound for batch dimension here) 
low-level execution graphs for batch sizes equal to powers of 2 to emulate dynamic behavior, so that incoming infer request 
with a specific batch size is executed via a minimal combination of internal networks. For example, batch size 33 may be executed via 2 internal networks with batch size 32 and 1.

.. note:: 

   Such approach requires much more memory and the overall model compilation time is significantly longer, compared to the static batch scenario or general dynamic shape supported from API 2.0.
 
The code snippet below demonstrates how to use dynamic batching in simple scenarios:

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/gpu/dynamic_batch_legacy.cpp # TBD
         :language: cpp
         :fragment: dynamic_batch_legacy

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/gpu/dynamic_batch_legacy.py
         :language: Python
         :fragment: dynamic_batch_legacy

#### General dynamic shape (>= API 2.0)

From 2023.1 release, GPU plugin supports dynamic shapes for all dimensions not only the batch dimension.
This mode is different from the legacy dynamic batch creating multiple copies of the model with sub-batches, 
Note that the dynamic rank is not supported.
Unlike the legacy dynamic batch which created multiple copies of the model with sub-batch sizes and assembled them, this mode
allowes the plugin infra to change the shapes of each layer at runtime.

The code snippet below demonstrates how to use dynamic shape in GPU plugin:

... note::
    This feature is newly supported from 2023.1 release and is mainly targeting NLP modes.
    Not all the operations are supported for dynamic shape and models with such unsupported operations may crashes.
    Also, the legaciy dynamic batch mode is no longer supported in API 2.0 from 2023.1 release.
    Since not all models are supported for dynamic shape yet, some models where only the batch dimension is dynamic may not work with API 2.0 from 2023.1 release, though they could work in the 2022.4 release.
    In such cases, please try with legacy API described above.

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/gpu/dynamic_batch_api_2.0.cpp # TBD
         :language: cpp
         :fragment: dynamic_shape

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/gpu/dynamic_shape_api_2.0.py
         :language: Python
         :fragment: dynamic_shape


.. note::
   For better performance, setting upper bound of the tensor dimensions is helpful for many cases, since it can reduce the number of runtime memory reallocation for new larger shapes. 

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/gpu/dynamic_batch_api_2.0_bounded.cpp # TBD
         :language: cpp
         :fragment: dynamic_shape_bounded

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/gpu/dynamic_shape_api_2.0_bounded.py
         :language: Python
         :fragment: dynamic_shape_bounded

For more details, see the :doc:`dynamic shapes guide<openvino_docs_OV_UG_DynamicShapes>`.

Preprocessing Acceleration
+++++++++++++++++++++++++++++++++++++++

The GPU plugin has the following additional preprocessing options:
- The ``ov::intel_gpu::memory_type::surface`` and ``ov::intel_gpu::memory_type::buffer`` values for the ``ov::preprocess::InputTensorInfo::set_memory_type()`` preprocessing method. These values are intended to be used to provide a hint for the plugin on the type of input Tensors that will be set in runtime to generate proper kernels.

.. tab-set::

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/gpu/preprocessing_nv12_two_planes.cpp
         :language: cpp
         :fragment: init_preproc

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/gpu/preprocessing_nv12_two_planes.py
         :language: Python
         :fragment: init_preproc


With such preprocessing, GPU plugin will expect ``ov::intel_gpu::ocl::ClImage2DTensor`` (or derived) to be passed for each NV12 plane via ``ov::InferRequest::set_tensor()`` or ``ov::InferRequest::set_tensors()`` methods.

For usage examples, refer to the :doc:`RemoteTensor API<openvino_docs_OV_UG_supported_plugins_GPU_RemoteTensor_API>`.

For more details, see the :doc:`preprocessing API<openvino_docs_OV_UG_Preprocessing_Overview>`.

Model Caching
+++++++++++++++++++++++++++++++++++++++

Model Caching helps reduce application startup delays by exporting and reusing 
the compiled model automatically. The cache for the GPU plugin may be enabled 
via the common OpenVINO ``ov::cache_dir`` property. 

This means that all plugin-specific model transformations are executed on each ``ov::Core::compile_model()`` 
call, regardless of the ``ov::cache_dir`` option. Still, since kernel compilation is a bottleneck in the model 
loading process, a significant load time reduction can be achieved.
Currently, GPU plugin implementation fully supports static models only. For dynamic models,
kernel caching is used instead and multiple ‘.cl_cache’ files are generated along with the ‘.blob’ file. 

For more details, see the :doc:`Model caching overview <openvino_docs_OV_UG_Model_caching_overview>`.

Extensibility
+++++++++++++++++++++++++++++++++++++++

For information on this subject, see the :doc:`GPU Extensibility <openvino_docs_Extensibility_UG_GPU>`.

GPU Context and Memory Sharing via RemoteTensor API
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

For information on this subject, see the :doc:`RemoteTensor API of GPU Plugin <openvino_docs_OV_UG_supported_plugins_GPU_RemoteTensor_API>`.

Supported Properties
#######################################

The plugin supports the properties listed below.

Read-write properties
+++++++++++++++++++++++++++++++++++++++

All parameters must be set before calling ``ov::Core::compile_model()`` in order to take effect or passed as additional argument to ``ov::Core::compile_model()``.

- ov::cache_dir
- ov::enable_profiling
- ov::hint::model_priority
- ov::hint::performance_mode
- ov::hint::execution_mode
- ov::hint::num_requests
- ov::hint::inference_precision
- ov::num_streams
- ov::compilation_num_threads
- ov::device::id
- ov::intel_gpu::hint::host_task_priority
- ov::intel_gpu::hint::queue_priority
- ov::intel_gpu::hint::queue_throttle
- ov::intel_gpu::enable_loop_unrolling

Read-only Properties
+++++++++++++++++++++++++++++++++++++++

- ov::supported_properties
- ov::available_devices
- ov::range_for_async_infer_requests
- ov::range_for_streams
- ov::optimal_batch_size
- ov::max_batch_size
- ov::device::full_name
- ov::device::type
- ov::device::gops
- ov::device::capabilities
- ov::intel_gpu::device_total_mem_size
- ov::intel_gpu::uarch_version
- ov::intel_gpu::execution_units_count
- ov::intel_gpu::memory_statistics

Limitations
#######################################

In some cases, the GPU plugin may implicitly execute several primitives on CPU using internal implementations, which may lead to an increase in CPU utilization.
Below is a list of such operations:

- Proposal
- NonMaxSuppression
- DetectionOutput

The behavior depends on specific parameters of the operations and hardware configuration.


GPU Performance Checklist: Summary
#######################################

Since OpenVINO relies on the OpenCL kernels for the GPU implementation, many general OpenCL tips apply:

-	Prefer ``FP16`` inference precision over ``FP32``, as Model Optimizer can generate both variants, and the ``FP32`` is the default. To learn about optimization options, see :doc:`Optimization Guide<openvino_docs_model_optimization_guide>`.
- Try to group individual infer jobs by using :doc:`automatic batching <openvino_docs_OV_UG_Automatic_Batching>`.
-	Consider :doc:`caching <openvino_docs_OV_UG_Model_caching_overview>` to minimize model load time.
-	If your application performs inference on the CPU alongside the GPU, or otherwise loads the host heavily, make sure that the OpenCL driver threads do not starve. :doc:`CPU configuration options <openvino_docs_OV_UG_supported_plugins_CPU>` can be used to limit the number of inference threads for the CPU plugin.
-	Even in the GPU-only scenario, a GPU driver might occupy a CPU core with spin-loop polling for completion. If CPU load is a concern, consider the dedicated ``queue_throttle`` property mentioned previously. Note that this option may increase inference latency, so consider combining it with multiple GPU streams or :doc:`throughput performance hints <openvino_docs_OV_UG_Performance_Hints>`.
- When operating media inputs, consider :doc:`remote tensors API of the GPU Plugin <openvino_docs_OV_UG_supported_plugins_GPU_RemoteTensor_API>`.


Additional Resources
#######################################

* :doc:`Supported Devices <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`.
* :doc:`Optimization guide <openvino_docs_deployment_optimization_guide_dldt_optimization_guide>`.
* `GPU plugin developers documentation <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/README.md>`__


@endsphinxdirective
