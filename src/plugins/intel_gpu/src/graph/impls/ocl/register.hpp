// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/arg_max_min.hpp"
#include "intel_gpu/primitives/average_unpooling.hpp"
#include "intel_gpu/primitives/batch_to_space.hpp"
#include "intel_gpu/primitives/binary_convolution.hpp"
#include "intel_gpu/primitives/border.hpp"
#include "intel_gpu/primitives/broadcast.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/custom_gpu_primitive.hpp"
#include "intel_gpu/primitives/deconvolution.hpp"
#include "intel_gpu/primitives/depth_to_space.hpp"
#include "intel_gpu/primitives/detection_output.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/experimental_detectron_roi_feature_extractor.hpp"
#include "intel_gpu/primitives/experimental_detectron_topk_rois.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/gather.hpp"
#include "intel_gpu/primitives/gather_nd.hpp"
#include "intel_gpu/primitives/gather_elements.hpp"
#include "intel_gpu/primitives/gemm.hpp"
#include "intel_gpu/primitives/lrn.hpp"
#include "intel_gpu/primitives/lstm.hpp"
#include "intel_gpu/primitives/lstm_dynamic.hpp"
#include "intel_gpu/primitives/max_unpooling.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/mvn.hpp"
#include "intel_gpu/primitives/non_max_suppression.hpp"
#include "intel_gpu/primitives/normalize.hpp"
#include "intel_gpu/primitives/one_hot.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/pooling.hpp"
#include "intel_gpu/primitives/pyramid_roi_align.hpp"
#include "intel_gpu/primitives/quantize.hpp"
#include "intel_gpu/primitives/random_uniform.hpp"
#include "intel_gpu/primitives/range.hpp"
#include "intel_gpu/primitives/reduce.hpp"
#include "intel_gpu/primitives/region_yolo.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reorg_yolo.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/reverse_sequence.hpp"
#include "intel_gpu/primitives/roi_align.hpp"
#include "intel_gpu/primitives/roi_pooling.hpp"
#include "intel_gpu/primitives/scale.hpp"
#include "intel_gpu/primitives/scatter_update.hpp"
#include "intel_gpu/primitives/scatter_elements_update.hpp"
#include "intel_gpu/primitives/scatter_nd_update.hpp"
#include "intel_gpu/primitives/select.hpp"
#include "intel_gpu/primitives/shuffle_channels.hpp"
#include "intel_gpu/primitives/slice.hpp"
#include "intel_gpu/primitives/softmax.hpp"
#include "intel_gpu/primitives/space_to_batch.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"
#include "intel_gpu/primitives/tile.hpp"
#include "intel_gpu/primitives/resample.hpp"
#include "intel_gpu/primitives/gather_tree.hpp"
#include "intel_gpu/primitives/lstm_dynamic_input.hpp"
#include "intel_gpu/primitives/lstm_dynamic_timeloop.hpp"
#include "intel_gpu/primitives/grn.hpp"
#include "intel_gpu/primitives/ctc_greedy_decoder.hpp"
#include "intel_gpu/primitives/convert_color.hpp"
#include "generic_layer.hpp"


namespace cldnn {
namespace ocl {
void register_implementations();

namespace detail {

#define REGISTER_OCL(prim)               \
    struct attach_##prim##_impl {        \
        attach_##prim##_impl();          \
    }

REGISTER_OCL(activation);
REGISTER_OCL(arg_max_min);
#if 0 // TODO(taylor)
REGISTER_OCL(average_unpooling);
REGISTER_OCL(batch_to_space);
REGISTER_OCL(binary_convolution);
REGISTER_OCL(border);
REGISTER_OCL(broadcast);
#endif
REGISTER_OCL(concatenation);
REGISTER_OCL(convolution);
REGISTER_OCL(crop);
#if 0 // TODO(taylor)
REGISTER_OCL(custom_gpu_primitive);
#endif
REGISTER_OCL(data);
#if 0 // TODO(taylor)
REGISTER_OCL(deconvolution);
REGISTER_OCL(deformable_conv);
REGISTER_OCL(deformable_interp);
REGISTER_OCL(depth_to_space);
REGISTER_OCL(detection_output);
REGISTER_OCL(experimental_detectron_roi_feature_extractor);
REGISTER_OCL(experimental_detectron_topk_rois);
#endif
REGISTER_OCL(eltwise);
#if 0 // TODO(andrew)
REGISTER_OCL(embed);
#endif
REGISTER_OCL(fully_connected);
#if 0 // TODO(andrew)
REGISTER_OCL(gather);
REGISTER_OCL(gather_nd);
REGISTER_OCL(gather_elements);
REGISTER_OCL(gemm);
REGISTER_OCL(lrn);
REGISTER_OCL(lstm_gemm);
REGISTER_OCL(lstm_elt);
REGISTER_OCL(max_unpooling);
REGISTER_OCL(mutable_data);
REGISTER_OCL(mvn);
#endif
REGISTER_OCL(non_max_suppression);
#if 0 // TODO(taylor)
REGISTER_OCL(normalize);
REGISTER_OCL(one_hot);
#endif
REGISTER_OCL(permute);
#if 0 // TODO(taylor)
REGISTER_OCL(pooling);
REGISTER_OCL(pyramid_roi_align);
#endif
REGISTER_OCL(quantize);
#if 0 // TODO(andrew)
REGISTER_OCL(random_uniform);
REGISTER_OCL(range);
REGISTER_OCL(reduce);
REGISTER_OCL(region_yolo);
#endif
REGISTER_OCL(reorder);
#if 0 // TODO(andrew)
REGISTER_OCL(reorg_yolo);
#endif
REGISTER_OCL(reshape);
#if 0 // TODO(andrew)
REGISTER_OCL(reverse_sequence);
REGISTER_OCL(roi_align);
REGISTER_OCL(roi_pooling);
#endif
REGISTER_OCL(scale);
#if 0 // TODO(andrew)
REGISTER_OCL(scatter_update);
REGISTER_OCL(scatter_elements_update);
REGISTER_OCL(scatter_nd_update);
REGISTER_OCL(select);
REGISTER_OCL(shuffle_channels);
REGISTER_OCL(slice);
REGISTER_OCL(softmax);
REGISTER_OCL(space_to_batch);
REGISTER_OCL(space_to_depth);
REGISTER_OCL(strided_slice);
REGISTER_OCL(tile);
REGISTER_OCL(lstm_dynamic_input);
REGISTER_OCL(lstm_dynamic_timeloop);
#endif
REGISTER_OCL(generic_layer);
#if 0 // TODO(andrew)
REGISTER_OCL(gather_tree);
REGISTER_OCL(resample);
REGISTER_OCL(grn);
REGISTER_OCL(ctc_greedy_decoder);
REGISTER_OCL(cum_sum);
REGISTER_OCL(embedding_bag);
REGISTER_OCL(extract_image_patches);
REGISTER_OCL(convert_color);
#endif
#undef REGISTER_OCL

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
