//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/op/prelu.hpp"
#include <ngraph/runtime/reference/prelu.hpp>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::PRelu, "PRelu", 0);

op::PRelu::PRelu()
    : Op()
{
}

op::PRelu::PRelu(const Output<Node>& data, const Output<Node>& slope)
    : Op({data, slope})
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::PRelu::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_PRelu_visit_attributes);
    return true;
}

void ngraph::op::v0::PRelu::validate_and_infer_types()
{
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::PRelu::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_PRelu_clone_with_new_inputs);
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<PRelu>(new_args.at(0), new_args.at(1));
}

namespace prelu
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& slope, const HostTensorPtr& out)
    {
        runtime::reference::prelu(arg->get_data_ptr<ET>(),
                                  slope->get_data_ptr<ET>(),
                                  out->get_data_ptr<ET>(),
                                  arg->get_shape(),
                                  slope->get_shape());
        return true;
    }

    bool evaluate_prelu(const HostTensorPtr& arg,
                        const HostTensorPtr& slope,
                        const HostTensorPtr& out)
    {
        bool rc = true;
        switch (arg->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_prelu, i8, arg, slope, out);
            NGRAPH_TYPE_CASE(evaluate_prelu, bf16, arg, slope, out);
            NGRAPH_TYPE_CASE(evaluate_prelu, f16, arg, slope, out);
            NGRAPH_TYPE_CASE(evaluate_prelu, f32, arg, slope, out);
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::PRelu::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_PRelu_evaluate);
    NGRAPH_CHECK(this,
                 validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 2));
    return prelu::evaluate_prelu(inputs[0], inputs[1], outputs[0]);
}
