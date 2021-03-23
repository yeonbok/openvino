"""
 Copyright (C) 2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import unittest

import numpy as np
from generator import generator, generate

from extensions.ops.scatter import ScatterElementsUpdate
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph, regular_op_with_empty_data, result, connect, \
    valued_const_with_data


@generator
class ScatterElementsInferTest(unittest.TestCase):
    @generate(*[
        ([[0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]],
         [[1, 0, 2],
          [0, 2, 1]],
         [[1.0, 1.1, 1.2],
          [2.0, 2.1, 2.2]],
         0,
         [[2.0, 1.1, 0.0],
          [1.0, 0.0, 2.2],
          [0.0, 2.1, 1.2]]),

        ([[1.0, 2.0, 3.0, 4.0, 5.0]],
         [[1, 3]],
         [[1.1, 2.1]],
         1,
         [[1.0, 1.1, 3.0, 2.1, 5.0]]),

        ([[1.0, 2.0, 3.0, 4.0, 5.0]],
         [[1, 3]],
         [[1.1, 2.1]],
         [1],
         [[1.0, 1.1, 3.0, 2.1, 5.0]]),

        ([  # 3D case
          [[1, 2],
           [3, 4]],
          [[5, 6],
           [7, 8]],
          [[9, 10],
           [11, 12]]
        ],
         [
          [[1, 0],
           [0, 1]],
          [[1, 0],
           [1, 0]],
          [[0, 1],
           [1, 0]]
         ],
         [
             [[21, 22],
              [23, 24]],
             [[25, 26],
              [27, 28]],
             [[29, 30],
              [31, 32]]
         ],
         -1,  # axis
         [
             [[22, 21],
              [23, 24]],
             [[26, 25],
              [28, 27]],
             [[29, 30],
              [32, 31]]
         ]),
    ])

    def test_scatterelements_value_infer(self, data, indices, updates, axis, ref_res):
        nodes = {
            **valued_const_with_data('data', np.array(data)),
            **valued_const_with_data('indices', int64_array(indices)),
            **valued_const_with_data('updates', np.array(updates)),
            **valued_const_with_data('axis', int64_array(axis)),
            **regular_op_with_empty_data('scatter_elements', {'op': 'ScatterElementsUpdate', 'axis': axis}),
            **result()
        }

        graph = build_graph(nodes_attrs=nodes, edges=[
            *connect('data', '0:scatter_elements'),
            *connect('indices', '1:scatter_elements'),
            *connect('updates', '2:scatter_elements'),
            *connect('axis', '3:scatter_elements'),
            *connect('scatter_elements', 'output')
        ], nodes_with_edges_only=True)
        graph.stage = 'middle'

        scatter_el_node = Node(graph, 'scatter_elements')
        ScatterElementsUpdate.infer(scatter_el_node)

        res_output_shape = scatter_el_node.out_node().shape
        self.assertTrue(np.array_equal(int64_array(ref_res).shape, res_output_shape))

        res_output_value = scatter_el_node.out_node().value
        self.assertTrue(np.array_equal(ref_res, res_output_value))
