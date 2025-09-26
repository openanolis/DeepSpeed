# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Tuple, List

import torch
from torch.fx import Graph, Node
from torch.utils.checkpoint import CheckpointPolicy
from torch._functorch.partitioners import _is_primal

from .util import get_no_copy_ops, is_cast_op


def recompute_params(joint_graph: Graph, param_indices: List[Tuple[int, int, torch.Size]]):
    no_copy_ops = get_no_copy_ops()
    primal_inputs = list(filter(_is_primal, joint_graph.nodes))
    ds_param_inputs = set([primal_inputs[arg_idx] for arg_idx, _, _ in param_indices])
    recomputed_nodes = set()

    need_recompute = lambda n: n.target in no_copy_ops or is_cast_op(n)
    for node in joint_graph.nodes:
        # Arguments can be non-tensor types some of which are not hashable. So
        # we must inspect the type of an argument before checking if it is in
        # any set.
        if need_recompute(node) and \
            any([(isinstance(a, Node) and a in ds_param_inputs or a in recomputed_nodes) for a in node.args]):
            node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
            recomputed_nodes.add(node)
