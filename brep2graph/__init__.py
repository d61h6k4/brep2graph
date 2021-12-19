# Copyright 2021 Petrov, Danil <ddbihbka@gmail.com>. All Rights Reserved.
# Author: Petrov, Danil <ddbihbka@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The main function of the package."""

from typing import Dict, Callable

import numpy as np

from OCC.Core.TopoDS import TopoDS_Shape

from brep2graph import brepnet_features
from brep2graph import configurations


def graph_from_brep(
    body: TopoDS_Shape,
    configuration: Callable = configurations.simple_edge
) -> Dict[str, np.ndarray]:
    """Convert given `body` to the graph representation."""
    body = brepnet_features.scale_solid_to_unit_box(body)

    if not brepnet_features.check_manifold(body):
        raise RuntimeError("Non-manifold bodies are not supported.")

    if not brepnet_features.check_closed(body):
        raise RuntimeError("Bodies which are not closed are not supported")

    if not brepnet_features.check_unique_coedges(body):
        raise RuntimeError(
            "Bodies where the same coedge is uses in multiple loops are not supported"
        )

    entity_mapper = brepnet_features.EntityMapper(body)

    face_features = brepnet_features.face_features_from_body(
        body, entity_mapper)
    edge_features = brepnet_features.edge_features_from_body(
        body, entity_mapper)
    coedge_features = brepnet_features.coedge_features_from_body(
        body, entity_mapper)

    coedge_to_next, coedge_to_mate, coedge_to_face, coedge_to_edge = brepnet_features.build_incidence_arrays(
        body, entity_mapper)

    graph = configuration(face_features, edge_features, coedge_features,
                          coedge_to_next, coedge_to_mate, coedge_to_face,
                          coedge_to_edge)
    return graph
