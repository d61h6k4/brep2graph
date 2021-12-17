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
"""Representation of the CAD model in Simple Edge configuration.
BRepNet: A topological message passing system for solid models.
https://arxiv.org/pdf/2104.00706.pdf
"""

import numpy as np


def simple_edge(
    face_features: np.ndarray,
    edge_features: np.ndarray,
    coedge_features: np.ndarray,
    coedge_to_next: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_face: np.ndarray,
    coedge_to_edge: np.ndarray,
):
    """Create graph according to the `simple edge` configuration."""
    del coedge_to_next

    faces_num = face_features.shape[0]
    edges_num = edge_features.shape[0]
    coedges_num = coedge_features.shape[0]

    face_to_node = np.arange(faces_num)
    edge_to_node = np.arange(edges_num) + faces_num
    coedge_to_node = np.arange(coedges_num) + (faces_num + edges_num)

    edges = []
    # Faces
    # F
    for coedge_ix, face_ix in enumerate(coedge_to_face):
        edges.append((coedge_to_node[coedge_ix], face_to_node[face_ix]))
    # MF
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
        edges.append((coedge_to_node[coedge_from_ix],
                      face_to_node[coedge_to_face[coedge_to_ix]]))
    # Edges
    # E
    for coedge_ix, edge_ix in enumerate(coedge_to_edge):
        edges.append((coedge_to_node[coedge_ix], edge_to_node[edge_ix]))
    # CoEdges
    # I
    for coedge_ix in range(coedges_num):
        edges.append((coedge_to_node[coedge_ix], coedge_to_node[coedge_ix]))
    # M
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
        edges.append(
            (coedge_to_node[coedge_from_ix], coedge_to_node[coedge_to_ix]))

    n_node = faces_num + edges_num + coedges_num

    senders = []
    receivers = []
    for edge_ix, (f, t) in enumerate(edges):
        senders.append(f)
        receivers.append(t)
        # don't add self-loops more than once
        if f != t:
            senders.append(t)
            receivers.append(f)

    assert len(senders) == len(receivers)
    n_edge = len(senders)

    nodes = np.concatenate(
        (np.pad(face_features,
                ((0, 0),
                 (0, edge_features.shape[1] + coedge_features.shape[1]))),
         np.pad(edge_features,
                ((0, 0), (face_features.shape[1], coedge_features.shape[1]))),
         np.pad(coedge_features,
                ((0, 0),
                 (face_features.shape[1] + edge_features.shape[1], 0)))))

    return {
        "n_node": np.array([n_node], dtype=np.int32),
        "n_edge": np.array([n_edge], dtype=np.int32),
        "nodes": nodes,
        "senders": np.array(senders, dtype=np.int32),
        "receivers": np.array(receivers, dtype=np.int32),
    }
