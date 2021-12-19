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

from typing import List, Tuple

import numpy as np


def _f(
    coedge_to_face: np.ndarray,
    coedge_to_node: np.ndarray,
    face_to_node: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """F.

    Creates an edge between coedge and corresponding face.
    """
    for coedge_ix, face_ix in enumerate(coedge_to_face):
        edges.append((coedge_to_node[coedge_ix], face_to_node[face_ix]))


def _mf(
    coedge_to_mate: np.ndarray,
    coedge_to_node: np.ndarray,
    coedge_to_face: np.ndarray,
    face_to_node: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """MF.

    Creates an edge between coedge and face of the mate of the coedge.
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
        edges.append((coedge_to_node[coedge_from_ix],
                      face_to_node[coedge_to_face[coedge_to_ix]]))


def _e(
    coedge_to_edge: np.ndarray,
    coedge_to_node: np.ndarray,
    edge_to_node: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """E.

    Creates an edge between coedge and corresponding edge.
    """
    for coedge_ix, edge_ix in enumerate(coedge_to_edge):
        edges.append((coedge_to_node[coedge_ix], edge_to_node[edge_ix]))


def _ne(
    coedge_to_next: np.ndarray,
    coedge_to_node: np.ndarray,
    coedge_to_edge: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """NE.

    Creates an edge between coedge and edge of the next coedge.
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_next):
        edges.append(
            (coedge_to_node[coedge_from_ix], coedge_to_edge[coedge_to_ix]))


def _pe(
    coedge_to_prev: np.ndarray,
    coedge_to_node: np.ndarray,
    coedge_to_edge: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """PE.

    Creates an edge between coedge and previous edge.
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_prev):
        edges.append(
            (coedge_to_node[coedge_from_ix], coedge_to_edge[coedge_to_ix]))


def _mne(
    coedge_to_next: np.ndarray,
    coedge_to_node: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_edge: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """MN.

    Creates an edge between coedge and edge of the mate next coedge.
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
        edges.append((coedge_to_node[coedge_from_ix],
                      coedge_to_edge[coedge_to_next[coedge_to_ix]]))


def _mpe(
    coedge_to_prev: np.ndarray,
    coedge_to_node: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_edge: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """MP.

    Creates an edge between coedge and edge of the mate previous coedge.
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
        edges.append((coedge_to_node[coedge_from_ix],
                      coedge_to_edge[coedge_to_prev[coedge_to_ix]]))


def _nmne(
    coedge_to_node: np.ndarray,
    coedge_to_next: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_edge: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """NMNE.

    Creates an edge between coedge and edge of next mate next coedge.
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_next):
        edges.append(
            (coedge_to_node[coedge_from_ix],
             coedge_to_edge[coedge_to_next[coedge_to_mate[coedge_to_ix]]]))


def _pmpe(
    coedge_to_node: np.ndarray,
    coedge_to_prev: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_edge: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """PMPE.

    Creates an edge between coedge and edge of previous mate previous coedge.
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_prev):
        edges.append(
            (coedge_to_node[coedge_from_ix],
             coedge_to_edge[coedge_to_prev[coedge_to_mate[coedge_to_ix]]]))


def _mpmpe(
    coedge_to_node: np.ndarray,
    coedge_to_prev: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_edge: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """PMPE.

    Creates an edge between coedge and edge of previous mate previous coedge.
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
        edges.append((coedge_to_node[coedge_from_ix], coedge_to_edge[
            coedge_to_prev[coedge_to_mate[coedge_to_prev[coedge_to_ix]]]]))


def _mnmne(
    coedge_to_node: np.ndarray,
    coedge_to_next: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_edge: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """PMPE.

    Creates an edge between coedge and edge of previous mate previous coedge.
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
        edges.append((coedge_to_node[coedge_from_ix], coedge_to_edge[
            coedge_to_next[coedge_to_mate[coedge_to_next[coedge_to_ix]]]]))


def _i(
    coedges_num: int,
    coedge_to_node: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """I.

    Creates self-loop for coedge.
    """
    for coedge_ix in range(coedges_num):
        edges.append((coedge_to_node[coedge_ix], coedge_to_node[coedge_ix]))


def _m(
    coedge_to_mate: np.ndarray,
    coedge_to_node: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """M.

    Creates an edge between coedge and corresponding mate coedge.
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
        edges.append(
            (coedge_to_node[coedge_from_ix], coedge_to_node[coedge_to_ix]))


def _n(
    coedge_to_next: np.ndarray,
    coedge_to_node: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """N.

    Creates an edge between coedge and next coedge.
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_next):
        edges.append(
            (coedge_to_node[coedge_from_ix], coedge_to_node[coedge_to_ix]))


def _p(
    coedge_to_prev: np.ndarray,
    coedge_to_node: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """P.

    Creates an edge between coedge and previous coedge.
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_prev):
        edges.append(
            (coedge_to_node[coedge_from_ix], coedge_to_node[coedge_to_ix]))


def _mn(
    coedge_to_next: np.ndarray,
    coedge_to_node: np.ndarray,
    coedge_to_mate: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """MN.

    Creates an edge between coedge and coedge of the mate next coedge.
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
        edges.append((coedge_to_node[coedge_from_ix],
                      coedge_to_node[coedge_to_next[coedge_to_ix]]))


def _mp(
    coedge_to_prev: np.ndarray,
    coedge_to_node: np.ndarray,
    coedge_to_mate: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """MP.

    Creates an edge between coedge and coedge of the previous mate.
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
        edges.append((coedge_to_node[coedge_from_ix],
                      coedge_to_node[coedge_to_prev[coedge_to_ix]]))


def _nm(
    coedge_to_mate: np.ndarray,
    coedge_to_node: np.ndarray,
    coedge_to_next: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """NM.

    Creates an edge between coedge and next coedge of the mate
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
        edges.append((coedge_to_node[coedge_from_ix],
                      coedge_to_node[coedge_to_next[coedge_to_ix]]))


def _pm(
    coedge_to_mate: np.ndarray,
    coedge_to_node: np.ndarray,
    coedge_to_prev: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """PM.

    Creates an edge between coedge and previous coedge of the mate
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
        edges.append((coedge_to_node[coedge_from_ix],
                      coedge_to_node[coedge_to_prev[coedge_to_ix]]))


def _mnm(
    coedge_to_mate: np.ndarray,
    coedge_to_node: np.ndarray,
    coedge_to_next: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """MNM.

    Creates an edge between coedge and coedge of mate of next mate
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
        edges.append(
            (coedge_to_node[coedge_from_ix],
             coedge_to_node[coedge_to_mate[coedge_to_next[coedge_to_ix]]]))


def _mpm(
    coedge_to_mate: np.ndarray,
    coedge_to_node: np.ndarray,
    coedge_to_prev: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """MPM.

    Creates an edge between coedge and coedge of mate of previous mate
    """
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
        edges.append(
            (coedge_to_node[coedge_from_ix],
             coedge_to_node[coedge_to_mate[coedge_to_prev[coedge_to_ix]]]))


def _nmn(
    coedge_to_next: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_node: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """NMN."""
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_next):
        edges.append(
            (coedge_to_node[coedge_from_ix],
             coedge_to_node[coedge_to_next[coedge_to_mate[coedge_to_ix]]]))


def _pmp(
    coedge_to_prev: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_node: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """PMP."""
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_prev):
        edges.append(
            (coedge_to_node[coedge_from_ix],
             coedge_to_node[coedge_to_prev[coedge_to_mate[coedge_to_ix]]]))


def _mpmp(
    coedge_to_prev: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_node: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """MPMP."""
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
        edges.append((coedge_to_node[coedge_from_ix], coedge_to_node[
            coedge_to_prev[coedge_to_mate[coedge_to_prev[coedge_to_ix]]]]))


def _mnmn(
    coedge_to_next: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_node: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """MNMN."""
    for coedge_from_ix, coedge_to_ix in enumerate(coedge_to_mate):
        edges.append((coedge_to_node[coedge_from_ix], coedge_to_node[
            coedge_to_next[coedge_to_mate[coedge_to_next[coedge_to_ix]]]]))


def _create_graph(
    face_features: np.ndarray,
    edge_features: np.ndarray,
    coedge_features: np.ndarray,
    edges: List[Tuple[int, int]],
):
    """Create the graph."""

    faces_num = face_features.shape[0]
    edges_num = edge_features.shape[0]
    coedges_num = coedge_features.shape[0]

    n_node = faces_num + edges_num + coedges_num

    senders = []
    receivers = []
    for (f, t) in edges:
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
    _f(coedge_to_face, coedge_to_node, face_to_node, edges)
    _mf(coedge_to_mate, coedge_to_node, coedge_to_face, face_to_node, edges)

    # Edges
    _e(coedge_to_edge, coedge_to_node, edge_to_node, edges)

    # CoEdges
    _i(coedges_num, coedge_to_node, edges)
    _m(coedge_to_mate, coedge_to_node, edges)

    return _create_graph(face_features, edge_features, coedge_features, edges)


def assymetric(
    face_features: np.ndarray,
    edge_features: np.ndarray,
    coedge_features: np.ndarray,
    coedge_to_next: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_face: np.ndarray,
    coedge_to_edge: np.ndarray,
):
    """Create graph according to the `assymetric` configuration."""

    faces_num = face_features.shape[0]
    edges_num = edge_features.shape[0]
    coedges_num = coedge_features.shape[0]

    face_to_node = np.arange(faces_num)
    edge_to_node = np.arange(edges_num) + faces_num
    coedge_to_node = np.arange(coedges_num) + (faces_num + edges_num)

    edges = []
    # Faces
    _f(coedge_to_face, coedge_to_node, face_to_node, edges)
    _mf(coedge_to_mate, coedge_to_node, coedge_to_face, face_to_node, edges)

    # Edges
    _e(coedge_to_edge, coedge_to_node, edge_to_node, edges)

    # CoEdges
    _i(coedges_num, coedge_to_node, edges)
    _n(coedge_to_next, coedge_to_node, edges)

    return _create_graph(face_features, edge_features, coedge_features, edges)


def assymetric_plus(
    face_features: np.ndarray,
    edge_features: np.ndarray,
    coedge_features: np.ndarray,
    coedge_to_next: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_face: np.ndarray,
    coedge_to_edge: np.ndarray,
):
    """Create graph according to the `assymetric_plus` configuration."""

    faces_num = face_features.shape[0]
    edges_num = edge_features.shape[0]
    coedges_num = coedge_features.shape[0]

    face_to_node = np.arange(faces_num)
    edge_to_node = np.arange(edges_num) + faces_num
    coedge_to_node = np.arange(coedges_num) + (faces_num + edges_num)

    edges = []
    # Faces
    _f(coedge_to_face, coedge_to_node, face_to_node, edges)
    _mf(coedge_to_mate, coedge_to_node, coedge_to_face, face_to_node, edges)

    # Edges
    _e(coedge_to_edge, coedge_to_node, edge_to_node, edges)

    # CoEdges
    _i(coedges_num, coedge_to_node, edges)
    _m(coedge_to_mate, coedge_to_node, edges)
    _n(coedge_to_next, coedge_to_node, edges)

    return _create_graph(face_features, edge_features, coedge_features, edges)


def assymetric_plus_plus(
    face_features: np.ndarray,
    edge_features: np.ndarray,
    coedge_features: np.ndarray,
    coedge_to_next: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_face: np.ndarray,
    coedge_to_edge: np.ndarray,
):
    """Create graph according to the `assymetric++` configuration."""

    faces_num = face_features.shape[0]
    edges_num = edge_features.shape[0]
    coedges_num = coedge_features.shape[0]

    face_to_node = np.arange(faces_num)
    edge_to_node = np.arange(edges_num) + faces_num
    coedge_to_node = np.arange(coedges_num) + (faces_num + edges_num)

    edges = []
    # Faces
    _f(coedge_to_face, coedge_to_node, face_to_node, edges)
    _mf(coedge_to_mate, coedge_to_node, coedge_to_face, face_to_node, edges)

    # Edges
    _e(coedge_to_edge, coedge_to_node, edge_to_node, edges)
    _ne(coedge_to_next, coedge_to_node, coedge_to_edge, edges)

    # CoEdges
    _i(coedges_num, coedge_to_node, edges)
    _m(coedge_to_mate, coedge_to_node, edges)
    _n(coedge_to_next, coedge_to_node, edges)

    return _create_graph(face_features, edge_features, coedge_features, edges)


def winged_edge(
    face_features: np.ndarray,
    edge_features: np.ndarray,
    coedge_features: np.ndarray,
    coedge_to_next: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_face: np.ndarray,
    coedge_to_edge: np.ndarray,
):
    """Create graph according to the `winged edge` configuration."""

    coedge_to_prev = np.zeros_like(coedge_to_next)
    for (from_ix, to_ix) in enumerate(coedge_to_next):
        coedge_to_prev[to_ix] = from_ix

    faces_num = face_features.shape[0]
    edges_num = edge_features.shape[0]
    coedges_num = coedge_features.shape[0]

    face_to_node = np.arange(faces_num)
    edge_to_node = np.arange(edges_num) + faces_num
    coedge_to_node = np.arange(coedges_num) + (faces_num + edges_num)

    edges = []
    # Faces
    _f(coedge_to_face, coedge_to_node, face_to_node, edges)
    _mf(coedge_to_mate, coedge_to_node, coedge_to_face, face_to_node, edges)

    # Edges
    _e(coedge_to_edge, coedge_to_node, edge_to_node, edges)
    _ne(coedge_to_next, coedge_to_node, coedge_to_edge, edges)
    _pe(coedge_to_prev, coedge_to_node, coedge_to_edge, edges)
    _mne(coedge_to_next, coedge_to_node, coedge_to_mate, coedge_to_edge, edges)
    _mpe(coedge_to_prev, coedge_to_node, coedge_to_mate, coedge_to_edge, edges)

    # CoEdges
    _i(coedges_num, coedge_to_node, edges)
    _m(coedge_to_mate, coedge_to_node, edges)
    _n(coedge_to_next, coedge_to_node, edges)
    _p(coedge_to_prev, coedge_to_node, edges)
    _mn(coedge_to_next, coedge_to_node, coedge_to_mate, edges)
    _mp(coedge_to_prev, coedge_to_node, coedge_to_mate, edges)

    return _create_graph(face_features, edge_features, coedge_features, edges)


def winged_edge_plus(
    face_features: np.ndarray,
    edge_features: np.ndarray,
    coedge_features: np.ndarray,
    coedge_to_next: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_face: np.ndarray,
    coedge_to_edge: np.ndarray,
):
    """Create graph according to the `winged edge+` configuration."""

    coedge_to_prev = np.zeros_like(coedge_to_next)
    for (from_ix, to_ix) in enumerate(coedge_to_next):
        coedge_to_prev[to_ix] = from_ix
    faces_num = face_features.shape[0]
    edges_num = edge_features.shape[0]
    coedges_num = coedge_features.shape[0]

    face_to_node = np.arange(faces_num)
    edge_to_node = np.arange(edges_num) + faces_num
    coedge_to_node = np.arange(coedges_num) + (faces_num + edges_num)

    edges = []
    # Faces
    _f(coedge_to_face, coedge_to_node, face_to_node, edges)
    _mf(coedge_to_mate, coedge_to_node, coedge_to_face, face_to_node, edges)

    # Edges
    _e(coedge_to_edge, coedge_to_node, edge_to_node, edges)
    _ne(coedge_to_next, coedge_to_node, coedge_to_edge, edges)
    _pe(coedge_to_prev, coedge_to_node, coedge_to_edge, edges)
    _mne(coedge_to_next, coedge_to_node, coedge_to_mate, coedge_to_edge, edges)
    _mpe(coedge_to_next, coedge_to_node, coedge_to_mate, coedge_to_edge, edges)

    # CoEdges
    _i(coedges_num, coedge_to_node, edges)
    _m(coedge_to_mate, coedge_to_node, edges)
    _n(coedge_to_next, coedge_to_node, edges)
    _nm(coedge_to_mate, coedge_to_node, coedge_to_next, edges)
    _p(coedge_to_prev, coedge_to_node, edges)
    _pm(coedge_to_prev, coedge_to_node, coedge_to_next, edges)
    _mn(coedge_to_next, coedge_to_node, coedge_to_mate, edges)
    _mnm(coedge_to_mate, coedge_to_node, coedge_to_next, edges)
    _mp(coedge_to_next, coedge_to_node, coedge_to_mate, edges)
    _mpm(coedge_to_mate, coedge_to_node, coedge_to_next, edges)

    return _create_graph(face_features, edge_features, coedge_features, edges)


def winged_edge_plus_plus(
    face_features: np.ndarray,
    edge_features: np.ndarray,
    coedge_features: np.ndarray,
    coedge_to_next: np.ndarray,
    coedge_to_mate: np.ndarray,
    coedge_to_face: np.ndarray,
    coedge_to_edge: np.ndarray,
):
    """Create graph according to the `winged edge++` configuration."""

    coedge_to_prev = np.zeros_like(coedge_to_next)
    for (from_ix, to_ix) in enumerate(coedge_to_next):
        coedge_to_prev[to_ix] = from_ix
    faces_num = face_features.shape[0]
    edges_num = edge_features.shape[0]
    coedges_num = coedge_features.shape[0]

    face_to_node = np.arange(faces_num)
    edge_to_node = np.arange(edges_num) + faces_num
    coedge_to_node = np.arange(coedges_num) + (faces_num + edges_num)

    edges = []
    # Faces
    _f(coedge_to_face, coedge_to_node, face_to_node, edges)
    _mf(coedge_to_mate, coedge_to_node, coedge_to_face, face_to_node, edges)

    # Edges
    _e(coedge_to_edge, coedge_to_node, edge_to_node, edges)
    _ne(coedge_to_next, coedge_to_node, coedge_to_edge, edges)
    _pe(coedge_to_prev, coedge_to_node, coedge_to_edge, edges)
    _mne(coedge_to_next, coedge_to_node, coedge_to_mate, coedge_to_edge, edges)
    _mpe(coedge_to_prev, coedge_to_node, coedge_to_mate, coedge_to_edge, edges)
    _nmne(coedge_to_node, coedge_to_next, coedge_to_mate, coedge_to_edge,
          edges)
    _pmpe(coedge_to_node, coedge_to_prev, coedge_to_mate, coedge_to_edge,
          edges)
    _mpmpe(coedge_to_node, coedge_to_prev, coedge_to_mate, coedge_to_edge,
           edges)
    _mnmne(coedge_to_node, coedge_to_next, coedge_to_mate, coedge_to_edge,
           edges)

    # CoEdges
    _i(coedges_num, coedge_to_node, edges)
    _m(coedge_to_mate, coedge_to_node, edges)
    _n(coedge_to_next, coedge_to_node, edges)
    _nm(coedge_to_mate, coedge_to_node, coedge_to_next, edges)
    _p(coedge_to_prev, coedge_to_node, edges)
    _pm(coedge_to_mate, coedge_to_node, coedge_to_prev, edges)
    _mn(coedge_to_next, coedge_to_node, coedge_to_mate, edges)
    _mnm(coedge_to_mate, coedge_to_node, coedge_to_next, edges)
    _mp(coedge_to_next, coedge_to_node, coedge_to_mate, edges)
    _mpm(coedge_to_mate, coedge_to_node, coedge_to_next, edges)
    _nmn(coedge_to_next, coedge_to_mate, coedge_to_node, edges)
    _pmp(coedge_to_prev, coedge_to_mate, coedge_to_node, edges)
    _mpmp(coedge_to_prev, coedge_to_mate, coedge_to_node, edges)
    _mnmn(coedge_to_next, coedge_to_mate, coedge_to_node, edges)

    return _create_graph(face_features, edge_features, coedge_features, edges)
