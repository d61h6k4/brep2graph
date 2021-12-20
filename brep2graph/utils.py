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
"""Collection of helper functions."""

import pathlib

from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Shape
from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCC.Extend.DataExchange import read_step_file
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer


def load_body(step_filepath: pathlib.Path) -> TopoDS_Shape:
    """Load the body from the given STEP file.
    """
    return read_step_file(str(step_filepath.absolute()), as_compound=False)


def find_box(body: TopoDS_Shape) -> Bnd_Box:
    """Find the bounding box that contains the given body."""
    box = Bnd_Box()
    use_triangulation = True
    use_shapetolerance = False
    brepbndlib.AddOptimal(body, box, use_triangulation, use_shapetolerance)
    return box


def scale_solid_to_unit_box(body: TopoDS_Shape) -> TopoDS_Shape:
    """ Centered the body on the origin and scaled so it just fits
    into a box [-1, 1]^3
    """
    bbox = find_box(body)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    longest_length = max(xmax - xmin, ymax - ymin, zmax - zmin)

    orig = gp_Pnt(0.0, 0.0, 0.0)
    center = gp_Pnt(
        (xmin + xmax) / 2.0,
        (ymin + ymax) / 2.0,
        (zmin + zmax) / 2.0,
    )
    vec_center_to_orig = gp_Vec(center, orig)
    move_to_center = gp_Trsf()
    move_to_center.SetTranslation(vec_center_to_orig)

    scale_trsf = gp_Trsf()
    scale_trsf.SetScale(orig, 2.0 / longest_length)
    trsf_to_apply = scale_trsf.Multiplied(move_to_center)

    apply_transform = BRepBuilderAPI_Transform(trsf_to_apply)
    apply_transform.Perform(body)
    transformed_body = apply_transform.ModifiedShape(body)

    return transformed_body


def check_manifold(body: TopoDS_Shape) -> bool:
    """Check that body is manifold."""
    faces = set()
    # Assumption: to create TopologyExplorer costs nothing
    for shell in TopologyExplorer(body, ignore_orientation=True).shells():
        for face in TopologyExplorer(shell, ignore_orientation=True).faces():
            if face in faces:
                return False
            faces.add(face)
    return True


def find_edges_from_wires(body: TopoDS_Shape) -> set[TopoDS_Edge]:
    """Return set of edges from Wires."""
    edge_set = set()
    for wire in TopologyExplorer(body, ignore_orientation=False).wires():
        for edge in WireExplorer(wire).ordered_edges():
            edge_set.add(edge)
    return edge_set


def find_edges_from_top_exp(body: TopoDS_Shape) -> set[TopoDS_Edge]:
    """Return set of edges from Shape."""
    return set(TopologyExplorer(body, ignore_orientation=False).edges())


def check_closed(body: TopoDS_Shape) -> bool:
    """Check the given body is closed.
    In Open Cascade, unlinked (open) edges can be identified
    as they appear in the edges iterator when ignore_orientation=False
    but are not present in any wire
    """
    edges_from_wires = find_edges_from_wires(body)
    edges_from_top_exp = find_edges_from_top_exp(body)

    missing_edges = edges_from_top_exp.symmetric_difference(edges_from_wires)
    return not missing_edges


def check_unique_coedges(body: TopoDS_Shape) -> bool:
    """Check the given body doesn't have coedges that used in a multiple
    loops. Additional check that the body is manifold?
    """
    coedge_set = set()
    for wire in TopologyExplorer(body, ignore_orientation=True).wires():
        for coedge in WireExplorer(wire).ordered_edges():
            oriented_edge = (coedge, coedge.Orientation())
            if oriented_edge in coedge_set:
                return False
            coedge_set.add(oriented_edge)
    return True
