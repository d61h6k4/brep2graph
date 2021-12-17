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
"""Tool to visualize the given STEP file as a mesh and corresponding graph."""

import argparse
import pathlib

import networkx as nx
import matplotlib.pyplot as plt

from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Display.WebGl.threejs_renderer import ThreejsRenderer

import brep2graph
import brep2graph.utils


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--step-file",
                        type=pathlib.Path,
                        help="Specify path to the CAD file in STEP format",
                        required=True)
    parser.add_argument(
        "--artifacts",
        type=pathlib.Path,
        help="Specify path to the folder to store artifacts of the program.",
        required=True)

    subparsers = parser.add_subparsers()
    parser_cad = subparsers.add_parser("cad")
    parser_cad.set_defaults(func=show_cad)

    parser_scene = subparsers.add_parser("scene")
    parser_scene.set_defaults(func=show_scene)

    return parser.parse_args()


def show_cad(body: TopoDS_Shape, artifacts: pathlib.Path):
    """Show the CAD model."""
    ren = ThreejsRenderer(path=artifacts)
    ren.DisplayShape(body, export_edges=True)
    ren.render()


def show_scene(body: TopoDS_Shape, artifacts: pathlib.Path):
    """Show the CAD model as a mesh with the graph."""
    g = create_graph(body)

    fig, ax = plt.subplots()
    nx.draw(g, pos=nx.spectral_layout(g), ax=ax, labels={})
    fig.savefig(str(artifacts / "graph.png"))


def create_graph(body: TopoDS_Shape):

    graph = brep2graph.graph_from_brep(body)

    g = nx.DiGraph()

    for (u, v) in enumerate(graph["senders"]):
        g.add_edge(u, v)

    for (u, v) in enumerate(graph["receivers"]):
        g.add_edge(u, v)
    return g


def main():
    args = parse_args()
    if not args.artifacts.exists():
        args.artifacts.mkdir()

    loaded_body = brep2graph.utils.load_body(args.step_file)
    args.func(loaded_body, args.artifacts)


if __name__ == "__main__":
    main()
