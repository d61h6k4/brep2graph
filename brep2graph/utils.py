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

from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Extend.DataExchange import read_step_file


def load_body(step_filepath: pathlib.Path) -> TopoDS_Shape:
    """Load the body from the given STEP file.
    """
    return read_step_file(str(step_filepath.absolute()), as_compound=False)
