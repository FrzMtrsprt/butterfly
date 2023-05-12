from __future__ import annotations

import os
import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from xml.dom import minidom


class Object(TypedDict):
    name: str
    pose: str
    truncated: bool
    difficult: bool
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class Annotation(TypedDict):
    folder: str
    filename: str
    path: str
    database: str
    width: int
    height: int
    depth: int
    segmented: bool
    objects: list[Object]


def parse_xml(file_name: str) -> Annotation:
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File not found: {file_name}")

    doc = minidom.parse(file_name)

    folder = doc.getElementsByTagName("folder")[0]
    filename = doc.getElementsByTagName("filename")[0]
    path = doc.getElementsByTagName("path")[0]
    database = doc.getElementsByTagName("database")[0]
    width = doc.getElementsByTagName("width")[0]
    height = doc.getElementsByTagName("height")[0]
    depth = doc.getElementsByTagName("depth")[0]
    segmented = doc.getElementsByTagName("segmented")[0]
    objects = doc.getElementsByTagName("object")
    object_list: list[Object] = []

    for obj in objects:
        name = obj.getElementsByTagName("name")[0]
        pose = obj.getElementsByTagName("pose")[0]
        truncated = obj.getElementsByTagName("truncated")[0]
        difficult = obj.getElementsByTagName("difficult")[0]
        xmin = obj.getElementsByTagName("xmin")[0]
        ymin = obj.getElementsByTagName("ymin")[0]
        xmax = obj.getElementsByTagName("xmax")[0]
        ymax = obj.getElementsByTagName("ymax")[0]

        object_list.append(Object(name=str(name.firstChild.data),
                                  pose=str(pose.firstChild.data),
                                  truncated=str(truncated.firstChild.data) == "1",
                                  difficult=str(difficult.firstChild.data) == "1",
                                  xmin=int(xmin.firstChild.data),
                                  ymin=int(ymin.firstChild.data),
                                  xmax=int(xmax.firstChild.data),
                                  ymax=int(ymax.firstChild.data)))

    return Annotation(folder=str(folder.firstChild.data),
                      filename=str(filename.firstChild.data),
                      path=str(path.firstChild.data),
                      database=str(database.firstChild.data),
                      width=int(width.firstChild.data),
                      height=int(height.firstChild.data),
                      depth=int(depth.firstChild.data),
                      segmented=str(segmented.firstChild.data) == "1",
                      objects=object_list)
