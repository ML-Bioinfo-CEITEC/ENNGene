#!/usr/bin/env python

import argparse
import os
import sys

parser = argparse.ArgumentParser()

parser.add_argument(
    "--path",
    required=True,
)

args = parser.parse_args(sys.argv[1:])

file_paths = []
for root, dirs, files in os.walk(args.path):
    print("root: " + root)
    for dir in dirs:
        print("dir: " + dir)
    for file in files:
        print("file: " + file)
        file_paths.append(os.path.join(root, file))

print(file_paths)
