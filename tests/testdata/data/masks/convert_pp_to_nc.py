# -*- coding: utf-8 -*-
"""
Pratiman Patel.

This converts PP file to NC file to generate mask.

"""

import iris
import sys


def main(file, varname):
    """
    Convert PP file variable into NC file.

    Parameters
    ----------
    file : str
        Path for PP file
    varname : str
        Variable name in the PP file

    Returns
    -------
    None.

    """
    print(f"Processing File: {file}")
    cubes = iris.load(file, varname)
    print(f"Saving file: {varname}.nc")
    iris.save(cubes, f"{varname}.nc")


if __name__ == "__main__":
    file = sys.argv[1]
    varname = sys.argv[2]
    main(file, varname)
