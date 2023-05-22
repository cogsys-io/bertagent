#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Top-level package for bertagent."""

# fmt: off
from . import _version
__version__ = _version.get_versions()['version']
__version_dict__ = _version.get_versions()
# fmt: on

__author__ = """cogsys.io"""
__email__ = "bertagent@cogsys.io"


def get_module_version():
    return __version__


# end
