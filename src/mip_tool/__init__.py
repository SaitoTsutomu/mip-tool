from importlib.metadata import metadata

from .util import (
    add_line,
    add_lines,
    add_lines_conv,
    model2str,
    model2toml,
    monotone_decreasing,
    monotone_increasing,
    pulp_model2toml,
    random_model,
    read_toml,
    scipy_milp,
    show_model,
    toml2model,
    toml2pulp_model,
    write_toml,
)

__all__ = [
    "add_line",
    "add_lines",
    "add_lines_conv",
    "model2str",
    "model2toml",
    "monotone_decreasing",
    "monotone_increasing",
    "pulp_model2toml",
    "random_model",
    "read_toml",
    "scipy_milp",
    "show_model",
    "toml2model",
    "toml2pulp_model",
    "write_toml",
]
_package_metadata = metadata(__package__)
__version__ = _package_metadata["Version"]
__author__ = _package_metadata.get("Author-email", "")
