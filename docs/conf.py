"""Build guides and API references from the installed analytics package."""

import tomllib
from pathlib import Path
from typing import cast

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
metadata: dict[str, object] = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
project_metadata: dict[str, object] = cast(dict[str, object], metadata["project"])
project = "HomeSight Analytics"
release: str = str(project_metadata["version"])
version = release

extensions = [
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["examples"]
html_theme = "alabaster"
html_title = f"{project} {release}"
html_show_sourcelink = False
html_show_copyright = False
html_static_path = ["_static"]
html_css_files = ["diagrams.css"]
html_sidebars = {"**": ["about.html", "navigation.html", "relations.html", "searchbox.html"]}
myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence"]
myst_fence_as_directive = ["mermaid"]
mermaid_version = "11.12.1"
mermaid_height = "auto"
mermaid_light_theme = "neutral"

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    # Include domain fields and methods, without Pydantic configuration/validation machinery.
    "exclude-members": (
        "model_config,validate_credentials,validate_mode_transport,validate_backend_inputs,"
        "validate_camera_names,validate_motion_decision,normalize_utc,validate_segment_bounds,"
        "validate_vectors"
    ),
}
autodoc_typehints = "signature"
autodoc_member_order = "bysource"
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
