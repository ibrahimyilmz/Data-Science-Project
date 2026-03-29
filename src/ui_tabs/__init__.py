"""UI tabs module."""

from .generation_tab import render_generation_tab
from .clustering_tab import render_clustering_tab
from .classification_tab import render_classification_tab
from .forecasting_tab import render_forecasting_tab
from .info_tab import render_info_tab

__all__ = [
    "render_generation_tab",
    "render_clustering_tab",
    "render_classification_tab",
    "render_forecasting_tab",
    "render_info_tab",
]
