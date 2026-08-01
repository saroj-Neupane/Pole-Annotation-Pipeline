"""
Calibration pipeline — pole / ruler / ruler-keypoints / pole-top.

Public API::

    from calibration import CalibrationPipeline

    pipe = CalibrationPipeline()
    result = pipe.run("photo.jpg")                       # raw dict
    result = pipe.run(img, return_annotated=True)        # with annotated image

See ``CalibrationPipeline.run`` for the result schema.
"""

from .pipeline import CalibrationPipeline

__all__ = ["CalibrationPipeline"]
__version__ = "0.1.0"
