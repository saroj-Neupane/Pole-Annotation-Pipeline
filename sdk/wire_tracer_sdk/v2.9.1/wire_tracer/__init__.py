"""Wire-tracer ONNX pipeline for desktop integration — V2.

Reconstructs the per-span wire structure (pole-A insulators <-> midspan wires <->
pole-B insulators) from three photos, using only ONNX models + numpy. No torch, no scipy,
no ultralytics on the destination machine.

V2 = the production operating point of src/wire_tracer: ONE unified joint-class pole model
(hardware x cable_type x crossarm-K) + a LEARNED pure-numpy edge-cost matcher. See the package
README for the full v1->v2 delta.
"""

from .pipeline import WireTracerPipeline

__all__ = ["WireTracerPipeline"]
__version__ = "2.8.0"
