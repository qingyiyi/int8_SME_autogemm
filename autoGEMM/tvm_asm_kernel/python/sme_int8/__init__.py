"""INT8 SME CBLAS driver generation support for autoGEMM.

The first backend deliberately keeps the SME assembly kernels external.  It
generates the C dispatcher that owns packing, thread partitioning, and calls
the existing packed SME kernel ABI.
"""

