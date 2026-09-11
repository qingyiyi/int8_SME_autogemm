"""INT8 SME CBLAS driver generation support for autoGEMM.

The backend bundles the validated SME assembly sources and generates the C
dispatcher that owns packing, thread partitioning, and calls the existing
packed SME kernel ABI.  Candidate builds therefore do not require externally
precompiled kernel objects.
"""
