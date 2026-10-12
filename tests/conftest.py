"""pytest 前置：必须在 import numpy 之前钉死 BLAS 线程，保证确定性测试有效。"""

import os

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_k, "1")
