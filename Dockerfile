FROM python:3.12-slim

WORKDIR /app

# 先装依赖（利用层缓存）
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY . .

# bit 级可复现：容器内同样钉死 BLAS 线程
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    PYTHONHASHSEED=0

RUN python -m pytest tests -q

CMD ["python", "-m", "scoreforge.examples.run_demo", "--fast"]
