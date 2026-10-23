FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
# CPU-only PyTorch keeps the image small enough for edge boxes
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY defect_vision ./defect_vision
# Train inside the build so the image ships with weights (replace with your own trained artifacts in production)
RUN python -m defect_vision.train --samples 4000 --epochs 12 --out artifacts
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "defect_vision.api:app", "--host", "0.0.0.0", "--port", "8000"]
