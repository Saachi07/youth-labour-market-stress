FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8050
ENV PYTHONUNBUFFERED=1
EXPOSE 8050
CMD ["python", "main.py"]