# ---------------------------------------------------------------------
# Stage 1  –  Build the Python environment inside a slim base image
# ---------------------------------------------------------------------
FROM python:3.12.10-slim AS builder

# Install build tools only for the build stage
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc git && \
    rm -rf /var/lib/apt/lists/*

# Copy Python dependency spec and lock it into /wheels
WORKDIR /install
COPY requirements.txt .
RUN pip install --upgrade pip wheel && \
    pip wheel --wheel-dir=/wheels -r requirements.txt

# ---------------------------------------------------------------------
# Stage 2  –  Runtime image (minimal)
# ---------------------------------------------------------------------
FROM python:3.12.10-slim

# Install system libs that binary wheels such as numpy / spaCy expect
RUN apt-get update && apt-get install -y --no-install-recommends \
        libblas3 liblapack3 libpq5 libopenblas0 \
        curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy wheels from the builder stage and install them
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy models
COPY spacy/models/wikiann /app/spacy/models/wikiann

# Copy application code
WORKDIR /app
COPY pages/ pages/
COPY utils/ utils/
COPY Home.py .

# Make Streamlit listen on all interfaces in the container
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

EXPOSE 8501

# Entry point
CMD ["streamlit", "run", "Home.py", "--server.fileWatcherType", "none"]
