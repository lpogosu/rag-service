# Build stage: install dependencies into a virtualenv that the runtime stage copies.
# Splitting it keeps compilers and build headers out of the shipped image.
FROM python:3.11.14-slim-bookworm AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies are declared in pyproject, so copy it alone first: the layer is then
# reused on every code change, which is most of them.
COPY pyproject.toml README.md ./
RUN mkdir -p rag eval api \
    && touch rag/__init__.py eval/__init__.py api/__init__.py \
    && pip install --no-cache-dir .

COPY rag ./rag
COPY eval ./eval
COPY api ./api
RUN pip install --no-cache-dir --no-deps --force-reinstall .


FROM python:3.11.14-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    RAG_CONFIG=config/offline.yaml

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin rag

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
COPY --chown=rag:rag rag ./rag
COPY --chown=rag:rag eval ./eval
COPY --chown=rag:rag api ./api
COPY --chown=rag:rag config ./config

USER rag
EXPOSE 8000

# No shell form: the process must receive SIGTERM directly so in-flight streams
# get a chance to finish instead of being killed with the shell.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
