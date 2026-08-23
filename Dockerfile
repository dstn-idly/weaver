FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    HOME=/tmp/weaver-home

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && scrapling install --force \
    && groupadd --gid 10001 weaver \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin weaver \
    && mkdir -p /ms-playwright /app/data \
    && chown -R weaver:weaver /ms-playwright /app

COPY --chown=weaver:weaver . .
RUN mkdir -p /app/data/runs && chown -R weaver:weaver /app/data

USER weaver

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=4s --start-period=25s --retries=3 \
  CMD curl --fail --silent http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "weaver.app:app", "--host", "0.0.0.0", "--port", "8000"]
