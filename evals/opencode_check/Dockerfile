# OpenCode with the arkive-rag MCP server, for `python -m evals.opencode_check`.
#
# Base: the official OpenCode image, pinned by version and digest. Its `opencode` binary is
# linked against musl, so it stays on its own Alpine base; Alpine 3.24 ships Python 3.14.
# The build context is assembled by the runner: this file, opencode.json, run-scenario.sh,
# the arkive-rag wheel built from the checkout and requirements.txt exported from uv.lock with
# hashes. No secret, snapshot or corpus is part of the image.
FROM ghcr.io/astral-sh/uv:0.13.5 AS uv

FROM ghcr.io/anomalyco/opencode:2.3.0@sha256:998e3e184c650e603d8bacb5cbeaeb090837078262c25521a7136b0375d2b652

RUN apk add --no-cache 'python3~3.14'
COPY --from=uv /uv /usr/local/bin/uv
COPY requirements.txt /tmp/arkive/requirements.txt
COPY arkive_rag-*.whl /tmp/arkive/
RUN uv venv --python /usr/bin/python3 /opt/arkive \
    && uv pip install --python /opt/arkive/bin/python --require-hashes --only-binary :all: \
        -r /tmp/arkive/requirements.txt \
    && uv pip install --python /opt/arkive/bin/python --no-deps /tmp/arkive/arkive_rag-*.whl \
    && rm -rf /tmp/arkive /root/.cache
ENV PATH=/opt/arkive/bin:$PATH \
    UV_NO_CACHE=1

# The system configuration file outranks global and project configuration.
COPY opencode.json /etc/opencode/opencode.json
COPY --chmod=755 run-scenario.sh /usr/local/bin/run-scenario

# The runner starts each container as the invoking host user (`docker run --user UID:GID`),
# which has no account here; run-scenario keeps HOME and the working directory under /tmp.
# The default user only applies to a container started by hand.
RUN mkdir /snapshots
USER 65534:65534
WORKDIR /tmp
