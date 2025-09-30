ARG FROM_IMAGE
FROM $FROM_IMAGE

# Set working directory
WORKDIR /app

# Install pandoc
RUN wget --no-check-certificate https://github.com/jgm/pandoc/releases/download/3.7.0.2/pandoc-3.7.0.2-linux-amd64.tar.gz -O pandoc-3.7.0.2-linux-amd64.tar.gz && \
    tar xzf pandoc-3.7.0.2-linux-amd64.tar.gz && \
    cp pandoc-3.7.0.2/bin/pandoc /usr/local/bin/ && \
    rm -rf pandoc-3.7.0.2*

RUN /opt/conda310/bin/pip install uv -i https://artlab.alibaba-inc.com/1/PYPI/simple/

RUN /opt/conda310/bin/uv pip install sphinx myst-nb jupyter nbconvert sphinx-autobuild \
     -i https://artlab.alibaba-inc.com/1/PYPI/simple/ \
     --extra-index-url=https://artlab.alibaba-inc.com/1/pypi/huiwa_rtp_internal \
     --extra-index-url=http://artlab.alibaba-inc.com/1/pypi/rtp_diffusion \
     --extra-index-url=https://artlab.alibaba-inc.com/1/PYPI/pytorch/whl \
     --trusted-host=artlab.alibaba-inc.com \
     --index-strategy unsafe-best-match \
     --cache-dir=/root/.cache/uv/ \
     --python=/opt/conda310/bin/python \
     --verbose

# Install additional utilities needed for documentation build
RUN yum update -y && \
    yum install -y moreutils --skip-broken || true && \
    yum install -y parallel --skip-broken || true && \
    rm -rf /var/cache/yum/*

# Copy documentation files
COPY docs/ /app/docs/

# Copy release version file
COPY rtp_llm/release_version.py /app/

# Install documentation requirements
RUN /opt/conda310/bin/uv pip install -r /app/docs/requirements.txt \
     -i https://artlab.alibaba-inc.com/1/PYPI/simple/ \
     --extra-index-url=https://artlab.alibaba-inc.com/1/pypi/huiwa_rtp_internal \
     --extra-index-url=http://artlab.alibaba-inc.com/1/pypi/rtp_diffusion \
     --extra-index-url=https://artlab.alibaba-inc.com/1/PYPI/pytorch/whl \
     --trusted-host=artlab.alibaba-inc.com \
     --index-strategy unsafe-best-match \
     --cache-dir=/root/.cache/uv/ \
     --python=/opt/conda310/bin/python \
     --verbose

# Set working directory to docs
WORKDIR /app/docs

# Expose port for documentation server
EXPOSE 20880

# Export PATH environment variable
ENV PATH="/opt/conda310/bin:$PATH"