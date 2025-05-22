ARG FROM_IMAGE
FROM $FROM_IMAGE

COPY buildfarm-server_deploy.jar \
    server_config.yml.tpl \
    run_server.sh \
    buildfarm-shard-worker_deploy.jar \
    worker_config.yml.tpl \
    run_worker.sh \
    device_resource.py /buildfarm/

RUN yum install ajdk17 -bcurrent -y

RUN /opt/conda310/bin/pip install filelock --index-url=https://artlab.alibaba-inc.com/1/PYPI/simple/ --trusted-host=artlab.alibaba-inc.com
