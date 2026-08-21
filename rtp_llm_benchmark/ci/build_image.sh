# hub.docker.alibaba-inc.com/isearch/rtp_llm_dev_rocm:2026_01_09_16_41_850f85e
# for image: kir.alibaba-inc.com/kis-docker-hub/amd:alios8u-x86-rocm_6.4.3-rtp_llm_dev_2025_09_17_10_31_24
# workspace下需要有这个文件夹：
# rtp-llm-private

GITHUB_WORKSPACE=/mnt/raid0/yuzho/0119_newimage
RTP_PATH=${GITHUB_WORKSPACE}/rtp-llm

# some temporary changes to adapt to our env
# grep -rlE '(-i https://artifacts\.antgroup-inc\.cn/simple/|--extra-index-url=https://artlab\.alibaba-inc\.com/1/PYPI/py-central/|--extra-index-url=https://artlab\.alibaba-inc\.com/1/PYPI/pytorch/|--extra-index-url=http://artlab\.alibaba-inc\.com/1/pypi/rtp_diffusion|--trusted-host=artlab\.alibaba-inc\.com)' $RTP_PATH \
# | xargs -r sed -i \
#   -e 's|-i https://artifacts\.antgroup-inc\.cn/simple/||g' \
#   -e 's|--extra-index-url=https://artlab\.alibaba-inc\.com/1/PYPI/py-central/||g' \
#   -e 's|--extra-index-url=https://artlab\.alibaba-inc\.com/1/PYPI/pytorch/||g' \
#   -e 's|--extra-index-url=http://artlab\.alibaba-inc\.com/1/pypi/rtp_diffusion||g' \
#   -e 's|--trusted-host=artlab\.alibaba-inc\.com||g'
# sed -i '/"amdsmi",/d' $RTP_PATH/rtp_llm/BUILD
# sed -i 's|"@//:using_rocm": \["pyrsmi", "amdsmi"\],|"@//:using_rocm": ["pyrsmi"],|g' $RTP_PATH/rtp_llm/BUILD
sed -i 's/^/#/' $RTP_PATH/bazel/bazel_downloader.cfg

wget https://github.com/Kitware/CMake/releases/download/v3.31.0/cmake-3.31.0-linux-x86_64.tar.gz
tar -xzf cmake-3.31.0-linux-x86_64.tar.gz
sudo mv cmake-3.31.0-linux-x86_64 /opt/cmake/cmake-3.31.0
rm -rf /opt/cmake/cmake-3.26.4
echo 'export PATH=/opt/cmake/cmake-3.31.0/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
cmake --version
echo 'export PATH=/opt/cmake/cmake-3.31.0/bin:$PATH' \
  > /etc/profile.d/cmake.sh
chmod +x /etc/profile.d/cmake.sh

cd $RTP_PATH
unset PIP_EXTRA_INDEX_URL
unset PIP_INDEX_URL
sed -i 's|^\s*index-url\s*=.*|#&|'  ~/.config/pip/pip.conf
# dnf --disablerepo="*" --enablerepo=alinux3-os install -y libdrm-devel
yum --disablerepo="*" --enablerepo=alinux3-os install -y patch
yum --disablerepo="*" --enablerepo=alinux3-os install -y jq
yum --disablerepo="*" --enablerepo=alinux3-updates install -y openblas openblas-devel

# /opt/conda310/bin/python -m pip install /mnt/raid0/yuzho/BACKUPS/torch-2.8.0+git1a24a85-cp310-cp310-linux_x86_64.whl
# /opt/conda310/bin/python -m pip install /mnt/raid0/yuzho/BACKUPS/torchvision-0.22.1+59a3e1f-cp310-cp310-linux_x86_64.whl
/opt/conda310/bin/python -m pip install ninja -i https://pypi.org/simple/
/opt/conda310/bin/python3 -m pip install /mnt/raid0/yuzho/BACKUPS/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl # flash attn whl
# /opt/conda310/bin/python -m pip install flash_attn --no-build-isolation --index-url https://pypi.org/simple

/opt/conda310/bin/python3 -m pip install -r ./open_source/deps/requirements_rocm.txt
bazelisk build //rtp_llm:rtp_llm_lib --jobs 150 --verbose_failures --config=rocm
/opt/conda310/bin/python3 -m pip install recommonmark sphinx-markdown-tables sphinx_pdj_theme
/opt/conda310/bin/python3 -m pip uninstall -y aiter
bazelisk clean --expunge
pip cache purge
/opt/conda310/bin/python -m pip install --upgrade pip --index-url https://pypi.org/simple
