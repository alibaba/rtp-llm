git clone git@github.com:fzyzcjy/DeepEP.git;
cd DeepEP;
git checkout gb200_blog_part_2;


export MASTER_ADDR=11.13.195.78
export MASTER_PORT=32343
export WORLD_SIZE=1
export RANK=0
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
python tests/test_low_latency.py --num-processes 4 --allow-mnnvl


export RANK=0
python tests/test_low_latency.py --num-processes 4 --allow-mnnvl

