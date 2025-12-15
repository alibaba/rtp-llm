set -x;

analyze_dir=$1
conf_file=$2

# 获取脚本所在目录的绝对路径
script_dir=$(dirname $(readlink -f "$0"))

if [ -z "$analyze_dir" ]; then
    echo "Usage: $0 <analyze_dir> [module_conf]"
    exit 1
fi

if [ ! -d "$analyze_dir" ]; then
    echo "Error: $analyze_dir is not a directory"
    exit 1
fi

dir_name=$(basename $analyze_dir)
output_dir=$analyze_dir/$dir_name_analyzed

mkdir -p $output_dir
echo "" > $output_dir/analyze.log
echo "Analyzing $analyze_dir..."
if [ -n "$conf_file" ]; then
    python $script_dir/batch_analyze.py $analyze_dir -o $output_dir --module-conf "$conf_file" | tee -a $output_dir/analyze.log
else
    python $script_dir/batch_analyze.py $analyze_dir -o $output_dir | tee -a $output_dir/analyze.log
fi
echo "Merging decode results..."
python $script_dir/merge_decode_results.py $analyze_dir -o $output_dir/decode_result_merged.csv | tee -a $output_dir/analyze.log


