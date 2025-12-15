set -x;

dir_to_find=$1;
conf_file=$2;

if [ -z "$dir_to_find" ]; then
    echo "Usage: $0 <dir_to_find>";
    exit 1;
fi

latest_run=$(ls -dt $dir_to_find/*/ | head -n 1);

echo "latest_run: $latest_run";

latest_dir=$latest_run/$(basename $(ls -dt $latest_run/*/ | head -n 1));

echo "latest_dir: $latest_dir";

script_dir=$(dirname $(readlink -f "$0"));

sh $script_dir/analyze_all.sh $latest_dir $conf_file;
