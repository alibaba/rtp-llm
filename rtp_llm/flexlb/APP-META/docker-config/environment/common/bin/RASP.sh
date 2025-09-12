#!/bin/bash

# RASP 安装脚本
# 联系: 简柏

mkdir -p /home/admin/rasp/logs

RASP_SH=/home/admin/rasp/RASP_Deploy.sh
RASP_Deploy='http://jam-security.bohr.alibaba-inc.com/deploy/download/engine?ip=2.2.2.2&osVersion=all&deployMode=static_auto_repair_deploy_sh&cpuArch=all'
function downloadRaspDeploy()
{
    echo 'Download rasp_deploy.sh'

    sudo rm -rf $RASP_SH
    
    sudo wget -c -O $RASP_SH $RASP_Deploy
	
	sudo chmod +x $RASP_SH
}

function start()
{
		downloadRaspDeploy
	  sudo $RASP_SH start > /home/admin/rasp/logs/deploy.log
	  source /home/admin/.jam_bashrc
}

function stop()
{
		downloadRaspDeploy
	  sudo $RASP_SH stop > /home/admin/rasp/logs/deploy.log
	  source /home/admin/.jam_bashrc
}

case "$1" in
    start)
      start
      ;;
    stop)
      stop
      ;;
    *)
    echo "Usage: $0 {start|stop}"
esac





