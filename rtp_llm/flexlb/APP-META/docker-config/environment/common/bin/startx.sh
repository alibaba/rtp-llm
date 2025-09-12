#!/bin/sh
STARTUP=/home/admin/start.sh
STOP=/home/admin/stop.sh

listen_signal() {
    echo "trap to listen signal"
    trap do_stop SIGUSR1 15
}

detect_app_name() {
    if [ "x$BUILD_APP_NAME" == "x" ];then
        local n=`cat $STARTUP | grep 'jbossctl restart' | grep '/home/admin' | head -1 | awk -F '/' '{print $4}'`
        if [ "x$n" == "x" ];then
            echo `cat $STARTUP | grep '/home/admin' | head -1 | awk -F '/' '{print $4}'`
        else
            echo "$n"
        fi
    else
        # not working with hippo
        echo "$BUILD_APP_NAME"
    fi
}

prepare() {
    local app=`detect_app_name`
    if [ "x$app" == "x" ];then
        echo "detect app name failed,  only support AONE docker." && exit 1
    fi
    echo "detected app name: $app"
    local loc=/home/admin/$app
    sudo chown admin:admin $loc -R
    sudo chown admin:admin /home/admin/cai/ -R
    chmod +x /home/admin/cai/bin/nginxctl
    chmod +x $STARTUP
    echo "prepare done"
}

do_stop() {
    echo "to stop app..."
    sh $STOP
}

do_start() {
    prepare
    setsid $STARTUP
    local r=$?
    echo "$STARTUP exit code $r"
    [[ $r -ne 0 ]] && echo "start failed, exit" && exit 1
}

_test() {
    local STARTUP=./a
    if [ "x$BUILD_APP_NAME" == "x" ];then
        local n=`cat $STARTUP | grep 'jbossctl restart' | grep '/home/admin' | head -1 | awk -F '/' '{print $4}'`
        if [ "x$n" == "x" ];then
            echo `cat $STARTUP | grep '/home/admin' | head -1 | awk -F '/' '{print $4}'`
        else
            echo "$n"
        fi
    else
        # not working with hippo
        echo "$BUILD_APP_NAME"
    fi
    exit 1
}

listen_signal
do_start
while true; do
    # TODO: check process alive
    sleep 1
done