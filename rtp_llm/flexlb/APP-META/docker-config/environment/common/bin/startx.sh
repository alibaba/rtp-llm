#!/bin/bash
STARTUP=/home/admin/start.sh
STOP=/home/admin/stop.sh

listen_signal() {
    echo "trap to listen signal"
    trap do_stop SIGUSR1 TERM INT
}

detect_app_name() {
    if [ "x$BUILD_APP_NAME" = "x" ];then
        local n=`cat $STARTUP | grep 'jbossctl restart' | grep '/home/admin' | head -1 | awk -F '/' '{print $4}'`
        if [ "x$n" = "x" ];then
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
    if [ "x$app" = "x" ];then
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
    # Repeated signals must not interrupt the synchronous stop/drain operation.
    trap '' SIGUSR1 TERM INT
    echo "to stop app: supervisor pid=$$; waiting for Java shutdown..."
    if /bin/bash "$STOP"; then
        echo "application stopped; exiting container supervisor"
        exit 0
    fi
    # Exiting here would let the runtime destroy a JVM that is still draining.
    # Keep the supervisor alive; the platform termination deadline remains the bound.
    echo "ERROR: application stop failed; keeping supervisor alive for termination grace period" >&2
}

do_start() {
    prepare
    setsid $STARTUP
    local r=$?
    echo "$STARTUP exit code $r"
    [ "$r" -ne 0 ] && echo "start failed, exit" && exit 1
}

_test() {
    local STARTUP=./a
    if [ "x$BUILD_APP_NAME" = "x" ];then
        local n=`cat $STARTUP | grep 'jbossctl restart' | grep '/home/admin' | head -1 | awk -F '/' '{print $4}'`
        if [ "x$n" = "x" ];then
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
failed_checks=0
while true; do
    if ss -H -ltn 'sport = :7001' | grep -q .; then
        failed_checks=0
    else
        failed_checks=$((failed_checks + 1))
        echo "Java port 7001 is not listening ($failed_checks/3)" >&2
        if [ "$failed_checks" -ge 3 ]; then
            echo "Java port 7001 failed 3 consecutive checks, exiting" >&2
            exit 1
        fi
    fi
    sleep 1
done
