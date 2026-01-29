#!/bin/bash

#######  error code specification  #########
# Please update this documentation if new error code is added.
# 1   => reserved for script error
# 2   => bad usage
# 3   => bad user
# 4   => service start failed
# 5   => preload.sh check failed
# 6   => hsf online failed
# 7   => nginx start failed
# 8   => status.taobao check failed
# 128 => exit with error message


PROG_NAME=$0
ACTION=$1

usage() {
    echo "Usage: $PROG_NAME {start|stop|online|offline|pubstart|restart|deploy}"
    exit 2 # bad usage
}

# gitlab pipeline run as root
# if [ "$UID" -eq 0 ]; then
#     echo "ERROR: can't run as root, please use: sudo -u admin $0 $@"
#     exit 3 # bad user
# fi

if [ $# -lt 1 ]; then
    usage
    exit 2 # bad usage
fi

APP_HOME=$(cd $(dirname ${BASH_SOURCE[0]})/..; pwd)
source "${APP_HOME}/bin/hook.sh"
source "${APP_HOME}/bin/setenv.sh"

# Try to find the tgz file - could be ${APP_NAME}.tgz or ai-whale.tgz
if [ -f "${APP_HOME}/target/${APP_NAME}.tgz" ]; then
    APP_TGZ_FILE=${APP_NAME}.tgz
elif [ -f "${APP_HOME}/target/ai-whale.tgz" ]; then
    APP_TGZ_FILE=ai-whale.tgz
elif ls "${APP_HOME}/target/${APP_NAME}"____*.tgz 1> /dev/null 2>&1; then
    # support b2b app, which tgz name like appName____hz.tgz
    echo "INFO: try to find ${APP_HOME}/target/${APP_NAME}____*.tgz"
    APP_TGZ_FILE=$(basename $(ls ${APP_HOME}/target/${APP_NAME}____*.tgz))
else
    die "app tgz file not found. Checked: ${APP_NAME}.tgz, ai-whale.tgz, ${APP_NAME}____*.tgz"
fi
echo "Using tgz file: ${APP_TGZ_FILE}"

# check previous pipe command exit code. if no zero, whill exit with previous pipe status.
check_first_pipe_exit_code() {
  local first_pipe_exit_code=${PIPESTATUS[0]};
  if test $first_pipe_exit_code -ne 0; then
    exit $first_pipe_exit_code;
  fi
}

die() {
    if [[ "$#" -gt 0 ]]; then
        echo "ERROR: " "$@"
    fi
    exit 128
}

printLogPathInfo() {
  echo "Please check deploy log: ${APP_HOME}/logs/${APP_NAME}_deploy.log"
  echo "Please check application stdout: ${SERVICE_OUT}"
}

exit1() {
  echo "exit code 1"
  printLogPathInfo
  exit 1
}

exit4() {
  echo "exit code 4"
  printLogPathInfo
  exit 4
}

exit5() {
  echo "exit code 5"
  printLogPathInfo
  exit 5
}


extract_tgz() {
    local tgz_path="$1"
    local dir_path="$2"

    echo "extract ${tgz_path}"
    echo "INFO: tgz_path=${tgz_path}, dir_path=${dir_path}"
    echo "INFO: current working directory: $(pwd)"
    
    cd "${APP_HOME}/target" || exit1
    echo "INFO: changed to directory: $(pwd)"
    
    # Show what's in the tgz file before extraction
    echo "INFO: Contents of ${tgz_path}:"
    tar -tzf "${tgz_path}" | head -10
    echo "INFO: (showing first 10 entries only)"
    
    rm -rf "${dir_path}" || exit1
    echo "INFO: removed existing directory ${dir_path}"
    
    tar xzf "${tgz_path}" || exit1
    echo "INFO: extraction completed"
    
    # Show what was actually extracted
    echo "INFO: Contents of target directory after extraction:"
    ls -la
    
    # in order to support fat.jar, unzip it.
    test -f "${dir_path}.jar" && unzip -q "${dir_path}.jar" -d "${dir_path}"
    # Check if we have a jar file to run directly
    if [ -f "${dir_path}.jar" ]; then
        echo "Found jar file ${dir_path}.jar, will use it for execution"
    fi
    
    echo "INFO: Looking for directory: ${dir_path}"
    if [ -d "${dir_path}" ]; then
        echo "INFO: Directory ${dir_path} exists"
    else
        echo "INFO: Directory ${dir_path} does NOT exist"
        echo "INFO: Available directories and files:"
        find . -maxdepth 2 -type d
        echo "INFO: All files and directories:"
        find . -maxdepth 3
    fi
    
    test -d "${dir_path}" || die "no directory: ${dir_path}"
    touch --reference "${tgz_path}" "${tgz_path}.timestamp" || exit1
}

update_target() {
    local tgz_name="$1"
    local dir_name="$2"

    local tgz_path="${APP_HOME}/target/${tgz_name}"
    local dir_path="${APP_HOME}/target/${dir_name}"

    local error=0
    # dir exists
    if [ -d "${dir_path}" ]; then
        # tgz exists
        if [ -f "${tgz_path}" ]; then
            local need_tar=0
            if [ ! -e "${tgz_path}.timestamp" ]; then
                need_tar=1
            else
                local tgz_time=$(stat -L -c "%Y" "${tgz_path}")
                local last_time=$(stat -L -c "%Y" "${tgz_path}.timestamp")
                if [ $tgz_time -gt $last_time ]; then
                    need_tar=1
                fi
            fi
            # tgz is new - extract_tgz
            if [ "${need_tar}" -eq 1 ]; then
                extract_tgz "${tgz_path}" "${dir_path}"
            fi
            # tgz is not new - return SUCCESS
        fi
        # tgz not exists - return SUCCESS
    # dir not exists
    else
        # tgz exists - extract_tgz
        if [ -f "${tgz_path}" ]; then
            extract_tgz "${tgz_path}" "${dir_path}"
        # tgz not exists - return FAIL
        else
            echo "ERROR: ${tgz_path} NOT EXISTS"
            error=1
        fi
    fi

    return $error
}

start_spring_boot() {
    # prepare_service_out
    # delete old SERVICE_OUT, keep last 20 logs
    ls -t "$SERVICE_OUT".* 2>/dev/null | tail -n +$((20 + 1)) | xargs --no-run-if-empty rm -f
    if [ -e "$SERVICE_OUT" ]; then
        mv "$SERVICE_OUT" "$SERVICE_OUT.$(date '+%Y%m%d%H%M%S')" || exit1
    fi
    mkdir -p "$(dirname "${SERVICE_OUT}")" || exit1
    touch "$SERVICE_OUT" || exit1

    echo "INFO: spring boot service log: $SERVICE_OUT"

    if [ ! -z "$SERVICE_PID" ]; then
        if [ -f "$SERVICE_PID" ]; then
            if [ -s "$SERVICE_PID" ]; then
                echo "Existing PID file found during start."
                if [ -r "$SERVICE_PID" ]; then
                    PID=`cat "$SERVICE_PID"`
                    ps -p $PID >/dev/null 2>&1
                    if [ $? -eq 0 ] ; then
                        echo "Service(spring boot) appears to still be running with PID $PID. Start aborted."
                        exit1
                    else
                        echo "Removing/clearing stale PID file."
                        rm -f "$SERVICE_PID" >/dev/null 2>&1
                        if [ $? != 0 ]; then
                            if [ -w "$SERVICE_PID" ]; then
                                cat /dev/null > "$SERVICE_PID"
                            else
                                echo "Unable to remove or clear stale PID file. Start aborted."
                                exit1
                            fi
                        fi
                    fi
                else
                    echo "Unable to read PID file. Start aborted."
                    exit1
                fi
            else
                rm -f "$SERVICE_PID" >/dev/null 2>&1
                if [ $? != 0 ]; then
                    if [ ! -w "$SERVICE_PID" ]; then
                        echo "Unable to remove or write to empty PID file. Start aborted."
                        exit1
                    fi
                fi
            fi
        fi
    fi

    # make sure work_dir is ${APP_HOME}
    cd ${APP_HOME}
    # Check if we should use jar file directly or classpath
    if [ -f "${APP_HOME}/target/${APP_NAME}.jar" ]; then
        echo "Using jar file ${APP_HOME}/target/${APP_NAME}.jar"
        eval exec "\"$JAVA_HOME/bin/java\"" $SERVICE_OPTS \
                -jar "\"${APP_HOME}/target/${APP_NAME}.jar\"" \
                -Dapp.location="\"${APP_HOME}/target/${APP_NAME}\"" \
                -Djava.io.tmpdir="\"$SERVICE_TMPDIR\"" \
                ${SPRING_BOOT_ARGS} "$@" \
                >> "$SERVICE_OUT" 2>&1 "&"
    else
        echo "Using classpath ${APP_HOME}/target/${APP_NAME}"
        eval exec "\"$JAVA_HOME/bin/java\"" $SERVICE_OPTS \
                -classpath "\"${APP_HOME}/target/${APP_NAME}\"" \
                -Dapp.location="\"${APP_HOME}/target/${APP_NAME}\"" \
                -Djava.endorsed.dirs="\"$JAVA_ENDORSED_DIRS\""  \
                -Djava.io.tmpdir="\"$SERVICE_TMPDIR\"" \
                org.springframework.boot.loader.JarLauncher ${SPRING_BOOT_ARGS} "$@" \
                >> "$SERVICE_OUT" 2>&1 "&"
    fi

    if [ ! -z "$SERVICE_PID" ]; then
        echo $! > "$SERVICE_PID"
    fi

    local exptime=0
    echo "INFO: Waiting for service to start..."
    while true
    do
        . "$APP_HOME/bin/preload.sh" $TOMCAT_PORT
        if [ $? -gt 0 ]; then
            sleep 1
            ((exptime++))
            echo -n -e "\rINFO: Service startup progress: ${exptime}s elapsed..."
            if [ `expr $exptime \% 10` -eq 0 ]; then
                ps -p `cat "$SERVICE_PID"` >/dev/null 2>&1
                if [ $? -gt 0 ] ; then
                    echo
                    echo "ERROR: Service process exited unexpectedly, startup failed."
                    exit4 # service start failed
                fi
            fi
        else
            echo
            echo "INFO: Service ${APP_NAME} started successfully and is ready to serve requests."
            break
        fi
    done
}

start_nginx() {
    echo "INFO: ${APP_NAME} try to start nginx..."
    if [ "${NGINX_SKIP}" -ne "1" ]; then
        echo "now start nginx..."
        "$NGINXCTL" start
        if [ "$?" == "0" ]; then
            echo "Nginx Start SUCCESS."
        else
            echo "Nginx Start Failed."
            exit 7 # nginx start failed
        fi
        # 判断是否ENV SKIP_CHECK_STATUS_TAOBAO="true"
        echo "SKIP_CHECK_STATUS_TAOBAO: ${SKIP_CHECK_STATUS_TAOBAO}"
        if test -n "${SKIP_CHECK_STATUS_TAOBAO}" && test "${SKIP_CHECK_STATUS_TAOBAO}" = "true"; then
            sleep ${STATUS_TAOBAO_WAIT_TIME}
            echo "skip check http://127.0.0.1:${STATUS_PORT}/status.taobao"
        elif test -n "${STATUS_PORT}"; then
            sleep ${STATUS_TAOBAO_WAIT_TIME}
            status_code=`/usr/bin/curl -L -o /dev/null --connect-timeout 5 -s -w %{http_code}  "http://127.0.0.1:${STATUS_PORT}/status.taobao"`
            if [ x$status_code != x200 ];then
                echo "check http://127.0.0.1:${STATUS_PORT}/status.taobao failed with status ${status_code} after wait ${STATUS_TAOBAO_WAIT_TIME} seconds."
                echo "You can adjust STATUS_TAOBAO_WAIT_TIME in setenv.sh"
                echo "See http://gitlab.alibaba-inc.com/middleware/apps-deploy/issues/31"
                exit 8 # status.taobao check failed
            fi
            echo "check http://127.0.0.1:${STATUS_PORT}/status.taobao success"
        fi
        echo "app online success"
    else
        echo "NGINX_SKIP=1 and ignore start nginx"
    fi
}

after_start_up() {
  now=`date "+%Y-%m-%d %H:%M:%S"`
  echo "INFO: ${APP_NAME} try to execute whale-handler after_start_up ... $now"
  echo "INFO: Calling /usr/bin/curl --max-time 5 -s -w '\n%{http_code}' 'http://localhost:7001/hook/after_start'"
  ret_str=`/usr/bin/curl --max-time 5 -s -w "\n%{http_code}" "http://localhost:7001/hook/after_start" 2>&1`
  http_code=$(echo "$ret_str" | tail -n1)
  response_body=$(echo "$ret_str" | head -n-1)
  echo "INFO: curl raw output: ${ret_str}"
  echo "INFO: http_code=${http_code}, response_body='${response_body}'"
  if [ x$response_body = x"success" ] || [ x$http_code = x"200" ]; then
    echo "check http://localhost:7001/hook/after_start success"
  else
    if [ x$http_code = x"000" ]; then
      echo "ERROR: Service port 7001 not ready or connection failed (http_code=000)"
    elif [ x$http_code = x"403" ]; then
      echo "ERROR: Request not allowed by isLocalRequest check (http_code=403)"
    elif [ x$http_code = x"500" ]; then
      echo "ERROR: gracefulOnlineService.online() threw exception (http_code=500)"
    fi
    echo "check http://localhost:7001/hook/after_start failed with http_code: ${http_code}, response: ${response_body}. will exit."
    exit 128
  fi
}

start_xagent() {
  if [[ "${ENABLE_XAGENT}" != "true" ]]; then
    echo "skip start xagent."
    return
  fi

  echo "start xagent..."
  # start xagent
  cd /home/admin/xagent/ && /home/tops/bin/supervisord -c /home/admin/xagent/supervisord.conf ;
  /home/tops/bin/supervisorctl -c /home/admin/xagent/supervisord.conf restart xagent
  echo "start xagent finished..."
}

stop_xagent() {
  if [[ "${ENABLE_XAGENT}" != "true" ]]; then
      echo "skip stop xagent."
      return
  fi

  echo "stop xagent..."
  /home/tops/bin/supervisorctl -c /home/admin/xagent/supervisord.conf stop xagent
  echo "stop xagent finished..."
}

start() {
    echo "INFO: ${APP_NAME} try to start..."
    echo "APP_TGZ_FILE=${APP_TGZ_FILE}, APP_NAME=${APP_NAME}"
    mkdir -p "${APP_HOME}/target" || exit1
    mkdir -p "${APP_HOME}/logs" || exit1
    mkdir -p "${SERVICE_TMPDIR}" || exit1
    HOME="$(getent passwd "$UID" | awk -F":" '{print $6}')" # fix "$HOME" by "$UID"
    echo "[start 1] start to unzip app tgz file..."
    update_target "${APP_TGZ_FILE}" "${APP_NAME}" || exit1

    beforeStartApp
    echo "[start 2] try to start spring boot..."
    start_spring_boot
    echo "INFO: ${APP_NAME} start success"
    cat /etc/instanceInfo | grep env_spas_accessKey
    afterStartApp
}

stop_tomcat() {
    if [ -f "$CATALINA_PID" ]; then
        echo "stop old tomcat..."
        local PID=$(cat "$CATALINA_PID")
        if kill -0 "$PID" 2>/dev/null; then
            "$CATALINA_HOME"/bin/catalina.sh stop $TOMCAT_STOP_WAIT_TIME -force
            mv /home/admin/logs/gc.log /home/admin/logs/gc.log.`date +%Y%m%d%H%M%S`
            echo "delete old tomcat pid file"
            rm -f "$CATALINA_PID" >/dev/null 2>&1
        fi
    else
        echo "no old tomcat, ignore this"
    fi
}

stop_spring_boot() {
    SLEEP=5
    FORCE=1

    PID=`cat "$SERVICE_PID"`

    # Try a normal kill.
    echo "Attempting to signal the process to stop through OS signal."
    kill -15 "$PID" >/dev/null 2>&1

    while [ $SLEEP -ge 0 ]; do
        kill -0 "$PID" >/dev/null 2>&1
        if [ $? -gt 0 ]; then
            rm -f "$SERVICE_PID" >/dev/null 2>&1
            if [ $? != 0 ]; then
                if [ -w "$SERVICE_PID" ]; then
                    cat /dev/null > "$SERVICE_PID"
                else
                    echo "The PID file could not be removed or cleared."
                fi
            fi
            echo "Service stopped."
            # If Service has stopped don't try and force a stop with an empty PID file
            FORCE=0
            break
        fi
        if [ $SLEEP -gt 0 ]; then
            sleep 1
        fi
        SLEEP=`expr $SLEEP - 1 `
    done

    KILL_SLEEP_INTERVAL=5
    if [ $FORCE -eq 1 ]; then
        if [ -f "$SERVICE_PID" ]; then
            PID=`cat "$SERVICE_PID"`
            echo "Killing Service with the PID: $PID"
            kill -9 $PID
            while [ $KILL_SLEEP_INTERVAL -ge 0 ]; do
                kill -0 `cat "$SERVICE_PID"` >/dev/null 2>&1
                if [ $? -gt 0 ]; then
                    rm -f "$SERVICE_PID" >/dev/null 2>&1
                    if [ $? != 0 ]; then
                        if [ -w "$SERVICE_PID" ]; then
                            cat /dev/null > "$SERVICE_PID"
                        else
                            echo "The PID file could not be removed."
                        fi
                    fi
                    # Set this to zero else a warning will be issued about the process still running
                    KILL_SLEEP_INTERVAL=0
                    echo "The Service process has been killed."
                    break
                fi
                if [ $KILL_SLEEP_INTERVAL -gt 0 ]; then
                    sleep 1
                fi
                KILL_SLEEP_INTERVAL=`expr $KILL_SLEEP_INTERVAL - 1 `
            done
            if [ $KILL_SLEEP_INTERVAL -gt 0 ]; then
                echo "Service has not been killed completely yet. The process might be waiting on some system call or might be UNINTERRUPTIBLE."
            fi
        fi
    fi
}

stop() {
    beforeStopApp
    echo "INFO: ${APP_NAME} try to stop..."
    if [[ -f ${APP_HOME}/target/${APP_NAME}/bin/appctl.sh ]]; then
        call_tappctl "stop"
    else
        # 0. stop xagent if need
        echo "[stop 0] try to stop xaxgent..."
        stop_xagent


        # 1. stop nginx
        echo "[stop 1] try to stop nginx..."
        "$NGINXCTL" stop

        # 2. stop old tomcat process
        echo "[stop 2] try stop old tomcat..."
        stop_tomcat

        # 3. stop spring boot
        echo "[stop 3] try stop spring boot..."
        if [ -f "$SERVICE_PID" ]; then
            if [ -s "$SERVICE_PID" ]; then
                kill -0 `cat "$SERVICE_PID"` >/dev/null 2>&1
                if [ $? -gt 0 ]; then
                    echo "PID file found but no matching process was found. Stop aborted."
                else
                    stop_spring_boot
                fi
            else
                echo "PID file is empty and has been ignored."
                rm -f "$SERVICE_PID" >/dev/null 2>&1
            fi
        else
            echo "\$SERVICE_PID was set but the specified file does not exist. Is Service running? Stop aborted."
        fi
    fi
    echo "INFO: ${APP_NAME} stop success"
    afterStopApp
}

backup() {
    if [ -f "${APP_HOME}/target/${APP_TGZ_FILE}" ]; then
        mkdir -p "${APP_HOME}/target/backup" || exit1
        tgz_time=$(date --reference "${APP_HOME}/target/${APP_TGZ_FILE}" +"%Y%m%d%H%M%S")
        local backup_file=$(basename ${APP_TGZ_FILE} .tgz)".$tgz_time"".tgz"
        cp -f "${APP_HOME}/target/${APP_TGZ_FILE}" "${APP_HOME}/target/${backup_file}"
    fi
}

main() {
    echo "=========================================================================================="
    echo "==============================[      $ACTION      ]======================================="
    echo "=========================================================================================="

    now=`date "+%Y-%m-%d %H:%M:%S"`
    echo "$now"
    echo "SERVICE_PID $SERVICE_PID"

    echo "INFO: deploy log: ${APP_HOME}/logs/${APP_NAME}_deploy.log"

    echo "INFO: ${APP_NAME} action: $ACTION"

    case "$ACTION" in
        start)
            start
        ;;
        stop)
            stop
        ;;
        pubstart)
            stop
            start
            start_nginx
            after_start_up
        ;;
        restart)
            stop
            start
            start_nginx
            after_start_up
            start_xagent
        ;;
        deploy)
            stop
            start
            start_nginx
            after_start_up
            backup
        ;;
        *)
            usage
        ;;
    esac
}

main | tee -a ${APP_HOME}/logs/${APP_NAME}_deploy.log; check_first_pipe_exit_code;
