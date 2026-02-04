#!/bin/bash

APP_HOME=$(cd $(dirname ${BASH_SOURCE[0]})/..; pwd)
source "${APP_HOME}/bin/setenv.sh"

CHECK_PORT=$STATUS_PORT
if [ $1 -a "$1" -gt 0 ] 2>/dev/null; then
    CHECK_PORT=$1
fi

CURL_BIN=/usr/bin/curl
SPACE_STR="........................................................................"

OUTIF=`/sbin/route -n | tail -1  | sed -e 's/.* \([^ ]*$\)/\1/'`
HTTP_IP="http://`/sbin/ifconfig | grep -A1 ${OUTIF} | grep inet | awk '{print $2}' | sed 's/addr://g'`:${CHECK_PORT}"

#####################################
checkpage() {
    echo ""
    echo ${SPACE_STR}
    echo "INFO: ${APP_NAME} checking status... $(date '+%Y-%m-%d %H:%M:%S')"
    echo ${SPACE_STR}
    echo ""
    if [ ! -z "$SERVICE_PID" ]; then
        if [ -f "$SERVICE_PID" ]; then
            if [ -s "$SERVICE_PID" ]; then
                if [ -r "$SERVICE_PID" ]; then
                    PID=`cat "$SERVICE_PID"`
                    ps -p $PID >/dev/null 2>&1
                    if [ $? -eq 0 ] ; then
                        # Only check /hook/process_ok endpoint
                        health_check "/hook/process_ok" "${APP_NAME}" "200"
                        if [ $? -eq 0 ]; then
                            echo "INFO: ${APP_NAME} health check passed"
                            echo ${SPACE_STR}
                            echo "INFO: ${APP_NAME} check status [  OK  ]"
                            echo ${SPACE_STR}
                            status=1
                            return 0
                        else
                            echo "INFO: ${APP_NAME} health check failed, service not ready"
                        fi
                    fi
                fi
            fi
        fi
    fi
    echo "INFO: ${APP_NAME} check status [FAILED]"
    return 1
}

health_check() {
    # check health endpoint with HTTP status code
    echo "Starting health_check function"
    echo "TOMCAT_PORT=${TOMCAT_PORT}, NGINX_SKIP=${NGINX_SKIP}"
    
    # Always try health check regardless of port listening status
    # This allows checking even if ss command doesn't detect the port yet
    echo "INFO: checking health endpoint through port: ${CHECK_PORT}"

    URL=$1
    TITLE=$2
    EXPECTED_STATUS=$3
    echo "$CURL_BIN" "${HTTP_IP}${URL}"
    
    # Check HTTP status code
    HTTP_STATUS=`$CURL_BIN --silent --write-out '%{http_code}' --output /dev/null -m 120 "${HTTP_IP}${URL}" 2>&1`
    echo ""
    echo "$CURL_BIN" "${HTTP_IP}${URL}" " HTTP status: $HTTP_STATUS"
    
    if [ "$EXPECTED_STATUS" != "" ]; then
        if [ "$HTTP_STATUS" == "$EXPECTED_STATUS" ]; then
            echo "INFO: Health check passed with status $HTTP_STATUS"
            error=0
        else
            echo "ERROR: Health check failed. Expected: $EXPECTED_STATUS, Got: $HTTP_STATUS"
            error=1
        fi
    fi
    return $error
}

preload_htm() {
    # check preload.htm
    portret=`(/usr/sbin/ss -ln4 sport = :${TOMCAT_PORT}; /usr/sbin/ss -ln6 sport = :${TOMCAT_PORT}) | grep -c ":${TOMCAT_PORT}"`
    if [ $portret -ne 0 -a "${NGINX_SKIP}" -ne "1" ]; then
        echo "INFO: tomcat is running at port: ${TOMCAT_PORT}"
        if [ "$CHECK_PORT" == 80 ]; then
          echo "INFO: try to chek checkpreload.htm through nginx port: 80"
        else
          echo "INFO: try to check checkpreload.htm through tomcat port: ${CHECK_PORT}"
        fi

        URL=$1
        TITLE=$2
        CHECK_TXT=$3
        echo "$CURL_BIN" "${HTTP_IP}${URL}"
        if [ "$TITLE" == "" ]; then
            TITLE=$URL
        fi
        len=`echo $TITLE | wc -c`
        len=`expr 60 - $len`
        echo -n -e "$TITLE ...${SPACE_STR:1:$len}"
        TMP_FILE=`$CURL_BIN --silent -m 150 "${HTTP_IP}${URL}" 2>&1`
        echo ""
        echo "$CURL_BIN" "${HTTP_IP}${URL}" " return: "
        echo "$TMP_FILE"
        if [ "$CHECK_TXT" != "" ]; then
            checkret=`echo "$TMP_FILE" | fgrep "$CHECK_TXT"`
            if [ "$checkret" == "" ]; then
                echo "ERROR: Please make sure checkpreload.htm return: success"
                error=1
            else
                error=0
            fi
        fi
        return $error
    fi

    return 0
}
#####################################
checkpage
