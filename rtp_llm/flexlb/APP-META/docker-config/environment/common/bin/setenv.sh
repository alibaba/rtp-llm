#!/bin/bash

java_version() {
    # maybe 1.8.0_162 , 11-ea
    local local_java_version

    local IFS=$'\n'
    # remove \r for Cygwin
    local lines=$("${JAVA_HOME}"/bin/java -version 2>&1 | tr '\r' '\n')
    for line in $lines; do
      if [[ (-z $local_java_version) && ($line = *"version \""*) ]]
      then
        local ver=$(echo $line | sed -e 's/.*version "\(.*\)"\(.*\)/\1/; 1q')
        # on macOS, sed doesn't support '?'
        if [[ $ver = "1."* ]]
        then
          local_java_version=$(echo $ver | sed -e 's/1\.\([0-9]*\)\(.*\)/\1/; 1q')
        else
          local_java_version=$(echo $ver | sed -e 's/\([0-9]*\)\(.*\)/\1/; 1q')
        fi
      fi
    done
    echo "$local_java_version"
}

available_cpu_count() {
    local proc_stat_file=${1:-/proc/stat}
    local processor_count=${SIGMA_MAX_PROCESSORS_LIMIT:-}
    if [[ ! "$processor_count" =~ ^[1-9][0-9]*$ ]]; then
        processor_count=$(grep -cE '^cpu[0-9]+[[:space:]]' "$proc_stat_file" 2>/dev/null)
    fi
    if [[ ! "$processor_count" =~ ^[1-9][0-9]*$ ]]; then
        processor_count=1
    fi
    echo "$processor_count"
}

available_memory_mb() {
    local meminfo_file=${1:-/proc/meminfo}
    local cgroup_v2_file=${2:-/sys/fs/cgroup/memory.max}
    local cgroup_v1_file=${3:-/sys/fs/cgroup/memory/memory.limit_in_bytes}
    local total_mb limit_file limit_bytes limit_mb
    total_mb=$(awk '/^MemTotal:/ {printf "%d", $2 / 1024}' "$meminfo_file" 2>/dev/null)
    if [[ ! "$total_mb" =~ ^[1-9][0-9]*$ ]]; then
        echo "WARN: cannot read available memory; using a conservative 1024MB JVM budget" >&2
        total_mb=1024
    fi
    for limit_file in "$cgroup_v2_file" "$cgroup_v1_file"; do
        if [[ ! -r "$limit_file" ]]; then
            continue
        fi
        limit_bytes=$(cat "$limit_file")
        # 'max' and huge v1 unlimited values do not restrict physical memory.
        if [[ "$limit_bytes" == "max" || "$limit_bytes" =~ ^[0-9]{19,}$ ]]; then
            continue
        fi
        if [[ "$limit_bytes" =~ ^[0-9]{1,18}$ ]]; then
            limit_mb=$((limit_bytes / 1048576))
            if [[ "$limit_mb" -gt 0 && "$limit_mb" -lt "$total_mb" ]]; then
                total_mb=$limit_mb
            fi
        else
            echo "WARN: invalid cgroup memory limit; using at most a 1024MB JVM budget" >&2
            if [[ "$total_mb" -gt 1024 ]]; then
                total_mb=1024
            fi
        fi
    done
    echo "$total_mb"
}

configure_default_jvm_memory() {
    local total_mb=$1
    local heap_limit_mb
    # Thread stacks, GC bookkeeping and native libraries are not covered by the
    # explicit JVM pools below. Reserve one eighth of the container for them.
    NATIVE_MEMORY_HEADROOM_MB=$((total_mb / 8))
    maxMetaspace=512m
    reservedCodeCache=512m
    if [ "$total_mb" -le 2048 ]; then
        DEFAULT_JVM_XMS="$((total_mb / 2))m"
        maxDirectMemory="$((total_mb / 8))m"
        maxMetaspace="$((total_mb / 8))m"
        reservedCodeCache="$((total_mb / 16))m"
    elif [ "$total_mb" -le 16384 ]; then
        DEFAULT_JVM_XMS="$((total_mb * 5 / 8))m"
        maxDirectMemory="$((total_mb / 16))m"
        if [ "$total_mb" -eq 16384 ]; then
            maxDirectMemory=2g
        fi
        maxMetaspace="$((total_mb / 32))m"
        reservedCodeCache="$((total_mb / 32))m"
    elif [ "$total_mb" -le 24576 ]; then
        # The 12c24g ASI pool exposes about 19GiB to the container.
        DEFAULT_JVM_XMS=12g
        maxDirectMemory=2g
    elif [ "$total_mb" -le 32768 ]; then
        DEFAULT_JVM_XMS=18g
        maxDirectMemory=2g
    else
        DEFAULT_JVM_XMS=32g
        maxDirectMemory=2g
    fi
    heap_limit_mb=$((total_mb - NATIVE_MEMORY_HEADROOM_MB
        - $(jvm_memory_mb "$maxDirectMemory") - $(jvm_memory_mb "$maxMetaspace")
        - $(jvm_memory_mb "$reservedCodeCache")))
    if [ "$(jvm_memory_mb "$DEFAULT_JVM_XMS")" -gt "$heap_limit_mb" ]; then
        # A cgroup limit just above a profile boundary must not inherit a heap
        # that consumes the whole container before direct/native allocations.
        DEFAULT_JVM_XMS="${heap_limit_mb}m"
    fi
    DEFAULT_JVM_XMX=$DEFAULT_JVM_XMS
}

jvm_memory_mb() {
    local size=$1 number unit
    if [[ ! "$size" =~ ^([0-9]{1,12})([kKmMgG]?)$ ]]; then
        return 1
    fi
    number=$((10#${BASH_REMATCH[1]}))
    unit=${BASH_REMATCH[2]}
    if [ "$number" -le 0 ]; then
        return 1
    fi
    case "$unit" in
        g|G) echo "$((number * 1024))" ;;
        m|M) echo "$number" ;;
        k|K) echo "$(((number + 1023) / 1024))" ;;
        *) echo "$(((number + 1048575) / 1048576))" ;;
    esac
}

validate_jvm_memory_budget() {
    local total_mb=$1 heap_start=$2 heap_max=$3
    local heap_start_mb heap_max_mb budget_mb
    if [[ ! "$heap_start" =~ ^[0-9]+[kKmMgG]$ ]]; then
        echo "ERROR: invalid JVM initial heap size: $heap_start; include a k/m/g unit, for example 2048m or 2g" >&2
        return 1
    fi
    if [[ ! "$heap_max" =~ ^[0-9]+[kKmMgG]$ ]]; then
        echo "ERROR: invalid JVM maximum heap size: $heap_max; include a k/m/g unit, for example 2048m or 2g" >&2
        return 1
    fi
    heap_start_mb=$(jvm_memory_mb "$heap_start") || {
        echo "ERROR: invalid JVM initial heap size: $heap_start" >&2
        return 1
    }
    heap_max_mb=$(jvm_memory_mb "$heap_max") || {
        echo "ERROR: invalid JVM maximum heap size: $heap_max" >&2
        return 1
    }
    if [ "$heap_start_mb" -gt "$heap_max_mb" ]; then
        echo "ERROR: JVM initial heap $heap_start exceeds maximum heap $heap_max" >&2
        return 1
    fi
    budget_mb=$((heap_max_mb + $(jvm_memory_mb "$maxDirectMemory")
        + $(jvm_memory_mb "$maxMetaspace") + $(jvm_memory_mb "$reservedCodeCache")
        + NATIVE_MEMORY_HEADROOM_MB))
    if [ "$budget_mb" -gt "$total_mb" ]; then
        echo "ERROR: JVM memory budget ${budget_mb}MB exceeds container limit ${total_mb}MB: heap=$heap_max direct=$maxDirectMemory metaspace=$maxMetaspace code_cache=$reservedCodeCache native_headroom=${NATIVE_MEMORY_HEADROOM_MB}MB" >&2
        return 1
    fi
}

# SETENV_SETTED promise run this only once.
if [ -z $SETENV_SETTED ]; then
    SETENV_SETTED="true"

    # app
    # set ${APP_NAME}, if empty $(basename "${APP_HOME}") will be used.
    APP_HOME=$(cd $(dirname ${BASH_SOURCE[0]})/.. || exit; pwd)
    # Force APP_NAME to be FlexLB to match tgz structure
    APP_NAME=FlexLB
    echo "setenv.sh setting APP_NAME to: ${APP_NAME}"

    NGINX_HOME=/home/admin/cai

    # 显式export APP_NAME，以规避非K8S HIPPO调度时，无法将docker_file的env传递到进程启动参数的问题
    export APP_NAME=${APP_NAME}
    echo "APP_NAME:[${APP_NAME}]"

    export JAVA_HOME=/opt/taobao/java
    export PATH=${PATH}:${JAVA_HOME}/bin
    ulimit -c unlimited

    echo "INFO: OS max open files: "`ulimit -n`

    JAVA_VERSION=$(java_version)
    echo "INFO: java version: $JAVA_VERSION"

    # when stop spring boot process, will try to stop old tomcat process
    export CATALINA_HOME=/opt/taobao/tomcat
    export CATALINA_BASE=$APP_HOME/.default
    export CATALINA_PID=$CATALINA_BASE/catalina.pid

    # 禁用 glibc的禁用per thread arena，只用main arena
    export MALLOC_ARENA_MAX=1

    # time to wait tomcat to stop before killing it
    TOMCAT_STOP_WAIT_TIME=5
    TOMCAT_PORT=7001

    if [[ ! -f ${APP_HOME}/target/${APP_NAME}/bin/appctl.sh ]]; then
        # env for service(pandora boot)
        export LANG=zh_CN.UTF-8
        export JAVA_FILE_ENCODING=UTF-8
        export NLS_LANG=AMERICAN_AMERICA.ZHS16GBK
        export LD_LIBRARY_PATH=/opt/taobao/oracle/lib:/opt/taobao/lib:$LD_LIBRARY_PATH
        CPU_COUNT=$(available_cpu_count)
        export CPU_COUNT

        # Match HotSpot G1 ergonomics to the CPU quota visible to this container.
        if [ "$CPU_COUNT" -le 8 ]; then
          parallelGCThreads=$CPU_COUNT
        else
          parallelGCThreads=$((8 + (CPU_COUNT - 8) * 5 / 8))
        fi
        concGCThreads=$((parallelGCThreads / 4))
        if [ "$concGCThreads" -lt 1 ]; then
          concGCThreads=1
        fi

        export SERVICE_PID=$APP_HOME/.default/${APP_NAME}.pid
        export MIDDLEWARE_LOGS="${HOME}/logs"
        export MIDDLEWARE_SNAPSHOTS="${HOME}/snapshots"
        mkdir -p "$APP_HOME"/.default "$APP_HOME"/logs \
          "$MIDDLEWARE_LOGS" "$MIDDLEWARE_SNAPSHOTS" || exit 1

        if [ -z "$SERVICE_TMPDIR" ] ; then
            # Define the java.io.tmpdir to use for Service(pandora boot)
            SERVICE_TMPDIR="${APP_HOME}"/.default/temp
        fi

        SERVICE_OPTS="${SERVICE_OPTS} -server"

        memTotal=$(available_memory_mb)
        echo "INFO: available container memory: ${memTotal}M"
        # Keep enough native-memory headroom for direct buffers, metaspace,
        # code cache, thread stacks, and the container runtime.
        configure_default_jvm_memory "$memTotal"

        FLEXLB_HEAP_SIZE=${FLEXLB_JVM_HEAP_SIZE:-${MASTER_JVM_HEAP_SIZE}}
        SERVICE_JVM_XMS=${FLEXLB_JVM_XMS:-${MASTER_JVM_XMS:-${FLEXLB_HEAP_SIZE:-${DEFAULT_JVM_XMS}}}}
        SERVICE_JVM_XMX=${FLEXLB_JVM_XMX:-${MASTER_JVM_XMX:-${FLEXLB_HEAP_SIZE:-${DEFAULT_JVM_XMX}}}}
        validate_jvm_memory_budget "$memTotal" "$SERVICE_JVM_XMS" "$SERVICE_JVM_XMX" || exit 1
        echo "INFO: JVM heap config: -Xms${SERVICE_JVM_XMS} -Xmx${SERVICE_JVM_XMX}"
        SERVICE_OPTS="${SERVICE_OPTS} -Xms${SERVICE_JVM_XMS} -Xmx${SERVICE_JVM_XMX}"

        SERVICE_OPTS="${SERVICE_OPTS} -XX:MetaspaceSize=${maxMetaspace} -XX:MaxMetaspaceSize=${maxMetaspace}"
        SERVICE_OPTS="${SERVICE_OPTS} -XX:ReservedCodeCacheSize=${reservedCodeCache} -XX:MaxDirectMemorySize=${maxDirectMemory}"
        # 使用G1GC
        SERVICE_OPTS="${SERVICE_OPTS} -XX:+UseG1GC"
        SERVICE_OPTS="${SERVICE_OPTS} -XX:+UnlockExperimentalVMOptions"
        SERVICE_OPTS="${SERVICE_OPTS} -XX:MaxGCPauseMillis=150"
        SERVICE_OPTS="${SERVICE_OPTS} -XX:InitiatingHeapOccupancyPercent=40"
        SERVICE_OPTS="${SERVICE_OPTS} -XX:G1HeapRegionSize=32M"
        SERVICE_OPTS="${SERVICE_OPTS} -XX:G1NewSizePercent=20"
        SERVICE_OPTS="${SERVICE_OPTS} -XX:+ExplicitGCInvokesConcurrent"
        SERVICE_OPTS="${SERVICE_OPTS} -XX:SurvivorRatio=8"
        SERVICE_OPTS="${SERVICE_OPTS} -Dsun.rmi.dgc.server.gcInterval=2592000000 -Dsun.rmi.dgc.client.gcInterval=2592000000"
        SERVICE_OPTS="${SERVICE_OPTS} -XX:ParallelGCThreads=${parallelGCThreads} -XX:ConcGCThreads=${concGCThreads}"
        if [[ "$JAVA_VERSION" -lt 9 ]]; then
            SERVICE_OPTS="${SERVICE_OPTS} -Xloggc:${MIDDLEWARE_LOGS}/gc.log -XX:+PrintGCDetails -XX:+PrintGCDateStamps"
        else
            SERVICE_OPTS="${SERVICE_OPTS} -Xlog:gc*:${MIDDLEWARE_LOGS}/gc.log:time"
        fi
        SERVICE_OPTS="${SERVICE_OPTS} -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=${MIDDLEWARE_LOGS}/java.hprof"
        SERVICE_OPTS="${SERVICE_OPTS} -Djava.awt.headless=true"
        SERVICE_OPTS="${SERVICE_OPTS} -Dsun.net.client.defaultConnectTimeout=10000"
        SERVICE_OPTS="${SERVICE_OPTS} -Dsun.net.client.defaultReadTimeout=30000"
        SERVICE_OPTS="${SERVICE_OPTS} -DJM.LOG.PATH=${MIDDLEWARE_LOGS}"
        SERVICE_OPTS="${SERVICE_OPTS} -DJM.SNAPSHOT.PATH=${MIDDLEWARE_SNAPSHOTS}"
        SERVICE_OPTS="${SERVICE_OPTS} -Dfile.encoding=${JAVA_FILE_ENCODING}"
        SERVICE_OPTS="${SERVICE_OPTS} -Dhsf.publish.delayed=true"
        SERVICE_OPTS="${SERVICE_OPTS} -Dproject.name=${APP_NAME}"
        SERVICE_OPTS="${SERVICE_OPTS} -Dlog4j.defaultInitOverride=true"
        SERVICE_OPTS="${SERVICE_OPTS} -Dserver.port=${TOMCAT_PORT} -Dmanagement.port=7002 -Dmanagement.server.port=7002"

        # JDK17 JMPS opts
        if [[ "$JAVA_VERSION" -ge 17 ]]; then
          SERVICE_OPTS="${SERVICE_OPTS} --add-modules ALL-SYSTEM"
          SERVICE_OPTS="${SERVICE_OPTS} --add-opens java.base/java.lang=ALL-UNNAMED"
          SERVICE_OPTS="${SERVICE_OPTS} --add-opens java.base/java.lang.invoke=ALL-UNNAMED"
          SERVICE_OPTS="${SERVICE_OPTS} --add-opens java.base/java.util=ALL-UNNAMED"
          SERVICE_OPTS="${SERVICE_OPTS} --add-opens java.base/java.util.concurrent=ALL-UNNAMED"
          SERVICE_OPTS="${SERVICE_OPTS} --add-opens=java.base/jdk.internal.misc=ALL-UNNAMED"
          SERVICE_OPTS="${SERVICE_OPTS} --add-opens java.base/java.nio=ALL-UNNAMED"
          SERVICE_OPTS="${SERVICE_OPTS} --add-opens java.base/sun.nio.ch=ALL-UNNAMED"
          SERVICE_OPTS="${SERVICE_OPTS} --add-opens java.instrument/sun.instrument=ALL-UNNAMED"
        fi

        # debug opts

        # jpda options
        test -z "$JPDA_ENABLE" && JPDA_ENABLE=0
        test -z "$JPDA_ADDRESS" && export JPDA_ADDRESS=8000
        test -z "$JPDA_SUSPEND" && export JPDA_SUSPEND=n

        if [ "$JPDA_ENABLE" -eq 1 ]; then
            if [ -z "$JPDA_TRANSPORT" ]; then
                JPDA_TRANSPORT="dt_socket"
            fi
            if [ -z "$JPDA_ADDRESS" ]; then
                JPDA_ADDRESS="8000"
            fi
            if [ -z "$JPDA_SUSPEND" ]; then
                JPDA_SUSPEND="n"
            fi
            if [ -z "$JPDA_OPTS" ]; then
                if [[ "$JAVA_VERSION" -lt 9 ]]; then
                    JPDA_OPTS="-agentlib:jdwp=transport=$JPDA_TRANSPORT,address=$JPDA_ADDRESS,server=y,suspend=$JPDA_SUSPEND"
                else
                    JPDA_OPTS="-agentlib:jdwp=transport=$JPDA_TRANSPORT,address=*:$JPDA_ADDRESS,server=y,suspend=$JPDA_SUSPEND"
                fi
            fi
            SERVICE_OPTS="$SERVICE_OPTS $JPDA_OPTS"
        fi

        export SERVICE_OPTS

        if [ -z "$NGINX_HOME" ]; then
            NGINX_HOME=/home/admin/cai
        fi

        # if set to "1", skip start nginx.
        test -z "$NGINX_SKIP" && NGINX_SKIP=0
        # set port for checking status.taobao file. Comment it if no need.
        STATUS_PORT=80
        # time to wait for /status.taobao is ready
        STATUS_TAOBAO_WAIT_TIME=3
        STATUSROOT_HOME="${APP_HOME}/target/${APP_NAME}/META-INF/resources"
        # make sure the directory exist, before tomcat start
        mkdir -p $STATUSROOT_HOME
        NGINXCTL=$NGINX_HOME/bin/nginxctl
    else
        # compatible with the existing jar application
        export LANG=zh_CN.UTF-8
    fi

fi
