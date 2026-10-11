"""Java client input validation, runtime-owned settings and launch isolation."""

LOAD_CLIENT_ENV_VARS = (
    'LIVE_CLIENT_EVENTS',
    'TRACE_FILE',
    'TARGET_ADDR',
    'GRPC_TARGET',
    'GRPC_TARGETS',
    'MASTER_DISCOVERY_FILE',
    'DURATION_S',
    'MAX_CONCURRENCY',
    'REPLAY_SPEED',
    'LOAD_CLIENT_WORKERS',
    'OUTPUT_DIR',
    'NUM_SHARDS',
    'SHARD_INDEX',
    'LIMIT',
    'TIMEOUT_MS',
    'SLA_TTFT_MS',
    'FETCH_OUTPUT_STREAM',
    'LOOP',
    'N_CHANNELS',
    'EVENT_LOOP_THREADS',
    'START_AT_EPOCH_MS',
    'RESPONSE_TIMEOUT',
    'SKIP_SERVER_LATENCY',
    'MODEL',
    'API_KEY',
    'GRADIENT',
    'GRADIENT_START_SPEED',
    'GRADIENT_MAX_SPEED',
    'MAX_INPUT_LEN',
    'MAX_OUTPUT_LEN',
    'PUSHGATEWAY_URL',
    'ENABLE_FALLBACK',
    'ENDPOINTS_FILE',
    'DRY_RUN',
    'SEND_MODE',
    'SEND_MODE_QPS',
    'REPLAY_UNIQUE_PREFIX',
    'PRIORITY',
    'FORCE_PRIORITY',
    'RAMP_UP_SECONDS',
    'FLOW_CONTROL_DIR',
    'FLOW_RUN_ID',
    'FLOW_GROUP_ID',
    'FLOW_PHASE_ID',
    'MAX_LAPS',
    'LAP_IDENTITY',
    'LAP_RETAIN_PROBABILITY',
    'PLAYBACK_SEED',
    'BURST_FACTOR',
    'BURST_PERIOD_SECONDS',
    'BURST_DUTY',
    'DIURNAL_AMPLITUDE',
    'DIURNAL_PERIOD_SECONDS',
    'COLLECTION_PROFILE',
    'CLIENT_MONITORING',
    'RATE_CURVE',
    'ARRIVAL_PROCESS',
    'LAP_RETAIN_SCHEDULE',
)


FRAMEWORK_CLIENT_ENV_VARS = frozenset({
    'FLOW_CONTROL_DIR', 'FLOW_RUN_ID', 'FLOW_GROUP_ID', 'FLOW_PHASE_ID',
    'TRACE_FILE', 'OUTPUT_DIR', 'GRPC_TARGET', 'GRPC_TARGETS',
    'MASTER_DISCOVERY_FILE', 'LIVE_CLIENT_EVENTS', 'COLLECTION_PROFILE',
    'CLIENT_MONITORING', 'SKIP_SERVER_LATENCY',
})


def validate_environment(environment):
    """Validate names and scalar values even at the final subprocess boundary."""
    import math

    if not isinstance(environment, dict) or set(environment) - set(LOAD_CLIENT_ENV_VARS):
        raise ValueError('unknown Java client configuration names')
    if any(type(value) not in (str, int, float)
           or (type(value) is float and not math.isfinite(value))
           for value in environment.values()):
        raise ValueError('Java client environment values must be finite scalars')
    return {key: str(value) for key, value in environment.items()}


def validate_flow_environment(environment):
    import math

    env = validate_environment(environment)
    if env.get('REPLAY_UNIQUE_PREFIX') != 'false' or env.get('FETCH_OUTPUT_STREAM') != 'true':
        raise ValueError('scenario flow requires faithful prefix replay and response consumption')
    if int(env.get('DURATION_S', 0)) <= 0 or int(env.get('MAX_CONCURRENCY', 0)) <= 0:
        raise ValueError('scenario flow requires bounded duration and concurrency')
    speed = float(env.get('REPLAY_SPEED', 1))
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError('replay speed must be finite and positive')
    mode = env.get('SEND_MODE', 'replay')
    qps = float(env.get('SEND_MODE_QPS', 0))
    if mode not in {'uniform', 'replay'} or not math.isfinite(qps) or (mode == 'uniform' and qps <= 0):
        raise ValueError('invalid Java client pacing')
    return env


def client_environment(client):
    """Normalize authored settings; framework-owned fields cannot be supplied."""
    from traffic.playback_config import normalize

    if not isinstance(client, dict):
        raise ValueError('Java client configuration must be a mapping')
    if set(client) & FRAMEWORK_CLIENT_ENV_VARS:
        raise ValueError('client endpoint, runtime identities and collection policy are resolved by framework')
    env, playback = normalize(client)
    return validate_flow_environment(env), playback


def collection_environment(profile, *, monitoring=False, live_events=None):
    if profile not in {'aggregate', 'request', 'diagnostic'}:
        raise ValueError('unknown collection profile')
    return dict(
        LIVE_CLIENT_EVENTS=str(profile != 'aggregate' if live_events is None else live_events).lower(),
        COLLECTION_PROFILE=profile,
        CLIENT_MONITORING=str(monitoring).lower(),
        SKIP_SERVER_LATENCY=str(profile != 'diagnostic').lower(),
    )


def bind_environment(environment, framework):
    """Join explicit inputs and runtime-owned values without silent overrides."""
    if set(environment) & set(framework):
        raise ValueError('framework Java client configuration conflicts with explicit inputs')
    return validate_flow_environment({**environment, **framework})
