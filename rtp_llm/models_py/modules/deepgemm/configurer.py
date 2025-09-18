try:
    from deep_gemm import fp8_gemm_nt

    # They have not given a name to this breaking change
    DEEPGEMM_BLACKWELL = True
except ImportError:
    DEEPGEMM_BLACKWELL = False

DEEPGEMM_SCALE_UE8M0 = False