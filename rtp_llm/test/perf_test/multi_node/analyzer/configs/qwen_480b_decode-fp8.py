from module_base import ModuleConf, TestSetUp, calc_gemm_mem_io, calc_gemm_num_flop

hidden_size = 6144
qkv_size = 14336
o_size = 12288

moe_inter_size = 2560
num_experts = 160
num_experts_per_tok = 8

head_dim = 128
num_q_heads = 96
num_kv_heads = 8
q_size = num_q_heads * head_dim
kv_size = num_kv_heads * head_dim


module_config = {
    "qkv_gemm": ModuleConf(
        name_pattern="14336u, 6144u",
        num_flop_calc_func=lambda x: calc_gemm_num_flop(
            qkv_size, hidden_size, x.batch_size
        ),
        mem_io_calc_func=lambda x: calc_gemm_mem_io(
            qkv_size, hidden_size, x.batch_size, 1, 2
        ),
    ),
    "fmha": ModuleConf(
        name_pattern="fmha",
        num_flop_calc_func=lambda x: 2
        * x.batch_size
        * x.seq_len
        * head_dim
        * (num_q_heads + num_q_heads),
        mem_io_calc_func=lambda x: 2
        * x.batch_size
        * (x.seq_len * 2 * num_kv_heads + 2 * num_q_heads)
        * head_dim,
    ),
    "o_gemm": ModuleConf(
        name_pattern="6144u, 12288u",
        num_flop_calc_func=lambda x: calc_gemm_num_flop(
            hidden_size, o_size, x.batch_size
        ),
        mem_io_calc_func=lambda x: calc_gemm_mem_io(
            hidden_size, o_size, x.batch_size, 1, 2
        ),
    ),
    "dispatch": ModuleConf(
        name_pattern="dispatch",
        num_flop_calc_func=lambda x: 0,
        mem_io_calc_func=lambda x: hidden_size
        * int(num_experts_per_tok * x.batch_size),
    ),
    "moe_up": ModuleConf(
        name_pattern="5120u, 6144u,",
        num_flop_calc_func=lambda x: calc_gemm_num_flop(
            5120, 6144, int(x.batch_size * num_experts_per_tok)
        ),
        mem_io_calc_func=lambda x: int(num_experts / x.dp_size) * (5120 * 6144)
        + int(x.batch_size * num_experts_per_tok) * (5120 + 6144),
    ),
    "moe_down": ModuleConf(
        name_pattern="6144u, 2560u,",
        num_flop_calc_func=lambda x: calc_gemm_num_flop(
            6144, 2560, int(x.batch_size * num_experts_per_tok)
        ),
        mem_io_calc_func=lambda x: int(num_experts / x.dp_size) * (6144 * 2560)
        + int(x.batch_size * num_experts_per_tok) * (6144 + 2560),
    ),
    "combine": ModuleConf(
        name_pattern="combine<",
        num_flop_calc_func=lambda x: 0,
        mem_io_calc_func=lambda x: 2
        * hidden_size
        * int(num_experts_per_tok * x.batch_size),
    ),
}
