"""生成式推荐场景：从 prompt 中解析「已推荐曝光的商品序列」并构造 banned_combo_token_ids。

典型 prompt 片段（格式严格）：
    已推荐曝光的商品序列和位置:pos0:C1071C2997C4163,pos1:C741C3248C4162,...

本模块只在 generate_config.auto_parse_banned_combo=True 且 combo_token_size>0
时被 pipeline 调用；对非推荐场景零侵入。
"""

import logging
import re
from typing import Any, List, Optional

# 匹配 posN:C123C456C789 这种片段。商品语义 ID 段数由 combo_token_size 决定。
_POS_ITEM_PREFIX_RE = re.compile(r"pos\d+:((?:C\d+)+)")
_SEMANTIC_ID_RE = re.compile(r"C\d+")


def _extract_exposed_items(prompt: str, combo_token_size: int) -> List[List[str]]:
    """从 prompt 中抽出所有已曝光商品，按 combo_token_size 切分为语义 ID 组合。

    返回每个商品对应的 combo_token_size 个语义 ID 字符串（例如 ["C1071", "C2997", "C4163"]）。
    若商品段包含的 CXXX 数量与 combo_token_size 不符，则跳过该商品。
    """
    exposed_items: List[List[str]] = []
    for match in _POS_ITEM_PREFIX_RE.finditer(prompt):
        combo_str = match.group(1)
        semantic_ids = _SEMANTIC_ID_RE.findall(combo_str)
        if len(semantic_ids) != combo_token_size:
            logging.warning(
                "recommendation_parser: skip item '%s' since its semantic id count %d != combo_token_size %d",
                combo_str,
                len(semantic_ids),
                combo_token_size,
            )
            continue
        exposed_items.append(semantic_ids)
    return exposed_items


def _encode_semantic_id(tokenizer: Any, semantic_id: str) -> Optional[int]:
    """把单个语义 ID 字符串（如 'C1071'）编码成一个 token id。

    语义 ID 在训练时被扩展为词表中的独立 token，因此优先用 convert_tokens_to_ids 精确查询；
    若 tokenizer 不支持或返回 unk，则回退到 encode（add_special_tokens=False）并要求长度为 1。
    """
    # 路径 1：convert_tokens_to_ids
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(semantic_id)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if isinstance(token_id, int) and token_id >= 0 and token_id != unk_id:
                return token_id
        except Exception:
            pass

    # 路径 2：encode 回退
    try:
        ids = tokenizer.encode(semantic_id, add_special_tokens=False)
        if isinstance(ids, list) and len(ids) == 1:
            return int(ids[0])
    except Exception:
        pass

    return None


def parse_and_fill_banned_combo(
    prompt: str,
    generate_config: Any,
    tokenizer: Any,
) -> int:
    """解析 prompt 中的已曝光商品并合并到 generate_config.banned_combo_token_ids。

    只有 generate_config.auto_parse_banned_combo=True 且 combo_token_size>0 时才会执行。
    已有的 banned_combo_token_ids 会被保留，通过 set 去重合并新解析出的组合。
    返回本次追加的商品数量（用于日志/监控）。
    """
    combo_token_size = getattr(generate_config, "combo_token_size", 0)
    auto_parse = getattr(generate_config, "auto_parse_banned_combo", False)
    if not auto_parse or combo_token_size <= 0 or tokenizer is None or not prompt:
        return 0

    exposed_items = _extract_exposed_items(prompt, combo_token_size)
    if not exposed_items:
        return 0

    # 已有组合先放入 set 做去重基准
    existing = {tuple(combo) for combo in generate_config.banned_combo_token_ids}
    appended = 0
    for semantic_ids in exposed_items:
        token_ids: List[int] = []
        invalid = False
        for sid in semantic_ids:
            tid = _encode_semantic_id(tokenizer, sid)
            if tid is None:
                logging.warning(
                    "recommendation_parser: failed to encode semantic id '%s', skip item %s",
                    sid,
                    semantic_ids,
                )
                invalid = True
                break
            token_ids.append(tid)
        if invalid or len(token_ids) != combo_token_size:
            continue
        key = tuple(token_ids)
        if key in existing:
            continue
        existing.add(key)
        generate_config.banned_combo_token_ids.append(token_ids)
        appended += 1

    if appended > 0:
        logging.info(
            "recommendation_parser: parsed %d exposed items from prompt, total banned_combo=%d",
            appended,
            len(generate_config.banned_combo_token_ids),
        )
    return appended
