#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iostream>

// token_num：每条约束路径的语义ID长度（token数量）
static const int token_num = 3;

// sids<N>：N个token ID组成的定长数组，表示一条约束路径
template<int N>
struct sids {
    int rq_id[N];

    inline bool operator<(const sids<N>& other) const {
        for (int i = 0; i < N - 1; ++i) {
            if (rq_id[i] != other.rq_id[i]) {
                return rq_id[i] < other.rq_id[i];
            }
        }
        return rq_id[N - 1] < other.rq_id[N - 1];
    }
};

// CSRIndex<N>：对长度为N的约束路径集合构建的CSR格式前缀树索引。
// 每次请求在CPU上构建一次，然后以flat int32数组形式上传到GPU。
//
// 数据布局（对应 csr_utils.py 中的 build_static_index）：
//   packed_csr_tokens/states : [边数 + vocab_size]  (token, next_state) 转移对
//   indptr                   : [状态数 + 2]  行指针数组
//   start_mask               : [vocab_size]  第0层（根节点出发）的合法token掩码
//   layer_max_branches       : [N]  每层前缀树的最大分支数
template<int N>
struct CSRIndex {
    std::vector<int>  indptr;             // 行指针数组，大小 = 状态数 + 2
    std::vector<int>  packed_csr_tokens;  // 转移表的token列
    std::vector<int>  packed_csr_states;  // 转移表的next_state列
    std::vector<bool> start_mask;         // 第0层合法token掩码，大小 = vocab_size
    std::vector<int>  layer_max_branches; // 每层最大分支数，大小 = N

    int num_states = 0;
    int vocab_size = 0;

    CSRIndex() = default;
};

// ---------------------------------------------------------------------------
// build_csr_from_fresh_data<N>
//
// 从已排序的 sids<N> 数组构建 CSRIndex<N>。
// 对应 csr_utils.py 中的 build_static_index（仅 CSR 路径，无 dense head）。
//
// 状态ID分配规则：
//   state 0      ：根节点（隐式，CSR中无对应行，只有末尾填充行）
//   state 1..V   ：第0层前缀树节点（第一个token t -> state t+1）
//   state V+1..  ：更深层节点，按 DFS 顺序递增分配
//
// packed_csr 末尾追加 vocab_size 个填充行（token=V, next_state=0），
// 保证越界索引安全（与 Python 版本一致）。
// ---------------------------------------------------------------------------
template<int N>
bool build_csr_from_fresh_data(const std::vector<sids<N>>& fresh_data,
                                CSRIndex<N>&                index,
                                int                         vocab_size) {
    if (fresh_data.empty()) {
        return false;
    }

    const int data_size = static_cast<int>(fresh_data.size());

    index.vocab_size = vocab_size;

    // --- 1. 构建第0层合法token掩码（start_mask） ---
    index.start_mask.assign(vocab_size, false);
    for (const auto& s : fresh_data) {
        if (s.rq_id[0] >= 0 && s.rq_id[0] < vocab_size) {
            index.start_mask[s.rq_id[0]] = true;
        }
    }

    // --- 2. is_new[i][j]：第i行是否在第j层引入了新的前缀树节点 ---
    // 用 vector<vector<bool>> 代替 C99 可变长数组（VLA）
    std::vector<std::vector<bool>> is_new(data_size, std::vector<bool>(N, false));
    for (int j = 0; j < N; ++j) {
        is_new[0][j] = true;
    }
    for (int i = 1; i < data_size; ++i) {
        is_new[i][0] = (fresh_data[i].rq_id[0] != fresh_data[i - 1].rq_id[0]);
        for (int j = 1; j < N; ++j) {
            if (!is_new[i][j - 1] && fresh_data[i].rq_id[j] == fresh_data[i - 1].rq_id[j]) {
                is_new[i][j] = false;
            } else {
                is_new[i][j] = true;
            }
        }
    }

    // --- 3. 状态ID分配 ---
    // 第0层节点：state_id = token_id + 1（占用 state 1..vocab_size）
    // 更深层节点：从 vocab_size+1 开始顺序分配
    std::vector<std::vector<int>> state_ids(data_size, std::vector<int>(N - 1, 0));
    for (int i = 0; i < data_size; ++i) {
        state_ids[i][0] = fresh_data[i].rq_id[0] + 1;
    }

    int num_states = vocab_size;  // 下一个可用状态ID - 1
    for (int depth = 1; depth < N - 1; ++depth) {
        for (int i = 0; i < data_size; ++i) {
            if (i == 0 || is_new[i][depth]) {
                ++num_states;
                state_ids[i][depth] = num_states;
            } else {
                state_ids[i][depth] = state_ids[i - 1][depth];
            }
        }
    }
    ++num_states;  // 最终状态总数 = 已分配的最大状态ID
    index.num_states = num_states;

    // --- 4. 收集前缀树的所有边 ---
    std::vector<int> parent_ids_vec;
    std::vector<int> token_ids_vec;
    std::vector<int> child_ids_vec;

    // layer_max_branches[0] = 第0层不同起始token的个数
    int depth0_count = 0;
    for (int v = 0; v < vocab_size; ++v) {
        if (index.start_mask[v]) ++depth0_count;
    }
    index.layer_max_branches.clear();
    index.layer_max_branches.push_back(depth0_count);

    for (int depth = 1; depth < N; ++depth) {
        int start_pos = static_cast<int>(parent_ids_vec.size());
        for (int i = 0; i < data_size; ++i) {
            if (is_new[i][depth]) {
                int parent = state_ids[i][depth - 1];
                int token  = fresh_data[i].rq_id[depth];
                int child  = (depth < N - 1) ? state_ids[i][depth] : 0;  // 0 表示终止状态
                parent_ids_vec.push_back(parent);
                token_ids_vec.push_back(token);
                child_ids_vec.push_back(child);
            }
        }
        int end_pos = static_cast<int>(parent_ids_vec.size());

        // 计算当前层的最大分支数
        int max_branches = 0;
        int cur_branches = 0;
        for (int i = start_pos; i < end_pos; ++i) {
            if (i == start_pos || parent_ids_vec[i] != parent_ids_vec[i - 1]) {
                cur_branches = 1;
            } else {
                ++cur_branches;
            }
            if (cur_branches > max_branches) max_branches = cur_branches;
        }
        index.layer_max_branches.push_back(max_branches);
    }

    // --- 5. 构建 indptr ---
    // indptr 大小 = num_states + 2：
    //   第 0..num_states-1 行对应 state 0..num_states-1
    //   末尾多一行填充（state num_states），用于越界安全索引
    const int indptr_size = num_states + 2;
    index.indptr.assign(indptr_size, 0);
    for (int pid : parent_ids_vec) {
        index.indptr[pid + 1]++;
    }
    for (int i = 1; i < indptr_size - 1; ++i) {
        index.indptr[i] += index.indptr[i - 1];
    }
    // 末尾填充行：占用 vocab_size 个槽位，用于越界安全
    index.indptr[num_states + 1] = index.indptr[num_states] + vocab_size;

    // --- 6. 构建 packed_csr（转移表） ---
    const int num_transitions = static_cast<int>(token_ids_vec.size());
    const int packed_size     = num_transitions + vocab_size;
    index.packed_csr_tokens.resize(packed_size);
    index.packed_csr_states.resize(packed_size);

    for (int i = 0; i < num_transitions; ++i) {
        index.packed_csr_tokens[i] = token_ids_vec[i];
        index.packed_csr_states[i] = child_ids_vec[i];
    }
    // 填充行：token = vocab_size（哨兵值），state = 0
    for (int i = 0; i < vocab_size; ++i) {
        index.packed_csr_tokens[num_transitions + i] = vocab_size;
        index.packed_csr_states[num_transitions + i] = 0;
    }

    return true;
}

// ---------------------------------------------------------------------------
// split_strings<N>
// 将 "t0_t1_t2" 格式的字符串解析为 sids<N>。
// ---------------------------------------------------------------------------
template<int N>
std::vector<sids<N>> split_strings(const std::vector<std::string>& ele_rq_ids) {
    std::vector<sids<N>> result;
    result.reserve(ele_rq_ids.size());

    for (const std::string& str : ele_rq_ids) {
        sids<N> sid{};
        size_t  start = 0;
        int     count = 0;

        while (count < N) {
            size_t end = str.find('_', start);
            std::string token = (end == std::string::npos)
                                    ? str.substr(start)
                                    : str.substr(start, end - start);
            if (!token.empty() && count < N) {
                sid.rq_id[count] = std::stoi(token);
            }
            ++count;
            if (end == std::string::npos) break;
            start = end + 1;
        }
        result.push_back(sid);
    }
    return result;
}

// ---------------------------------------------------------------------------
// parseJsonArray<N>
// 从 JSON 文件中解析 [[t0,t1,...], ...] 格式的数组，返回 sids<N> 列表。
// ---------------------------------------------------------------------------
template<int N>
std::vector<sids<N>> parseJsonArray(const std::string& filename) {
    std::vector<sids<N>> result;
    std::ifstream        file(filename);
    if (!file.is_open()) {
        std::cerr << "错误：无法打开文件 " << filename << std::endl;
        return result;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();
    content.erase(std::remove_if(content.begin(), content.end(), ::isspace), content.end());

    size_t outer_start = content.find('[');
    size_t outer_end   = content.rfind(']');
    if (outer_start == std::string::npos || outer_end == std::string::npos) {
        std::cerr << "错误：JSON格式非法" << std::endl;
        return result;
    }

    size_t pos = outer_start + 1;
    while (pos < outer_end) {
        size_t array_start = content.find('[', pos);
        if (array_start == std::string::npos || array_start >= outer_end) break;
        size_t array_end = content.find(']', array_start);
        if (array_end == std::string::npos) break;

        std::string        inner = content.substr(array_start + 1, array_end - array_start - 1);
        std::stringstream  ss(inner);
        std::string        token;
        std::vector<int>   row;
        while (std::getline(ss, token, ',')) {
            if (!token.empty()) row.push_back(std::stoi(token));
        }

        if (static_cast<int>(row.size()) == N) {
            sids<N> s{};
            for (int i = 0; i < N; ++i) s.rq_id[i] = row[i];
            result.push_back(s);
        }
        pos = array_end + 1;
    }
    return result;
}
