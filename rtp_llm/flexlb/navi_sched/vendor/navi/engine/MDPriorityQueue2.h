#pragma once
#include <atomic>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>

// #define MD_PRIORITY_QUEUE_DEBUG
// #define MD_PRIORITY_QUEUE_NO_RECYCLE

#ifdef MD_PRIORITY_QUEUE_DEBUG
#include <vector>
#include <algorithm>
#endif

namespace md2 {

#ifdef __aarch64__
#define cpu_relax() __asm__ volatile("yield\n": : :"memory")
#else
#define cpu_relax() __asm__ volatile("rep; nop\n": : :"memory")
#endif

struct EpochCounter {
    std::atomic<uint32_t> count {0};
    char pad[60];
};

constexpr int EPOCH_SLOT_COUNT = 43;
extern std::atomic<int32_t> THREAD_ID_COUNTER;
thread_local extern int THREAD_SLOT_ID;

class EpochManager {
public:
    int enter() {
        if (THREAD_SLOT_ID < 0) {
            auto id = THREAD_ID_COUNTER++;
            if (id >= EPOCH_SLOT_COUNT) {
                id = (id % EPOCH_SLOT_COUNT);
            }
            THREAD_SLOT_ID = id;
        }
        auto index = THREAD_SLOT_ID;
        _slots[index].count++;
        return index;
    }
    void exit(int index) {
        _slots[index].count--;
    }
    void begin() {
        for (int i = 0; i < EPOCH_SLOT_COUNT; i++) {
            auto value = (0 == _slots[i].count.load());
            _snapshot[i] = value;
        }
    }
    bool check() {
        bool ret = true;
        for (int i = 0; i < EPOCH_SLOT_COUNT; i++) {
            if (!_snapshot[i]) {
                auto value = (0 == _slots[i].count.load());
                _snapshot[i] = value;
                ret = ret && value;
            }
        }
        return ret;
    }
private:
    EpochCounter _slots[EPOCH_SLOT_COUNT] __attribute__((aligned(64)));
    bool _snapshot[EPOCH_SLOT_COUNT];
};

constexpr int D = 4;

struct Node;

struct AdoptDesc {
    Node *curr;
    int dp;
    int dc;
#ifdef MD_PRIORITY_QUEUE_DEBUG
    std::atomic<uint64_t> adoptCount;
#endif
};

struct HeadNode;

struct Node {
    std::atomic<uint64_t> child[D];
    std::atomic<AdoptDesc *> adesc {NULL};
    std::atomic<uint64_t> val{0};
    size_t key;
    int k[D];
    int d = 0;
    AdoptDesc *finishAdesc = NULL;
#ifdef MD_PRIORITY_QUEUE_DEBUG
    uint64_t id = 0;
    uint64_t deleteId = 0;
    HeadNode *head = NULL;
    Node *insertPred = NULL;
    Node *adescNode = NULL;
#endif
    void setFromNode(Node *other) {
        val.store(other->val.load());
        key = other->key;
        for (int i = 0; i < D; i++) {
            k[i] = other->k[i];
            child[i].store(other->child[i].load());
        }
    }
};

struct HeadNode : public Node {
};

#define MD_SetMark(p, m) ((p) | (m))
#define MD_ClearMark(p, m) ((p) & ~((uint64_t)(m)))
#define MD_IsMarked(p, m) ((p) & (uint64_t)(m))
#define MD_F_ADP 0x1U
#define MD_F_PRG 0x2U
#define MD_F_DEL 0x1U
#define MD_F_ALL 0x3U
#define MD_Clear(p) MD_ClearMark(p, MD_F_ALL)

class Handler {
public:
    Node *newNode() {
        return new Node();
    }
    AdoptDesc *newDesc() {
        return new AdoptDesc();
    }
    HeadNode *newHeadNode() {
        return new HeadNode();
    }
};

class PriorityQueue {
public:
    PriorityQueue(size_t n, int r);
    ~PriorityQueue();
public:
    bool insert(size_t key, uint64_t val, Node *&retNode);
    uint64_t deleteMin(uint64_t &key, Node *&retNode);
    uint64_t size() {
        return _size.load();
    }
    uint64_t empty() {
        return (0 == _size.load());
    }
    void showStats();
private:
    inline void setCoords(Node *n, size_t key) {
        size_t quotient = key;
        int *k = n->k;
        for (int i = D - 1; i >= 0; i--) {
            k[i] = quotient % basis;
            quotient = quotient / basis;
        }
    }
    void purge(HeadNode *hn, Node *prg, Handler *h);
    void finishDeleting();
    inline void finishInserting(Node *n, int dp, int dc) {
        if (n == NULL) {
            return;
        }
        AdoptDesc *ad = n->adesc.load();
        if (ad == NULL || dc < ad->dp || dp > ad->dc) {
            return;
        }
        uint64_t child;
        Node *curr = ad->curr;
        for (int i = ad->dp; i < ad->dc; i++) {
            child = MD_ClearMark(curr->child[i].fetch_or(MD_F_ADP), MD_F_ADP);
            uint64_t temp = 0;
#ifdef MD_PRIORITY_QUEUE_DEBUG
            if (n->child[i].compare_exchange_strong(temp, child)) {
                ad->adoptCount++;
            } else if (child != 0 && child != MD_F_PRG && n->child[i].load() == MD_F_PRG) {
                std::cout << "finish insert failed: " << n << std::endl;
            }
#else
            n->child[i].compare_exchange_strong(temp, child);
#endif
        }
        n->adesc = NULL;
        n->finishAdesc = ad;
#ifdef MD_PRIORITY_QUEUE_DEBUG
#endif
    }
    inline void LocatePlace(Handler *h, int &dp, int &dc, Node *&pred, Node *&curr, Node *n) {
        while (dc < D) {
            while (curr != NULL && n->k[dc] > curr->k[dc]) {
                pred = curr;
                dp = dc;
                finishInserting(curr, dc, dc);
                curr = (Node *) (MD_Clear(curr->child[dc].load()));
            }
            if (curr == NULL || n->k[dc] < curr->k[dc]) {
                break;
            }
            dc++;
        }
    }
    inline void FillNewNode(Handler *h, int dp, int dc, Node *n, Node *curr) {
        if (dp < dc) {
            AdoptDesc *desc;
            if (n->adesc.load()) {
                desc = n->adesc.load();
            } else {
                desc = h->newDesc();
            }
            desc->curr = curr;
            desc->dc = dc;
            desc->dp = dp;
            n->adesc.store(desc);
        } else {
            if (n->adesc.load() != 0) {
                delete n->adesc.load();
            }
            n->adesc.store(NULL);
        }
        for (int i = 0; i < dp; i++) {
            n->child[i] = MD_F_ADP;
        }
        for (int i = dp; i < D; i++) {
            n->child[i] = 0;
        }
        n->child[dc] = (uint64_t)curr;
    }
#ifdef MD_PRIORITY_QUEUE_DEBUG
    inline void fillStack(Stack *s) {
        s->pid = (long)syscall(SYS_gettid);
        struct timeval tval;
        gettimeofday(&tval, NULL);
        s->time = (tval.tv_sec * 1000000LL + tval.tv_usec);
    }
#endif
    void checkInsertHead();
    HeadNode *reclaimOne(HeadNode *headNode);
    void reclaim();
    void showHeadStats();
private:
    size_t N;
    int R;
    size_t basis;
    std::atomic<bool> notPurging {true};
    size_t nMarkedNodes = 0;
    std::atomic<uint64_t> _size {0};
    std::atomic<uint64_t> head{0};
    EpochManager insertEpochManager;
    EpochManager deleteEpochManager;
    uint64_t checkingMinInsertHead = 0;
    uint64_t minInsertHead = 0;
    Handler handler;
#ifdef MD_PRIORITY_QUEUE_DEBUG
    std::atomic<uint64_t> nodeId {0};
#endif
};

}
