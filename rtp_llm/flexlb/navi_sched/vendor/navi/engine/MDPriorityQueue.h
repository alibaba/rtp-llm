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

namespace md {

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

constexpr int D = 7;

struct Node;

struct AdoptDesc {
    Node *curr;
    int dp;
    int dc;
#ifdef MD_PRIORITY_QUEUE_DEBUG
    std::atomic<uint64_t> adoptCount;
#endif
};

struct Stack;
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
    Stack *insertStack = NULL;
    Stack *deleteStack = NULL;
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
    uint64_t ver;
    std::atomic<Stack *> stacks{NULL};
#ifdef MD_PRIORITY_QUEUE_DEBUG
    std::atomic<size_t> nodeCount{0};
#else
    size_t nodeCount = 0;
#endif
#ifdef MD_PRIORITY_QUEUE_DEBUG
    HeadNode *prev = NULL;
#endif
};

struct Stack {
    Node *node[D];
    HeadNode *head = NULL;
    HeadNode *insertHead = NULL;
    Stack *parent = NULL;
    Stack *list = NULL;
#ifdef MD_PRIORITY_QUEUE_DEBUG
    int pid;
    size_t time;
    int type;
    size_t key;
    int rewind_type;
    Stack *prev = NULL;
    Stack *deletePrev = NULL;
    bool replaceOk = false;
    uint64_t followPurgeCount = 0;
    Node *lastPurgeNode = NULL;
    bool changeOld = false;
    Stack *ss = NULL;
    uint64_t seekCount = 0;
    uint64_t nodeId = 0;
    uint64_t endNodeId = 0;
    bool isDelete = false;
    bool isHead = false;
    uint64_t failedCount = 0;
    Stack *sNow = NULL;
    Stack *sOld = NULL;
#endif
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
    Stack *newStack() {
        auto s = new Stack();
#ifdef MD_PRIORITY_QUEUE_DEBUG
        s->type = 0;
        s->key = 0;
        s->rewind_type = 0;
#endif
        return s;
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
    uint64_t deleteMin(uint64_t &key, Node *&retNode) {
#ifdef MD_PRIORITY_QUEUE_DEBUG
        uint64_t count = 0;
#endif
        while (true) {
            uint64_t value = 0;
            if (deleteMinImpl(key, retNode, value)) {
                return value;
            }
#ifdef MD_PRIORITY_QUEUE_DEBUG
            count++;
            // if (count > 5) {
            //     std::cout << "count: " << count << std::endl;
            // }
#endif
        }
    }
    uint64_t size() {
        return _size.load();
    }
    uint64_t empty() {
        return (0 == _size.load());
    }
    void showStats();
private:
    bool deleteMinImpl(uint64_t &key, Node *&retNode, uint64_t &value);
    inline void setCoords(Node *n, size_t key) {
        size_t quotient = key;
        int *k = n->k;
        for (int i = D - 1; i >= 0; i--) {
            k[i] = quotient % basis;
            quotient = quotient / basis;
        }
    }
    void purge(HeadNode *hn, Node *prg, Handler *h);
    inline void finishInserting(Node *n, int dp, int dc) {
        if (n == NULL) {
            return;
        }
        AdoptDesc *ad = n->adesc;
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
    inline void LocatePlace(Handler *h, int &dp, int &dc, Node *&pred, Node *&curr, Node *n, Stack *s) {
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
            s->node[dc] = curr;
            dc++;
        }
    }
    inline void relocateStack(Node *last, int d, Stack *s) {
        int dp = 0;
        int dc = 0;
        Node *pred = NULL;
        Node *curr = s->head;
        Stack tempStack;
        LocatePlace(&handler, dp, dc, pred, curr, last, &tempStack);
        s->node[d] = tempStack.node[d];
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
    inline void RewindStack(Stack *s, Node *n, Node *pred, int dp) {
        // NOTE: no need to rewind stack if node is already deleted ...
        Stack saveStack = *s;
        if (s->head->ver < minStackHead) {
            abort();
        }
        while (true) {
            Stack *sNow = stack.load();
            if (s->head->ver == sNow->head->ver) {
                if (n->key > sNow->node[D - 1]->key) {
                    // if (!first_iteration) {
                    //     break;
                    // }
#ifdef MD_PRIORITY_QUEUE_DEBUG
                    auto key = s->key;
                    auto type = s->type;
                    auto id = s->nodeId;
                    auto endId = s->endNodeId;
#endif
                    *s = *sNow;
#ifdef MD_PRIORITY_QUEUE_DEBUG
                    s->key = key;
                    s->type = type;
                    s->nodeId = id;
                    s->endNodeId = endId;
                    s->rewind_type = 1;
#endif
                } else {
                    for (int i = dp; i < D; i++) {
                        s->node[i] = pred;
                    }
#ifdef MD_PRIORITY_QUEUE_DEBUG
                    s->rewind_type = 2;
#endif
                }
            } else if (s->head->ver > sNow->head->ver) {
                Node *prg = (Node *)(MD_ClearMark(sNow->head->val, MD_F_DEL));
                if (prg->key <= sNow->node[D - 1]->key) {
                    s->head = (HeadNode *)(MD_ClearMark(prg->val, MD_F_DEL));
                    for (int i = 0; i < D; i++) {
                        s->node[i] = s->head;
                    }
#ifdef MD_PRIORITY_QUEUE_DEBUG
                    s->rewind_type = 3;
#endif
                } else {
                    // if (!first_iteration) {
                    //     break;
                    // }
#ifdef MD_PRIORITY_QUEUE_DEBUG
                    auto key = s->key;
                    auto type = s->type;
                    auto id = s->nodeId;
                    auto endId = s->endNodeId;
#endif
                    *s = *sNow;
                    if (sNow->parent) {
                        s->parent = sNow->parent;
                    } else {
                        s->parent = sNow;
                    }
#ifdef MD_PRIORITY_QUEUE_DEBUG
                    s->key = key;
                    s->type = type;
                    s->nodeId = id;
                    s->endNodeId = endId;
                    s->rewind_type = 4;
#endif
                }
            } else { // s->head->ver < sNow->head->ver
                Node *prg = (Node *)(MD_ClearMark(s->head->val, MD_F_DEL));
                if (prg->key > n->key) {
                    for (int i = dp; i < D; i++) {
                        s->node[i] = pred;
                    }
#ifdef MD_PRIORITY_QUEUE_DEBUG
                    s->rewind_type = 5;
#endif
                } else {
                    s->head = (HeadNode *)(MD_ClearMark(prg->val, MD_F_DEL));
                    for (int i = 0; i < D; i++) {
                        s->node[i] = s->head;
                    }
#ifdef MD_PRIORITY_QUEUE_DEBUG
                    s->rewind_type = 6;
#endif
                }
            }
#ifdef MD_PRIORITY_QUEUE_DEBUG
            fillStack(s);
            n->insertStack = s;
            s->prev = sNow;
            s->isDelete = false;
#endif
            s->insertHead = saveStack.head;
            auto tmp = sNow;
            if (stack.compare_exchange_strong(tmp, s)) {
                while (true) {
                    auto current = s->head->stacks.load();
                    s->list = current;
                    if (s->head->stacks.compare_exchange_strong(current, s)) {
                        break;
                    }
                }
#ifdef MD_PRIORITY_QUEUE_DEBUG
                s->replaceOk = true;
#endif
                break;
            } else {
#ifdef MD_PRIORITY_QUEUE_DEBUG
                auto rewind_type = s->rewind_type;
#endif
                *s = saveStack;
#ifdef MD_PRIORITY_QUEUE_DEBUG
                s->rewind_type = rewind_type;
#endif
            }
        }
    }
#ifdef MD_PRIORITY_QUEUE_DEBUG
    inline void fillStack(Stack *s) {
        s->pid = (long)syscall(SYS_gettid);
        struct timeval tval;
        gettimeofday(&tval, NULL);
        s->time = (tval.tv_sec * 1000000LL + tval.tv_usec);
    }
#endif
    inline bool LocatePivot(Node *prg, Node *&pvt, int d, uint64_t &child) {
        while (pvt->k[d] < prg->k[d]) {
            finishInserting(pvt, d, d);
            pvt = (Node *)(MD_Clear(pvt->child[d]));
        }
        do {
            finishInserting(pvt, d, d);
            child = pvt->child[d];
        } while (!pvt->child[d].compare_exchange_weak(child, MD_SetMark(child, MD_F_PRG)) && !MD_IsMarked(child, MD_F_ALL));
        if (MD_IsMarked(child, MD_F_ADP)) {
            return false;
        } else {
            child = MD_ClearMark(child, MD_F_PRG);
            return true;
        }
    }
    void checkInsertHead();
    void checkMinStack();
    HeadNode *reclaimOne(HeadNode *headNode);
    void reclaim();
    void showStackStats();
    void showHeadStats();
#ifdef MD_PRIORITY_QUEUE_DEBUG
    void addToDeleteStackQueue(Stack *s) {
        while (true) {
            auto current = deleteStack.load();
            s->deletePrev = current;
            if (deleteStack.compare_exchange_strong(current, s)) {
                break;
            }
        }
    }
#endif
private:
    size_t N;
    int R;
    size_t basis;
    std::atomic<bool> notPurging {true};
    size_t nMarkedNodes = 0;
    std::atomic<uint64_t> _size {0};
    std::atomic<uint64_t> head{0};
    std::atomic<Stack *> stack{NULL};
    EpochManager insertEpochManager;
    EpochManager deleteEpochManager;
    uint64_t checkingMinInsertHead = 0;
    uint64_t minInsertHead = 0;
    uint64_t checkingMinStackHead = 0;
    uint64_t minStackHead = 0;
    HeadNode *recycleHead = NULL;
    Stack firstStack;
    Handler handler;
#ifdef MD_PRIORITY_QUEUE_DEBUG
    std::atomic<uint64_t> nodeId {0};
    std::atomic<Stack *> deleteStack{NULL};
#endif
};

}
