#pragma once
#include <atomic>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <assert.h>
#include "navi/log/NaviLogger.h"

// #define MD_PRIORITY_QUEUE_DEBUG
// #define MD_PRIORITY_QUEUE_NO_RECYCLE

#ifdef MD_PRIORITY_QUEUE_DEBUG
#include <vector>
#include <algorithm>
#endif

namespace md3 {

#define cpu_relax() __asm__ volatile("rep; nop\n": : :"memory")

#define MD_LOG_LEVEL SCHEDULE3
// #define MD_LOG(level, format, args...) NAVI_KERNEL_LOG(level, format, ##args)
#define MD_LOG(level, format, args...)

struct EpochCounter {
    std::atomic<uint32_t> count {0};
    char pad[60];
};

constexpr int EPOCH_SLOT_COUNT = 43;
extern std::atomic<int32_t> EPOCH_THREAD_ID_COUNTER;
thread_local extern int EPOCH_THREAD_SLOT_ID;

class EpochManager {
public:
    int enter() {
        if (EPOCH_THREAD_SLOT_ID < 0) {
            auto id = EPOCH_THREAD_ID_COUNTER++;
            if (id >= EPOCH_SLOT_COUNT) {
                id = (id % EPOCH_SLOT_COUNT);
            }
            EPOCH_THREAD_SLOT_ID = id;
        }
        auto index = EPOCH_THREAD_SLOT_ID;
        _slots[index].count++;
        return index;
    }
    void exit(int index) {
        _slots[index].count--;
    }
    void begin(bool *snapshot) {
        for (int i = 0; i < EPOCH_SLOT_COUNT; i++) {
            auto value = (0 == _slots[i].count.load());
            snapshot[i] = value;
        }
    }
    bool check(bool *snapshot) {
        bool ret = true;
        for (int i = 0; i < EPOCH_SLOT_COUNT; i++) {
            if (!snapshot[i]) {
                auto value = (0 == _slots[i].count.load());
                snapshot[i] = value;
                ret = ret && value;
            }
        }
        return ret;
    }
private:
    EpochCounter _slots[EPOCH_SLOT_COUNT] __attribute__((aligned(64)));
};

class DeleteQueues;
class EpochScope {
public:
    EpochScope(DeleteQueues *deleteQueues, EpochManager &manager)
        : _deleteQueues(deleteQueues)
        , _manager(manager)
        , _index(manager.enter())
    {
    }
    ~EpochScope();
private:
    DeleteQueues *_deleteQueues;
    EpochManager &_manager;
    int _index = 0;
};

constexpr int D = 1 * 2;
static_assert(D > 1);
static_assert((D % 2) == 0);

struct Node;

#define MD_SetMark(p, m) ((p) | (m))
#define MD_ClearMark(p, m) ((p) & ~((uint64_t)(m)))
#define MD_IsMarked(p, m) ((p) & (uint64_t)(m))
#define MD_F_ADP 0x1U
#define MD_F_PRG 0x2U
#define MD_F_DEL 0x1U
#define MD_F_ALL 0x3U
#define MD_Clear(p) MD_ClearMark(p, MD_F_ALL)

struct __attribute__((aligned(64))) RefNode {
    RefNode(uint64_t ptr)
        : ref(1)
        , val(ptr)
    {
    }
    void addRef() {
        ref++;
    }
    void unRef() {
        if (!--ref) {
            delete this;
        }
    }
    bool cas(uint64_t &oldVal, uint64_t newVal) {
        assert(newVal != 1);
        auto ret = val.compare_exchange_strong(oldVal, newVal);
        if (ret) {
            MD_LOG(MD_LOG_LEVEL, "cas change %p from 0x%lx to 0x%lx", this, oldVal, newVal);
        }
        return ret;
    }
    uint64_t fetchOr(uint64_t v) {
        assert(v != 1);
        return val.fetch_or(v);
    }
    void setVal(uint64_t v) {
        assert(v != 1);
        MD_LOG(MD_LOG_LEVEL, "change %p from 0x%lx to 0x%lx", this, val.load(), v);
        val = v;
    }
    uint64_t getVal() const {
        return val;
    }
private:
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-private-field"
#endif
    std::atomic<uint64_t> ref;
    char pad[56];
    std::atomic<uint64_t> val;
    char pad1[56];
#if defined(__clang__)
#pragma GCC diagnostic warning "-Wunused-private-field"
#endif
};

struct Node {
public:
    ~Node() {
        for (int i = 0; i < D; i++) {
            if (child[i]) {
                child[i]->unRef();
            }
        }
        if (val) {
            val->unRef();
        }
    }
public:
    RefNode *child[D];
    RefNode *val = NULL;
    size_t keyH;
    size_t keyL;
    size_t k[D];
    Node *deleteChild = NULL;
    Node *deletePrev = NULL;
#ifdef MD_PRIORITY_QUEUE_DEBUG
    uint64_t id = 0;
    uint64_t deleteId = 0;
    Node *head = NULL;
    Node *insertPred = NULL;
    Node *adescNode = NULL;
#endif
    Node *newFromNode(Node *n, int d, int index) {
        for (int i = 0; i < d; i++) {
            child[i] = NULL;
        }
        for (int i = d; i < index; i++) {
            auto c = n->child[i];
            c->addRef();
            child[i] = c;
        }
        auto c = (Node *)MD_Clear(n->child[index]->getVal());
        for (int i = index; i < D; i++) {
            auto cc = c->child[i];
            cc->addRef();
            child[i] = cc;
        }
        c->val->addRef();
        val = c->val;
        keyH = c->keyH;
        keyL = c->keyL;
        for (int i = 0; i < D; i++) {
            k[i] = c->k[i];
        }
        return c;
    }
};

class Handler {
public:
    Node *newNode() {
        auto n = new Node();
        for (int i = 0; i < D; i++) {
            n->child[i] = new RefNode(0);
        }
        return n;
    }
};

class __attribute__((aligned(64))) DeleteQueue {
public:
    DeleteQueue(EpochManager &insertEpochManager, EpochManager &deleteEpochManager)
        : _insertEpochManager(insertEpochManager)
        , _deleteEpochManager(deleteEpochManager)
    {
    }
    ~DeleteQueue() {
        endEpoch();
        startEpoch();
        endEpoch();
    }
public:
    void add(Node *n) {
        while (true) {
            auto current = _deleteQueue.load();
            n->deletePrev = current;
            if (_deleteQueue.compare_exchange_strong(current, n)) {
                break;
            }
        }
        MD_LOG(MD_LOG_LEVEL, "deletelist: %p", n);
        _deleteCount++;
    }
    void reclaim() {
        if (_deleteCount < 100) {
            return;
        }
        if (!_notReclaiming.load()) {
            return;
        }
        bool temp = true;
        if (!_notReclaiming.compare_exchange_strong(temp, false)) {
            return;
        }
        if (!_epochQueue) {
            startEpoch();
            __asm__ volatile("": : :"memory");
            if (_epochQueue) {
                _insertEpochManager.begin(_insertSnapshot);
                _deleteEpochManager.begin(_deleteSnapshot);
            }
        } else {
            if (_insertEpochManager.check(_insertSnapshot) && _deleteEpochManager.check(_deleteSnapshot)) {
                endEpoch();
            }
        }
        _notReclaiming.store(true);
    }
private:
    void startEpoch() {
        assert(!_epochQueue);
        Node *current = NULL;
        while (true) {
            current = _deleteQueue.load();
            if (_deleteQueue.compare_exchange_strong(current, NULL)) {
                break;
            }
        }
        _epochQueue = current;
    }
    void endEpoch() {
        if (!_epochQueue) {
            return;
        }
        auto n = _epochQueue;
        size_t count = 0;
        while (n) {
            auto tmp = n;
            n = n->deletePrev;
            if (tmp->deleteChild) {
                assert(!tmp->deleteChild->deleteChild);
                delete tmp->deleteChild;
            }
            delete tmp;
            count++;
            MD_LOG(MD_LOG_LEVEL, "deleted: %p", tmp);
        }
        _deleteCount.fetch_sub(count);
        _epochQueue = NULL;
    }
private:
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-private-field"
#endif
    std::atomic<Node *> _deleteQueue{NULL};
    std::atomic<size_t> _deleteCount{0};
    char pad[48];
    std::atomic<bool> _notReclaiming{true};
    Node * _epochQueue = NULL;
    EpochManager &_insertEpochManager;
    EpochManager &_deleteEpochManager;
    bool _insertSnapshot[EPOCH_SLOT_COUNT];
    bool _deleteSnapshot[EPOCH_SLOT_COUNT];
#if defined(__clang__)
#pragma GCC diagnostic warning "-Wunused-private-field"
#endif
};

constexpr int DELETE_SLOT_COUNT = 43;
extern std::atomic<int32_t> DELETE_THREAD_ID_COUNTER;
thread_local extern int DELETE_THREAD_SLOT_ID;

class DeleteQueues {
public:
    DeleteQueues() {
        for (int i = 0; i < DELETE_SLOT_COUNT; i++) {
            _queues[i] = new DeleteQueue(_insertEpochManager, _deleteEpochManager);
        }
    }
    ~DeleteQueues() {
        for (int i = 0; i < DELETE_SLOT_COUNT; i++) {
            delete _queues[i];
        }
    }
public:
    void initThreadSlotId() {
        if (DELETE_THREAD_SLOT_ID < 0) {
            auto id = DELETE_THREAD_ID_COUNTER++;
            if (id >= DELETE_SLOT_COUNT) {
                id = (id % DELETE_SLOT_COUNT);
            }
            DELETE_THREAD_SLOT_ID = id;
        }
    }
    void add(Node *n) {
        initThreadSlotId();
        _queues[DELETE_THREAD_SLOT_ID]->add(n);
    }
    void reclaim() {
        initThreadSlotId();
        _queues[DELETE_THREAD_SLOT_ID]->reclaim();
    }
    EpochManager &getInsertEpochManager() {
        return _insertEpochManager;
    }
    EpochManager &getDeleteEpochManager() {
        return _deleteEpochManager;
    }
private:
    EpochManager _insertEpochManager;
    EpochManager _deleteEpochManager;
    DeleteQueue *_queues[DELETE_SLOT_COUNT];
};

class PriorityQueue {
public:
    PriorityQueue(size_t n, int r);
    ~PriorityQueue();
public:
    bool insert(size_t keyH, size_t keyL, uint64_t val, Node *&retNode);
    uint64_t deleteMin(size_t &keyH, size_t &keyL, Node *&retNode);
    uint64_t size() {
        return _size.load();
    }
    uint64_t empty() {
        return (0 == _size.load());
    }
private:
    inline void setCoords(Node *n, size_t keyH, size_t keyL) {
        constexpr int mid = (D / 2);
        size_t quotient = keyH;
        size_t *k = n->k;
        for (int i = mid - 1; i >= 0; i--) {
            if (basis == 0) {
                k[i] = quotient;
            } else {
                k[i] = quotient % basis;
                quotient = quotient / basis;
            }
        }
        quotient = keyL;
        for (int i = D - 1; i >= mid; i--) {
            if (basis == 0) {
                k[i] = quotient;
            } else {
                k[i] = quotient % basis;
                quotient = quotient / basis;
            }
        }
    }
    int lockNode(Node *n, int end);
    bool deleteNode(Node *last, int d, Node *child);
    inline void LocatePlace(Handler *h, Node *&ppred, int &dpp, int &dp, int &dc, Node *&pred, Node *&curr, Node *n) {
        while (dc < D) {
            while (curr != NULL && n->k[dc] > curr->k[dc]) {
                ppred = pred;
                pred = curr;
                dpp = dp;
                dp = dc;
                curr = (Node *) (MD_Clear(curr->child[dc]->getVal()));
                while (curr && 0 == curr->val->getVal()) {
                    if (deleteNode(pred, dc, curr)) {
                        curr = (Node *)(MD_Clear(pred->child[dc]->getVal()));
                    }
                    break;
                }
            }
            if (curr == NULL || n->k[dc] < curr->k[dc]) {
                break;
            }
            dc++;
        }
    }
    inline void FillNewNode(Handler *h, int dp, int dc, Node *n, Node *curr) {
        for (int i = 0; i < D; i++) {
            n->child[i]->setVal(0);
        }
        for (int i = dp; i < dc; i++) {
            auto c = curr->child[i];
            c->addRef();
            n->child[i] = c;
        }
        n->child[dc]->setVal((uint64_t)curr);
    }
private:
    size_t N;
    size_t basis;
    std::atomic<uint64_t> _size {0};
    std::atomic<uint64_t> head{0};
    Node *headNode = NULL;
    DeleteQueues deleteQueues;
    Handler handler;
};

}
