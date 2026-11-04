#include "navi/engine/MDPriorityQueue3.h"
#include <limits>

namespace md3 {

std::atomic<int32_t> EPOCH_THREAD_ID_COUNTER{0};
thread_local int EPOCH_THREAD_SLOT_ID = -1;

std::atomic<int32_t> DELETE_THREAD_ID_COUNTER{0};
thread_local int DELETE_THREAD_SLOT_ID = -1;

EpochScope::~EpochScope() {
    _manager.exit(_index);
    _deleteQueues->reclaim();
}

PriorityQueue::PriorityQueue(size_t n, int r)
    : N(n)
{
    (void)r;
    double basisD = ceil(pow(N, 1.0 / (D / 2)));
    basis = (basisD >= (double)std::numeric_limits<size_t>::max()) ? std::numeric_limits<size_t>::max() : (size_t)basisD;
    headNode = handler.newNode();
    headNode->val = new RefNode(1);
    setCoords(headNode, 0, 0);
    head.store((uint64_t)headNode);
}

PriorityQueue::~PriorityQueue() {
    while (true) {
        auto child = (Node *)MD_Clear(headNode->child[0]->getVal());
        if (child) {
            deleteNode(headNode, 0, child);
        } else {
            break;
        }
    }
    delete headNode;
}

bool PriorityQueue::insert(size_t keyH, size_t keyL, uint64_t val, Node *&retNode) {
    EpochScope epochScope(&deleteQueues, deleteQueues.getInsertEpochManager());
    auto h = &handler;
    Node *n = h->newNode();
    n->keyH = keyH;
    n->keyL = keyL;
    n->val = new RefNode(val);
    setCoords(n, keyH, keyL);
    RefNode *saveChilds[D];
    for (int i = 0; i < D; i++) {
        saveChilds[i] = n->child[i];
    }
    _size++;
    while (true) {
        Node *ppred = NULL;
        Node *pred = NULL;
        int dpp = 0;
        int dp = 0;
        int dc = 0;
        auto headNode = (Node *)MD_Clear(head.load());
        Node *curr = headNode;
        LocatePlace(h, ppred, dpp, dp, dc, pred, curr, n);
        if (dc == D) {
            _size--;
            delete n;
            // this key is already in the queue
            // std::cout << "this key is already in the queue, keyH: " << keyH << ", keyL: " << keyL << std::endl;
            return false;
        }
        FillNewNode(h, dp, dc, n, curr);
        bool success = false;
        uint64_t temp = (uint64_t)curr;
        if (pred) {
            success = pred->child[dp]->cas(temp, (uint64_t)n);
        } else {
            success = head.compare_exchange_strong(temp, (uint64_t)n);
        }
        if (success) {
            for (int i = dp; i < dc; i++) {
                saveChilds[i]->unRef();
            }
            if (dp < dc) {
                MD_LOG(MD_LOG_LEVEL,
                       "insert move %p %lu%lu %lu, by %p %lu%lu %lu, dp: %d, dc: %d",
                       curr,
                       curr->keyH,
                       curr->keyL,
                       curr->val->getVal(),
                       n,
                       n->keyH,
                       n->keyL,
                       n->val->getVal(),
                       dp,
                       dc);
            }
            MD_LOG(MD_LOG_LEVEL,
                   "insert: %p, key: %lu %lu, val: %lu, pred: %p %d %lu, curr: %p",
                   n,
                   n->keyH,
                   n->keyL,
                   n->val->getVal(),
                   pred,
                   dp,
                   pred->val->getVal(),
                   curr);
            retNode = n;
            return true;
        } else {
            for (int i = 0; i < D; i++) {
                if (n->child[i] != saveChilds[i]) {
                    n->child[i]->unRef();
                    n->child[i] = saveChilds[i];
                }
            }
        }
    }
}

int PriorityQueue::lockNode(Node *n, int end) {
    int d = D - 1;
    while (d >= end) {
        auto childNode = n->child[d];
        auto temp = childNode->getVal();
        if (MD_IsMarked(temp, MD_F_ADP)) {
            deleteNode(n, d, (Node *)MD_Clear(temp));
            continue;
        } else if (temp == MD_F_PRG) {
            d--;
            continue;
        } else if (MD_IsMarked(temp, MD_F_PRG)) {
            break;
        }
        childNode->cas(temp, MD_SetMark(temp, MD_F_PRG));
        if (MD_IsMarked(temp, MD_F_PRG) && MD_Clear(temp)) {
            break;
        }
    }
    if (d >= end) {
        assert(MD_Clear(n->child[d]->getVal()));
    }
    return d;
}

bool PriorityQueue::deleteNode(Node *last, int d, Node *child) {
    while (true) {
        auto temp = (uint64_t)child;
        if (last->child[d]->cas(temp, MD_SetMark((uint64_t)child, MD_F_ADP))) {
            break;
        } else {
            if (MD_IsMarked(temp, MD_F_ADP)) {
                break;
            } else if (MD_IsMarked(temp, MD_F_PRG)) {
                return false;
            } else {
                return true;
            }
        }
    }
    auto temp = MD_SetMark((uint64_t)child, MD_F_ADP);
    auto index = lockNode(child, d);
    if (index < d) {
        auto ret = last->child[d]->cas(temp, 0);
        if (ret) {
            deleteQueues.add(child);
            MD_LOG(MD_LOG_LEVEL,
                   "delete null child %p %lu%lu %lu, parent %p %d, index: %d",
                   child,
                   child->keyH,
                   child->keyL,
                   child->val->getVal(),
                   last,
                   d,
                   index);
        }
        return true;
    }
    auto newNode = new Node();
    auto copy = newNode->newFromNode(child, d, index);
    if (!last->child[d]->cas(temp, (uint64_t)newNode)) {
        delete newNode;
        return true;
    } else {
        child->deleteChild = copy;
        deleteQueues.add(child);
        MD_LOG(MD_LOG_LEVEL,
               "delete node %p %lu%lu %lu, parent %p %d, copy child %p %lu%lu %d to new node %p",
               child,
               child->keyH,
               child->keyL,
               child->val->getVal(),
               last,
               d,
               copy,
               copy->keyH,
               copy->keyL,
               index,
               newNode);
    }
    return true;
}

uint64_t PriorityQueue::deleteMin(size_t &keyH, size_t &keyL, Node *&retNode) {
    EpochScope epochScope(&deleteQueues, deleteQueues.getDeleteEpochManager());
    Node *s[D];
    auto headNode = (Node *)head.load();
    if (!headNode) {
        retNode = NULL;
        keyH = 0;
        keyL = 0;
        return 0;
    }
    for (int i = 0; i < D; i++) {
        s[i] = headNode;
    }
    int d = D - 1;
    while (true) {
        Node *last = s[d];
        auto childVal = last->child[d]->getVal();
        Node *child = (Node *)(MD_Clear(childVal));
        if (child == NULL) {
            if (d == 0) {
                retNode = NULL;
                keyH = 0;
                keyL = 0;
                return 0;
            }
            d--;
            continue;
        }
        auto val = child->val->getVal();
        if (val) {
            if (child->val->cas(val, 0)) {
                deleteNode(last, d, child);
                retNode = child;
                keyH = child->keyH;
                keyL = child->keyL;
                _size--;
                assert(keyH != 0);
                MD_LOG(MD_LOG_LEVEL, "pop: %p, key: %lu%lu, val: %lu", child, keyH, keyL, val);
                return val;
            }
        }
        for (int i = d; i < D; i++) {
            s[i] = child;
        }
        d = D - 1;
    }
}

} // namespace md
