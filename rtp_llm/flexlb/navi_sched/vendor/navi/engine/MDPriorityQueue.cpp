#include "navi/engine/MDPriorityQueue.h"

namespace md {

#define barrier() __asm__ volatile("": : :"memory")

std::atomic<int32_t> THREAD_ID_COUNTER{0};
thread_local int THREAD_SLOT_ID = -1;

PriorityQueue::PriorityQueue(size_t n, int r)
    : N(n)
    , R(r)
{
    basis = ceil(pow(N, 1.0 / D));
    recycleHead = new HeadNode();
    recycleHead->val = MD_F_DEL;
    recycleHead->adesc = NULL;
    recycleHead->key = 0;
    setCoords(recycleHead, 0);
    recycleHead->ver = 1;
    for (int i = 0; i < D; i++) {
        recycleHead->child[i].store(0);
    }
    head.store((uint64_t)(recycleHead));
    firstStack.head = recycleHead;
    for (int i = 0; i < D; i++) {
        firstStack.node[i] = recycleHead;
    }
    stack.store(&firstStack);
}

PriorityQueue::~PriorityQueue() {
    showStats();
#ifndef MD_PRIORITY_QUEUE_NO_RECYCLE
    while (recycleHead) {
        recycleHead = reclaimOne(recycleHead);
    }
#endif
    recycleHead = NULL;
}

void PriorityQueue::showStats() {
    showStackStats();
    showHeadStats();
}

void PriorityQueue::showStackStats() {
#ifdef MD_PRIORITY_QUEUE_DEBUG
    std::vector<std::pair<int32_t, int32_t>> seekCounts;
    Stack *prev = deleteStack;
    (void)prev;
    Stack *stack = deleteStack;
    size_t sum = 0;
    while (stack) {
        sum += stack->seekCount;
        seekCounts.push_back(std::make_pair(stack->seekCount, stack->followPurgeCount));
        prev = stack;
        stack = stack->deletePrev;
    }
    std::sort(seekCounts.begin(), seekCounts.end());
    size_t count = seekCounts.size();
    int j = 0;
    for (size_t i = 0; i < count; i += std::max(1lu, (count / 300))) {
        std::cout << j++ << ": " << seekCounts[i].first << ", " << seekCounts[i].second << std::endl;
    }
    if (count > 0) {
        std::cout << "seek count, total count: " << count << ", avg count: " << sum / count
                  << ", max seekCount: " << seekCounts[count - 1].first << ", " << seekCounts[count - 1].second
                  << std::endl;
    }
#endif
}

void PriorityQueue::showHeadStats() {
#ifdef MD_PRIORITY_QUEUE_DEBUG
    std::vector<int32_t> nodeCounts;
    HeadNode *head = recycleHead;
    size_t sum = 0;
    while (head) {
        sum += head->nodeCount;
        nodeCounts.push_back(head->nodeCount);
        Node *prg = (Node *)(MD_ClearMark(head->val, MD_F_DEL));
        if (prg) {
            head = (HeadNode *)(MD_ClearMark(prg->val, MD_F_DEL));
        } else {
            head = NULL;
        }
    }
    std::sort(nodeCounts.begin(), nodeCounts.end());
    size_t count = nodeCounts.size();
    int j = 0;
    for (size_t i = 0; i < count; i += std::max(1lu, (count / 300))) {
        std::cout << j++ << ": " << nodeCounts[i] << std::endl;
    }
    if (count > 0) {
        std::cout << "Head node count, total count: " << count << ", avg count: " << sum / count
                  << ", max nodeCount: " << nodeCounts[count - 1] << std::endl;
    }
#endif
}

bool PriorityQueue::insert(size_t key, uint64_t val, Node *&retNode) {
    auto epochIndex = insertEpochManager.enter();
    auto h = &handler;
    Stack *s = h->newStack();
#ifdef MD_PRIORITY_QUEUE_DEBUG
    s->type = 1;
    s->key = key;
    s->isDelete = false;
    s->nodeId = nodeId.load();
#endif
    Node *n = h->newNode();
    n->key = key;
    n->val = val;
#ifdef MD_PRIORITY_QUEUE_DEBUG
    n->id = nodeId.fetch_add(1);
#endif
    setCoords(n, key);
    for (int i = 0; i < D; i++) {
        n->child[i].store(0);
    }
    while (true) {
        s->parent = NULL;
        Node *pred = NULL;
        int dp = 0;
        int dc = 0;
        s->head = (HeadNode *)(head.load());
        if (s->head->ver < minStackHead) {
            abort();
        }
        Node *curr = s->head;
        LocatePlace(h, dp, dc, pred, curr, n, s);
        if (dc == D) {
            if (n->adesc.load()) {
                delete n->adesc.load();
            }
            delete n;
            delete s;
            insertEpochManager.exit(epochIndex);
            // this key is already in the queue
            // std::cout << "this key is already in the queue: " << key << std::endl;
            return false;
        }
        finishInserting(curr, dp, dc);
        FillNewNode(h, dp, dc, n, curr);
#ifdef MD_PRIORITY_QUEUE_DEBUG
        n->head = s->head;
        n->insertPred = pred;
#endif
        uint64_t temp = (uint64_t)curr;
        if (pred->child[dp].compare_exchange_strong(temp, (uint64_t)n)) {
#ifdef MD_PRIORITY_QUEUE_DEBUG
            if (dp < dc) {
                curr->adescNode = n;
            }
            s->endNodeId = nodeId.load();
#endif
            finishInserting(n, dp, dc);
            auto deleteEpochIndex = deleteEpochManager.enter();
            RewindStack(s, n, pred, dp);
            deleteEpochManager.exit(deleteEpochIndex);
            _size++;
            retNode = n;
            insertEpochManager.exit(epochIndex);
            return true;
        }
    }
}

bool PriorityQueue::deleteMinImpl(uint64_t &key, Node *&retNode, uint64_t &value) {
    auto epochIndex = deleteEpochManager.enter();
    auto h = &handler;
    Stack *s = h->newStack();
    Stack *sOld = stack.load();
    *s = *sOld;
    s->parent = NULL;
    if (s->head->ver < minStackHead) {
        abort();
    }
#ifdef MD_PRIORITY_QUEUE_DEBUG
    fillStack(s);
    s->type = 2;
    s->key = 0;
    s->rewind_type = 0;
    s->isDelete = true;
    s->nodeId = nodeId.load();
    s->changeOld = false;
    s->prev = NULL;
    s->ss = sOld;
#endif
    int d = D - 1;
#ifdef MD_PRIORITY_QUEUE_DEBUG
    uint64_t currentNodeId = nodeId.load();
    (void)currentNodeId;
#endif
    uint64_t seekCount = 0;
    uint64_t followPurgeCount = 0;
    bool firstUpdateFailed = false;
    while (true) {
        seekCount++;
#ifdef MD_PRIORITY_QUEUE_DEBUG
        s->seekCount++;
#endif
        Node *last = s->node[d];
        finishInserting(last, d, d);
        auto childVal = last->child[d].load();
        if (MD_IsMarked(childVal, MD_F_ADP)) {
            relocateStack(last, d, s);
            if (s->node[d] != last) {
                continue;
            }
        }
        Node *child = (Node *)(MD_Clear(childVal));
        if (child == NULL) {
            if (d == 0) {
                // std::cout << "empty" << std::endl;
                value = 0;
                delete s;
                deleteEpochManager.exit(epochIndex);
                return true;
            }
            d--;
            continue;
        }
        uint64_t val = child->val;
        if (MD_IsMarked(val, MD_F_DEL)) {
            if (MD_Clear(val) == 0) {
                for (int i = d; i < D; i++) {
                    s->node[i] = child;
                }
            } else {
                s->head = (HeadNode *)(MD_Clear(val));
                for (int i = 0; i < D; i++) {
                    s->node[i] = s->head;
                }
#ifdef MD_PRIORITY_QUEUE_DEBUG
                s->isHead = true;
                s->key = child->key;
                s->prev = sOld;
                s->endNodeId = nodeId.load();
#endif
                if (followPurgeCount > 3) {
                    value = 0;
                    delete s;
                    deleteEpochManager.exit(epochIndex);
                    return false;
                }
                if (!firstUpdateFailed) {
                    auto tmp = sOld;
                    if (stack.compare_exchange_strong(tmp, s)) {
#ifdef MD_PRIORITY_QUEUE_DEBUG
                        addToDeleteStackQueue(s);
#endif
                        while (true) {
                            auto current = s->head->stacks.load();
                            s->list = current;
                            if (s->head->stacks.compare_exchange_strong(current, s)) {
                                break;
                            }
                        }
                        deleteEpochManager.exit(epochIndex);
                        return false;
                    } else {
                        firstUpdateFailed = true;
                    }
                }
                followPurgeCount++;
#ifdef MD_PRIORITY_QUEUE_DEBUG
                s->isHead = false;
                s->followPurgeCount++;
                s->lastPurgeNode = child;
                s->failedCount++;
#endif
            }
            d = D - 1;
        } else {
            if (child->val.compare_exchange_strong(val, MD_F_DEL)) {
                for (int i = d; i < D; i++) {
                    s->node[i] = child;
                }
#ifdef MD_PRIORITY_QUEUE_DEBUG
                s->key = child->key;
                child->deleteStack = s;
                child->deleteId = nodeId.fetch_add(1);
                s->prev = sOld;
                s->endNodeId = nodeId.load();
#endif
                auto tmp = sOld;
                bool success = stack.compare_exchange_strong(tmp, s);
#ifdef MD_PRIORITY_QUEUE_DEBUG
                s->replaceOk = success;
                addToDeleteStackQueue(s);
#endif
                int marked = nMarkedNodes++;
                if (marked > R) {
                    if (s->node[D - 1]->key != 0) {
                        purge(s->head, s->node[D - 1], h);
                    }
                }
                if (success) {
                    while (true) {
                        auto current = s->head->stacks.load();
                        s->list = current;
                        if (s->head->stacks.compare_exchange_strong(current, s)) {
                            break;
                        }
                    }
                } else {
#ifndef MD_PRIORITY_QUEUE_NO_RECYCLE
                    delete s;
#endif
                }
                (void)success;
                _size--;
                key = child->key;
                retNode = child;
                value = val;
                deleteEpochManager.exit(epochIndex);
                return true;
            }
        }
    }
}

void PriorityQueue::purge(HeadNode *hn, Node *prg, Handler *h) {
    if (!notPurging.load()) {
        return;
    }
    bool temp = true;
    if (!notPurging.compare_exchange_strong(temp, false)) {
        return;
    }
    if ((uint64_t)(hn) != head.load()) {
        notPurging.store(true);
        return;
    }
    nMarkedNodes = 0;
    HeadNode *hnNew = h->newHeadNode();
    Node *prgNew = h->newNode();
    prgNew->setFromNode(prg);
    hnNew->val = MD_F_DEL;
    hnNew->ver = hn->ver + 1;
    hnNew->key = hn->key;
#ifdef MD_PRIORITY_QUEUE_DEBUG
    hnNew->prev = hn;
    hnNew->id = nodeId.fetch_add(1);
    prgNew->head = hnNew;
#endif
    setCoords(hnNew, 0);
    for (int i = 0; i < D; i++) {
        hnNew->child[i].store(0);
    }
    int d = 0;
    Node *pvt = hn;
    uint64_t child;
    while (d < D) {
        if (!LocatePivot(prg, pvt, d, child)) {
            pvt = hn;
            d = 0;
            continue;
        }
        if (hn == pvt) {
            hnNew->child[d].store(child);
            prgNew->child[d].store(MD_F_ADP);
        } else {
            prgNew->child[d].store(child);
            if (d == 0 || prgNew->child[d - 1].load() == MD_F_ADP) {
                hnNew->child[d].store((uint64_t)prgNew);
            } else {
                hnNew->child[d].store(0);
            }
        }
        d++;
    }
    hn->val.store(MD_SetMark((uint64_t)prg, MD_F_DEL));
    prg->val.store(MD_SetMark((uint64_t)hnNew, MD_F_DEL));
    head.store((uint64_t)hnNew);
    reclaim();
    notPurging.store(true);
    return;
}

void PriorityQueue::checkInsertHead() {
    if (checkingMinInsertHead == 0) {
        checkingMinInsertHead = ((HeadNode *)(head.load()))->ver;
        barrier();
        insertEpochManager.begin();
    } else {
        if (insertEpochManager.check()) {
            minInsertHead = checkingMinInsertHead;
            checkingMinInsertHead = 0;
        }
    }
}

void PriorityQueue::checkMinStack() {
    if (minInsertHead == 0) {
        return;
    }
    if (checkingMinStackHead == 0) {
        checkingMinStackHead = std::min(minInsertHead, stack.load()->head->ver);
        barrier();
        deleteEpochManager.begin();
    } else {
        if (deleteEpochManager.check()) {
            minStackHead = checkingMinStackHead;
            checkingMinStackHead = 0;
            // std::cout << "success: " << minStackHead << ", " << minInsertHead << ", " << (uint64_t)this << std::endl;
        }
    }
}

void PriorityQueue::reclaim() {
    checkInsertHead();
    checkMinStack();
    if (minStackHead == 0) {
        return;
    }
    // std::cout << "recycleHead->ver: " << recycleHead->ver << ", minStackHead: " << minStackHead << std::endl;
#ifndef MD_PRIORITY_QUEUE_NO_RECYCLE
    while (recycleHead->ver < minStackHead) {
        recycleHead = reclaimOne(recycleHead);
    }
#endif
}

HeadNode *PriorityQueue::reclaimOne(HeadNode *headNode) {
    Stack *stack = headNode->stacks.load();
    while (stack) {
        auto tmp = stack;
        stack = stack->list;
        delete tmp;
    }
    HeadNode *retHead = NULL;
    Node *prg = (Node *)(MD_ClearMark(headNode->val, MD_F_DEL));
    if (prg) {
        retHead = (HeadNode *)(MD_ClearMark(prg->val, MD_F_DEL));
    }
    Stack s;
    headNode->d = 0;
    for (int i = 0; i < D; i++) {
        s.node[i] = headNode;
    }
    int d = D - 1;
    while (true) {
        Node *last = s.node[d];
        auto childPtr = last->child[d].load();
        Node *child = (Node *)(MD_Clear(childPtr));
        auto isPurgeNode = (last->key != 0) && (MD_Clear(last->val) != 0);
        if (childPtr == 0 || MD_IsMarked(childPtr, MD_F_ALL) || isPurgeNode) {
            auto c = s.node[d];
            if (c && c->d == d) {
                if (c->adesc.load() != 0) {
                    delete c->adesc.load();
                }
                if (c->finishAdesc) {
                    delete c->finishAdesc;
                }
                delete c;
            }
            s.node[d] = NULL;
            if (d == 0) {
                return retHead;
            }
            d--;
            continue;
        }
        child->d = d;
        for (int i = d; i < D; i++) {
            auto c = s.node[i];
            if (c && c->d == i) {
                if (c->adesc.load() != 0) {
                    delete c->adesc.load();
                }
                if (c->finishAdesc) {
                    delete c->finishAdesc;
                }
                delete c;
            }
            s.node[i] = child;
        }
        d = D - 1;
    }
    return retHead;
}

} // namespace md
