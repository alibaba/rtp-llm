#include "navi/engine/MDPriorityQueue2.h"
#include "navi/log/NaviLogger.h"

namespace md2 {

#define barrier() __asm__ volatile("": : :"memory")

std::atomic<int32_t> THREAD_ID_COUNTER{0};
thread_local int THREAD_SLOT_ID = -1;

PriorityQueue::PriorityQueue(size_t n, int r)
    : N(n)
    , R(r)
{
    basis = ceil(pow(N, 1.0 / D));
    head.store(0);
}

PriorityQueue::~PriorityQueue() {
    showStats();
}

void PriorityQueue::showStats() {
    showHeadStats();
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
        Node *pred = NULL;
        int dp = 0;
        int dc = 0;
        auto headNode = (HeadNode *)MD_Clear(head.load());
        Node *curr = headNode;
        LocatePlace(h, dp, dc, pred, curr, n);
        if (dc == D) {
            if (n->adesc.load()) {
                delete n->adesc.load();
            }
            delete n;
            insertEpochManager.exit(epochIndex);
            // this key is already in the queue
            // std::cout << "this key is already in the queue: " << key << std::endl;
            return false;
        }
        finishInserting(curr, dp, dc);
        FillNewNode(h, dp, dc, n, curr);
#ifdef MD_PRIORITY_QUEUE_DEBUG
        n->head = headNode;
        n->insertPred = pred;
#endif
        bool success = false;
        uint64_t temp = (uint64_t)curr;
        if (pred) {
            success = pred->child[dp].compare_exchange_strong(temp, (uint64_t)n);
        } else {
            success = head.compare_exchange_strong(temp, (uint64_t)n);
        }
        if (success) {
            // NAVI_KERNEL_LOG(SCHEDULE3, "insert: %p, key: %lu, val: %lu, pred: %p %d, curr: %p", n, n->key, n->val.load(), pred, dp, curr);
#ifdef MD_PRIORITY_QUEUE_DEBUG
            if (dp < dc) {
                curr->adescNode = n;
            }
            s->endNodeId = nodeId.load();
#endif
            finishInserting(n, dp, dc);
            _size++;
            retNode = n;
            insertEpochManager.exit(epochIndex);
            return true;
        }
        if (MD_IsMarked(temp, MD_F_PRG)) {
            finishDeleting();
        }
    }
}

void PriorityQueue::finishDeleting() {
    auto headVal = head.load();
    if (!MD_IsMarked(headVal, MD_F_PRG)) {
        return;
    }
    auto headNode = (HeadNode *)MD_Clear(headVal);
    if (!headNode) {
        return;
    }
    finishInserting(headNode, 0, D);
    Node *newHead = NULL;
    int d = D - 1;
    while (d >= 0) {
        uint64_t childVal = headNode->child[d].load();
        uint64_t temp = MD_ClearMark(childVal, MD_F_ADP);
        if (temp == MD_F_PRG) {
            d--;
            continue;
        } else if (MD_IsMarked(temp, MD_F_PRG)) {
            break;
        }
        Node *child = (Node *)MD_Clear(temp);
        if (child) {
            headVal = MD_SetMark(headVal, MD_F_PRG);
            for (int i = 0; i < d; i++) {
                headNode->child[i].fetch_or(MD_F_PRG);
            }
            for (int i = d; i < D; i++) {
                finishInserting(child, i, i);
                child->child[i].fetch_or(MD_F_PRG);
            }
            if (!newHead) {
                newHead = handler.newNode();
            }
            newHead->setFromNode(child);
            for (int i = 0; i < d; i++) {
                newHead->child[i] = MD_Clear(headNode->child[i]);
            }
            for (int i = d; i < D; i++) {
                newHead->child[i] = MD_Clear(newHead->child[i]);
            }
            if (headNode->child[d].compare_exchange_strong(childVal, MD_SetMark((uint64_t)newHead, MD_F_PRG))) {
                // NAVI_KERNEL_LOG(SCHEDULE3, "delete: %p, key: %lu, val: %lu", newHead, newHead->key, newHead->val.load());
                newHead = NULL;
            }//  else {
            //     NAVI_KERNEL_LOG(
            //         SCHEDULE3, "delete failed: %p, key: %lu, val: %lu", newHead, newHead->key, newHead->val.load());
            // }
            if (MD_IsMarked(childVal, MD_F_PRG)) {
                break;
            }
        } else {
            headNode->child[d].compare_exchange_strong(childVal, MD_F_PRG);
        }
        cpu_relax();
    }
    uint64_t swap = MD_SetMark(headVal, MD_IsMarked(head.load(), MD_F_PRG));
    if (d >= 0) {
        auto v = headNode->child[d].load();
        head.compare_exchange_strong(swap, MD_Clear(v));
    } else {
        head.compare_exchange_strong(swap, 0);
    }
    if (newHead) {
        delete newHead;
        newHead = NULL;
    }
}

uint64_t PriorityQueue::deleteMin(uint64_t &key, Node *&retNode) {
    Node *newHead = NULL;
    // int count = 0;
    while (true) {
        uint64_t headVal = head.load();
        Node *headNode = NULL;
        while (true) {
            headNode = (HeadNode *)MD_Clear(headVal);
            if (!headNode) {
                if (newHead) {
                    delete newHead;
                }
                key = 0;
                retNode = NULL;
                return 0;
            }
            if (MD_IsMarked(headVal, MD_F_PRG)) {
                break;
            }
            auto newVal = MD_SetMark(headVal, MD_F_PRG);
            if (head.compare_exchange_strong(headVal, newVal)) {
                headVal = newVal;
                break;
            }
        }
        finishInserting(headNode, 0, D);
        int d = D - 1;
        uint64_t childVal;
        uint64_t temp;
        // int type = 0;
        while (d >= 0) {
            childVal = headNode->child[d].load();
            temp = MD_ClearMark(childVal, MD_F_ADP);
            if (temp == MD_F_PRG) {
                d--;
                continue;
            } else if (MD_IsMarked(temp, MD_F_PRG)) {
                // type = 1;
                break;
            }
            Node *child = (Node *)MD_Clear(temp);
            if (child) {
                for (int i = 0; i < d; i++) {
                    headNode->child[i].fetch_or(MD_F_PRG);
                }
                for (int i = d; i < D; i++) {
                    finishInserting(child, i, i);
                    child->child[i].fetch_or(MD_F_PRG);
                }
                if (!newHead) {
                    newHead = handler.newNode();
                }
                newHead->setFromNode(child);
                for (int i = 0; i < d; i++) {
                    newHead->child[i] = MD_Clear(headNode->child[i]);
                }
                for (int i = d; i < D; i++) {
                    newHead->child[i] = MD_Clear(newHead->child[i]);
                }
                if (headNode->child[d].compare_exchange_strong(childVal, MD_SetMark((uint64_t)newHead, MD_F_PRG))) {
                    // NAVI_KERNEL_LOG(
                    //     SCHEDULE3, "delete1: %p, key: %lu, val: %lu", newHead, newHead->key, newHead->val.load());
                    newHead = NULL;
                    // type = 2;
                } else {
                    // type = 3;
                    // NAVI_KERNEL_LOG(
                    //     SCHEDULE3, "delete1 failed: %p, key: %lu, val: %lu", newHead, newHead->key, newHead->val.load());
                }
                if (MD_IsMarked(childVal, MD_F_PRG)) {
                    break;
                }
            } else {
                headNode->child[d].compare_exchange_strong(childVal, MD_F_PRG);
            }
        }
        // (void)type;
        uint64_t swap = MD_SetMark(headVal, MD_IsMarked(head.load(), MD_F_PRG));
        if (d >= 0) {
            auto v = headNode->child[d].load();
            head.compare_exchange_strong(swap, MD_Clear(v));
        } else {
            head.compare_exchange_strong(swap, 0);
        }
        auto val = headNode->val.load();
        if (val) {
            if (headNode->val.compare_exchange_strong(val, 0)) {
                if (newHead) {
                    delete newHead;
                }
                // NAVI_KERNEL_LOG(SCHEDULE3, "pop: %p, key: %lu, val: %lu", headNode, headNode->key, val);
                _size--;
                key = headNode->key;
                retNode = headNode;
                return val;
            }
        }
        cpu_relax();
        // count++;
        // if (count > 30) {
        //     std::cout << "count: " << count << ", type: " << type << ", d = " << d << std::endl;
        //     count = 0;
        // }
    }
}

void PriorityQueue::checkInsertHead() {
    if (checkingMinInsertHead == 0) {
        barrier();
        insertEpochManager.begin();
    } else {
        if (insertEpochManager.check()) {
            minInsertHead = checkingMinInsertHead;
            checkingMinInsertHead = 0;
        }
    }
}

void PriorityQueue::reclaim() {
    checkInsertHead();
}

HeadNode *PriorityQueue::reclaimOne(HeadNode *headNode) {
    HeadNode *retHead = NULL;
    // Node *prg = (Node *)(MD_ClearMark(headNode->val, MD_F_DEL));
    // if (prg) {
    //     retHead = (HeadNode *)(MD_ClearMark(prg->val, MD_F_DEL));
    // }
    // headNode->d = 0;
    // int d = D - 1;
    // while (true) {
    //     Node *last = s.node[d];
    //     auto childPtr = last->child[d].load();
    //     Node *child = (Node *)(MD_Clear(childPtr));
    //     auto isPurgeNode = (last->key != 0) && (MD_Clear(last->val) != 0);
    //     if (childPtr == 0 || MD_IsMarked(childPtr, MD_F_ALL) || isPurgeNode) {
    //         auto c = s.node[d];
    //         if (c && c->d == d) {
    //             if (c->adesc.load() != 0) {
    //                 delete c->adesc.load();
    //             }
    //             if (c->finishAdesc) {
    //                 delete c->finishAdesc;
    //             }
    //             delete c;
    //         }
    //         if (d == 0) {
    //             return retHead;
    //         }
    //         d--;
    //         continue;
    //     }
    //     child->d = d;
    //     for (int i = d; i < D; i++) {
    //         auto c = s.node[i];
    //         if (c && c->d == i) {
    //             if (c->adesc.load() != 0) {
    //                 delete c->adesc.load();
    //             }
    //             if (c->finishAdesc) {
    //                 delete c->finishAdesc;
    //             }
    //             delete c;
    //         }
    //         s.node[i] = child;
    //     }
    //     d = D - 1;
    // }
    return retHead;
}

} // namespace md
