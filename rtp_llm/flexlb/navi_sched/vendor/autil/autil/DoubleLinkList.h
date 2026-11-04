#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

namespace autil {
template <class T>
class DoubleLinkListNode;
} // namespace autil

namespace autil {

template <class T>
class DoubleLinkList {
public:
    DoubleLinkList();
    ~DoubleLinkList();

private:
    DoubleLinkList(const DoubleLinkList<T> &);
    DoubleLinkList &operator=(const DoubleLinkList<T> &);

public:
    inline void pushFront(DoubleLinkListNode<T> *listNode);
    inline void pushBack(DoubleLinkListNode<T> *listNode);
    inline void pushFront(const T &t);
    inline void pushBack(const T &t);
    inline void popFront();
    inline void popBack();
    inline void moveToHead(DoubleLinkListNode<T> *listNode);
    inline void moveToBack(DoubleLinkListNode<T> *listNode);
    inline void deleteNode(DoubleLinkListNode<T> *listNode);
    inline void deleteAndFreeNode(DoubleLinkListNode<T> *listNode);
    inline uint32_t getElementCount() const;
    inline DoubleLinkListNode<T> *getTail() const;
    inline DoubleLinkListNode<T> *getHead() const;
    inline bool isEmpty() const;

private:
    DoubleLinkListNode<T> *_head;
    DoubleLinkListNode<T> *_tail;

    uint32_t _elementCount;
};

//////////////////////////////////////////////////////////
template <class T>
inline DoubleLinkList<T>::DoubleLinkList() : _head(NULL), _tail(NULL), _elementCount(0) {}

template <class T>
inline DoubleLinkList<T>::~DoubleLinkList() {}

template <class T>
inline void DoubleLinkList<T>::pushFront(DoubleLinkListNode<T> *listNode) {
    assert(NULL != listNode);

    listNode->_nextListNode = NULL;
    listNode->_prevListNode = NULL;

    if (NULL == _head && NULL == _tail) {
        _head = listNode;
        _tail = listNode;
    } else {
        listNode->_nextListNode = _head;
        _head->_prevListNode = listNode;
        _head = listNode;
    }
    ++_elementCount;
}

template <class T>
inline void DoubleLinkList<T>::pushBack(DoubleLinkListNode<T> *listNode) {
    assert(NULL != listNode);

    listNode->_nextListNode = NULL;
    listNode->_prevListNode = NULL;

    if (NULL == _head && NULL == _tail) {
        _head = listNode;
        _tail = listNode;
    } else {
        listNode->_prevListNode = _tail;
        _tail->_nextListNode = listNode;
        _tail = listNode;
    }
    ++_elementCount;
}

template <class T>
inline void DoubleLinkList<T>::pushFront(const T &t) {
    auto listNode = new DoubleLinkListNode<T>(t);
    pushFront(listNode);
}

template <class T>
inline void DoubleLinkList<T>::pushBack(const T &t) {
    auto listNode = new DoubleLinkListNode<T>(t);
    pushBack(listNode);
}

template <class T>
inline void DoubleLinkList<T>::popFront() {
    auto head = getHead();
    if (head != NULL) {
        deleteAndFreeNode(head);
    }
}

template <class T>
inline void DoubleLinkList<T>::popBack() {
    auto tail = getTail();
    if (tail != NULL) {
        deleteAndFreeNode(tail);
    }
}

template <class T>
inline void DoubleLinkList<T>::moveToHead(DoubleLinkListNode<T> *listNode) {
    assert(NULL != listNode);
    deleteNode(listNode);
    pushFront(listNode);
}

template <class T>
inline void DoubleLinkList<T>::moveToBack(DoubleLinkListNode<T> *listNode) {
    assert(NULL != listNode);
    deleteNode(listNode);
    pushBack(listNode);
}

template <class T>
inline void DoubleLinkList<T>::deleteNode(DoubleLinkListNode<T> *listNode) {
    assert(NULL != listNode);

    if (listNode->_nextListNode == NULL && listNode->_prevListNode == NULL && _head != listNode) // not a node in list
    {
        return;
    }

    if (listNode->_nextListNode) {
        listNode->_nextListNode->_prevListNode = listNode->_prevListNode;
    }
    if (listNode->_prevListNode) {
        listNode->_prevListNode->_nextListNode = listNode->_nextListNode;
    }
    if (_head == listNode) {
        _head = listNode->_nextListNode;
    }
    if (_tail == listNode) {
        _tail = listNode->_prevListNode;
    }
    --_elementCount;
}

template <class T>
inline void DoubleLinkList<T>::deleteAndFreeNode(DoubleLinkListNode<T> *listNode) {
    deleteNode(listNode);
    delete listNode;
}

template <class T>
inline uint32_t DoubleLinkList<T>::getElementCount() const {
    return _elementCount;
}

template <class T>
inline DoubleLinkListNode<T> *DoubleLinkList<T>::getTail() const {
    return _tail;
}

template <class T>
inline DoubleLinkListNode<T> *DoubleLinkList<T>::getHead() const {
    return _head;
}

template <class T>
inline bool DoubleLinkList<T>::isEmpty() const {
    return _elementCount == 0;
}

} // namespace autil
