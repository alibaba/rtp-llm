#pragma once

#include <stddef.h>

namespace autil {

template <class T>
class DoubleLinkListNode {
public:
    DoubleLinkListNode();
    DoubleLinkListNode(const T &t);

public:
    void reset();
    T &getData();

private:
    DoubleLinkListNode(const DoubleLinkListNode &);

public:
    DoubleLinkListNode<T> *_nextListNode;
    DoubleLinkListNode<T> *_prevListNode;
    T _data;
};

////////////////////////////////////////////////////
template <class T>
DoubleLinkListNode<T>::DoubleLinkListNode() {
    reset();
}

template <class T>
DoubleLinkListNode<T>::DoubleLinkListNode(const T &t) {
    reset();
    _data = t;
}

template <class T>
inline void DoubleLinkListNode<T>::reset() {
    _nextListNode = NULL;
    _prevListNode = NULL;
}

template <class T>
inline T &DoubleLinkListNode<T>::getData() {
    return _data;
}

} // namespace autil
