#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace navi_sched {

// Forward-mode automatic differentiation for a formula with N parameters.
template <size_t N>
class AutoGrad {
public:
    AutoGrad() = default;
    AutoGrad(double value) : _value(value) {}

    static AutoGrad parameter(double value, size_t index)
    {
        AutoGrad result(value);
        result._gradient.fill(0.0);
        if (index < N) {
            result._gradient[index] = 1.0;
        }
        return result;
    }

    double value() const { return _value; }
    static constexpr size_t gradientCount() { return N; }

    double gradient(size_t index) const
    {
        assert(index < N);
        return _gradient[index];
    }

    AutoGrad operator-() const { return scale(-1.0, -_value); }

    friend AutoGrad operator+(const AutoGrad &left, const AutoGrad &right)
    {
        return combine(left, right, left._value + right._value, 1.0, 1.0);
    }

    friend AutoGrad operator-(const AutoGrad &left, const AutoGrad &right)
    {
        return combine(left, right, left._value - right._value, 1.0, -1.0);
    }

    friend AutoGrad operator*(const AutoGrad &left, const AutoGrad &right)
    {
        return combine(left, right, left._value * right._value, right._value,
                       left._value);
    }

    friend AutoGrad operator/(const AutoGrad &left, const AutoGrad &right)
    {
        auto inverse = 1.0 / right._value;
        return combine(left, right, left._value * inverse, inverse,
                       -left._value * inverse * inverse);
    }

    friend AutoGrad operator+(AutoGrad left, double right)
    {
        left._value += right;
        return left;
    }

    friend AutoGrad operator+(double left, AutoGrad right)
    {
        right._value += left;
        return right;
    }

    friend AutoGrad operator-(AutoGrad left, double right)
    {
        left._value -= right;
        return left;
    }

    friend AutoGrad operator-(double left, const AutoGrad &right)
    {
        return right.scale(-1.0, left - right._value);
    }

    friend AutoGrad operator*(const AutoGrad &left, double right)
    {
        return left.scale(right, left._value * right);
    }

    friend AutoGrad operator*(double left, const AutoGrad &right)
    {
        return right.scale(left, left * right._value);
    }

    friend AutoGrad operator/(const AutoGrad &left, double right)
    {
        auto inverse = 1.0 / right;
        return left.scale(inverse, left._value * inverse);
    }

    friend AutoGrad operator/(double left, const AutoGrad &right)
    {
        auto inverse = 1.0 / right._value;
        return right.scale(-left * inverse * inverse, left * inverse);
    }

    friend AutoGrad sqrt(const AutoGrad &value)
    {
        auto resultValue = std::sqrt(value._value);
        return value.scale(0.5 / resultValue, resultValue);
    }

private:
    AutoGrad scale(double factor, double resultValue) const
    {
        AutoGrad result(resultValue);
        for (size_t index = 0; index < N; ++index) {
            result._gradient[index] = factor * _gradient[index];
        }
        return result;
    }

    static AutoGrad combine(const AutoGrad &left, const AutoGrad &right, double value,
                            double leftFactor, double rightFactor)
    {
        AutoGrad result(value);
        for (size_t index = 0; index < N; ++index) {
            result._gradient[index] =
                leftFactor * left._gradient[index] + rightFactor * right._gradient[index];
        }
        return result;
    }

private:
    double _value = 0.0;
    std::array<double, N> _gradient{};
};

// Reverse-mode automatic differentiation for a runtime parameter count.
class AutoGradTape {
public:
    using NodeId = uint32_t;
    static constexpr NodeId INVALID_NODE =
        std::numeric_limits<NodeId>::max();

    class Value {
    public:
        Value() = default;
        Value(double value) : _value(value) {}

        double value() const { return _value; }

        Value operator-() const { return scale(-1.0, -_value); }

        friend Value operator+(const Value &left, const Value &right)
        {
            return binary(left, right, left._value + right._value, 1.0, 1.0);
        }

        friend Value operator-(const Value &left, const Value &right)
        {
            return binary(left, right, left._value - right._value, 1.0, -1.0);
        }

        friend Value operator*(const Value &left, const Value &right)
        {
            return binary(left, right, left._value * right._value, right._value,
                          left._value);
        }

        friend Value operator/(const Value &left, const Value &right)
        {
            auto inverse = 1.0 / right._value;
            return binary(left, right, left._value * inverse, inverse,
                          -left._value * inverse * inverse);
        }

        friend Value operator+(Value left, double right)
        {
            left._value += right;
            return left;
        }

        friend Value operator+(double left, Value right)
        {
            right._value += left;
            return right;
        }

        friend Value operator-(Value left, double right)
        {
            left._value -= right;
            return left;
        }

        friend Value operator-(double left, const Value &right)
        {
            return right.scale(-1.0, left - right._value);
        }

        friend Value operator*(const Value &left, double right)
        {
            return left.scale(right, left._value * right);
        }

        friend Value operator*(double left, const Value &right)
        {
            return right.scale(left, left * right._value);
        }

        friend Value operator/(const Value &left, double right)
        {
            auto inverse = 1.0 / right;
            return left.scale(inverse, left._value * inverse);
        }

        friend Value operator/(double left, const Value &right)
        {
            auto inverse = 1.0 / right._value;
            return right.scale(-left * inverse * inverse, left * inverse);
        }

        friend Value sqrt(const Value &value)
        {
            auto resultValue = std::sqrt(value._value);
            return value.scale(0.5 / resultValue, resultValue);
        }

        friend Value log1p(const Value &value)
        {
            auto resultValue = std::log1p(value._value);
            return value.scale(1.0 / (1.0 + value._value), resultValue);
        }

    private:
        friend class AutoGradTape;
        template <size_t N>
        friend class LinearSums;
        template <size_t N>
        friend class StaticParameterLinearSums;

        Value(double value, AutoGradTape *tape, NodeId nodeId,
              double nodeDerivative = 1.0)
            : _value(value)
            , _tape(tape)
            , _nodeId(nodeId)
            , _nodeDerivative(nodeDerivative)
        {
        }

        static Value binary(const Value &left, const Value &right, double value,
                            double leftDerivative, double rightDerivative)
        {
            auto *tape = selectTape(left, right);
            if (!tape) {
                return Value(value);
            }
            if (!left._tape) {
                return Value(value, tape, right._nodeId,
                             rightDerivative * right._nodeDerivative);
            }
            if (!right._tape) {
                return Value(value, tape, left._nodeId,
                             leftDerivative * left._nodeDerivative);
            }
            return Value(
                value, tape,
                tape->addNode(
                    left._nodeId,
                    leftDerivative * left._nodeDerivative,
                    right._nodeId,
                    rightDerivative * right._nodeDerivative));
        }

        Value scale(double derivative, double value) const
        {
            if (!_tape) {
                return Value(value);
            }
            return Value(value, _tape, _nodeId,
                         derivative * _nodeDerivative);
        }

        static AutoGradTape *selectTape(const Value &left, const Value &right)
        {
            if (left._tape && right._tape && left._tape != right._tape) {
                assert(false);
                std::abort();
            }
            return left._tape ? left._tape : right._tape;
        }

    private:
        double _value = 0.0;
        AutoGradTape *_tape = nullptr;
        NodeId _nodeId = INVALID_NODE;
        double _nodeDerivative = 1.0;
    };

    template <size_t N>
    class LinearSums {
    public:
        void add(const Value &value,
                 const std::array<double, N> &coefficients)
        {
            for (size_t index = 0; index < N; ++index) {
                _values[index] += value._value * coefficients[index];
            }
            if (!_tape && value._tape) {
                _tape = value._tape;
                _edgeBegin = _tape->edgePosition();
                for (size_t term = 0; term < _termCount; ++term) {
                    for (size_t index = 0; index < N; ++index) {
                        _tape->addEdge(INVALID_NODE, 0.0);
                    }
                }
            }
            if (_tape && value._tape && _tape != value._tape) {
                assert(false);
                std::abort();
            }
            if (_tape) {
                for (size_t index = 0; index < N; ++index) {
                    _tape->addEdge(
                        value._tape ? value._nodeId : INVALID_NODE,
                        value._tape
                            ? coefficients[index] * value._nodeDerivative
                            : 0.0);
                }
            }
            ++_termCount;
        }

        std::array<Value, N> finish()
        {
            std::array<Value, N> result;
            for (size_t index = 0; index < N; ++index) {
                if (_tape) {
                    auto nodeId = _tape->addLinearNode(
                        _edgeBegin + index, _termCount, N);
                    result[index] = Value(_values[index], _tape, nodeId);
                } else {
                    result[index] = Value(_values[index]);
                }
            }
            return result;
        }

    private:
        AutoGradTape *_tape = nullptr;
        NodeId _edgeBegin = INVALID_NODE;
        size_t _termCount = 0;
        std::array<double, N> _values{};
    };

    // Linear reductions over sequential parameters with invariant coefficients.
    template <size_t N>
    class StaticParameterLinearSums {
    public:
        explicit StaticParameterLinearSums(AutoGradTape *tape)
            : _tape(tape)
            , _parameterBegin(
                  tape ? tape->sequentialParameterPosition() : INVALID_NODE)
            , _derivativeBegin(
                  tape ? tape->staticDerivativePosition() : INVALID_NODE)
        {
        }

        void add(double parameterValue,
                 const std::array<double, N> &coefficients)
        {
            for (size_t index = 0; index < N; ++index) {
                _values[index] += parameterValue * coefficients[index];
            }
            if (_tape) {
                _tape->addStaticDerivatives(coefficients);
            }
            ++_termCount;
        }

        std::array<Value, N> finish()
        {
            std::array<Value, N> result;
            if (!_tape) {
                for (size_t index = 0; index < N; ++index) {
                    result[index] = Value(_values[index]);
                }
                return result;
            }
            _tape->consumeSequentialParameters(_termCount);
            for (size_t index = 0; index < N; ++index) {
                auto nodeId = _tape->addStaticLinearNode(
                    _parameterBegin, _derivativeBegin + index,
                    _termCount, N);
                result[index] = Value(_values[index], _tape, nodeId);
            }
            return result;
        }

    private:
        AutoGradTape *_tape = nullptr;
        NodeId _parameterBegin = INVALID_NODE;
        NodeId _derivativeBegin = INVALID_NODE;
        size_t _termCount = 0;
        std::array<double, N> _values{};
    };

    explicit AutoGradTape(size_t parameterCount, size_t nodeCapacity = 0,
                          size_t edgeCapacity = 0,
                          size_t staticDerivativeCapacity = 0)
        : _parameterBound(parameterCount, 0)
    {
        if (parameterCount >= INVALID_NODE) {
            assert(false);
            std::abort();
        }
        reserve(nodeCapacity, edgeCapacity, staticDerivativeCapacity);
    }

    size_t parameterCount() const { return _parameterBound.size(); }

    void reserve(size_t nodeCapacity, size_t edgeCapacity = 0,
                 size_t staticDerivativeCapacity = 0)
    {
        _nodes.reserve(nodeCapacity);
        _edges.reserve(edgeCapacity);
        _staticDerivatives.reserve(staticDerivativeCapacity);
        _adjoints.reserve(parameterCount() + nodeCapacity);
    }

    void reset()
    {
        _nodes.clear();
        _edges.clear();
        _staticDerivatives.clear();
        _adjoints.clear();
        std::fill(_parameterBound.begin(), _parameterBound.end(), 0);
        _sequentialParameterCursor = 0;
        _replayCursor = 0;
        _replayEdgeCursor = 0;
        _replayStaticDerivativeCursor = 0;
        _replaying = false;
        _recorded = false;
        _indexedParameters = false;
    }

    bool rewind()
    {
        if (!_recorded) {
            return false;
        }
        if (_indexedParameters) {
            std::fill(_parameterBound.begin(), _parameterBound.end(), 0);
        }
        _sequentialParameterCursor = 0;
        _replayCursor = 0;
        _replayEdgeCursor = 0;
        _replayStaticDerivativeCursor = 0;
        _replaying = true;
        return true;
    }

    Value parameter(double value, size_t index)
    {
        if (_sequentialParameterCursor > 0 || index >= parameterCount() ||
            _parameterBound[index]) {
            assert(false);
            std::abort();
        }
        _indexedParameters = true;
        _parameterBound[index] = 1;
        _recorded = true;
        return Value(value, this, static_cast<NodeId>(index));
    }

    bool backward(const Value &output)
    {
        _adjoints.assign(parameterCount() + _nodes.size(), 0.0);
        if (!output._tape) {
            return true;
        }
        auto adjointCount = _adjoints.size();
        if (output._tape != this || output._nodeId >= adjointCount) {
            return false;
        }
        if (_replaying &&
            (_replayCursor != _nodes.size() ||
             _replayEdgeCursor != _edges.size() ||
             _replayStaticDerivativeCursor !=
                 _staticDerivatives.size())) {
            return false;
        }
        _adjoints[output._nodeId] = output._nodeDerivative;
        for (size_t index = _nodes.size(); index > 0; --index) {
            const auto &node = _nodes[index - 1];
            auto adjoint = _adjoints[parameterCount() + index - 1];
            if (NodeKind::STATIC_PARAMETERS == node.kind) {
                auto derivativeIndex = node.edgeBegin;
                auto parameterIndex = node.leftNode;
                for (NodeId edgeOffset = 0;
                     edgeOffset < node.edgeCount; ++edgeOffset) {
                    _adjoints[parameterIndex++] +=
                        adjoint * _staticDerivatives[derivativeIndex];
                    derivativeIndex += node.edgeStride;
                }
            } else if (NodeKind::EDGES == node.kind) {
                auto edgeIndex = node.edgeBegin;
                for (NodeId edgeOffset = 0;
                     edgeOffset < node.edgeCount; ++edgeOffset) {
                    const auto &edge = _edges[edgeIndex];
                    if (edge.node != INVALID_NODE) {
                        _adjoints[edge.node] +=
                            adjoint * edge.derivative;
                    }
                    edgeIndex += node.edgeStride;
                }
            } else {
                if (node.leftNode != INVALID_NODE) {
                    _adjoints[node.leftNode] +=
                        adjoint * node.leftDerivative;
                }
                if (node.rightNode != INVALID_NODE) {
                    _adjoints[node.rightNode] +=
                        adjoint * node.rightDerivative;
                }
            }
        }
        return true;
    }

    bool backward(const Value &output, std::vector<double> &gradients)
    {
        if (!backward(output)) {
            return false;
        }
        gradients.assign(_adjoints.begin(),
                         _adjoints.begin() + parameterCount());
        return true;
    }

    const double *gradientData() const
    {
        return _adjoints.size() >= parameterCount()
                   ? _adjoints.data()
                   : nullptr;
    }

private:
    enum class NodeKind : uint8_t {
        BINARY,
        EDGES,
        STATIC_PARAMETERS,
    };

    struct Node {
        NodeId leftNode = INVALID_NODE;
        NodeId rightNode = INVALID_NODE;
        double leftDerivative = 0.0;
        double rightDerivative = 0.0;
        NodeId edgeBegin = INVALID_NODE;
        NodeId edgeCount = 0;
        NodeId edgeStride = 0;
        NodeKind kind = NodeKind::BINARY;
    };

    struct Edge {
        NodeId node = INVALID_NODE;
        double derivative = 0.0;
    };

    NodeId addNode(NodeId leftNode, double leftDerivative,
                   NodeId rightNode, double rightDerivative)
    {
        if (_replaying) {
            if (_replayCursor >= _nodes.size()) {
                assert(false);
                std::abort();
            }
            auto &node = _nodes[_replayCursor];
            if (NodeKind::BINARY != node.kind ||
                node.leftNode != leftNode ||
                node.rightNode != rightNode) {
                assert(false);
                std::abort();
            }
            node.leftDerivative = leftDerivative;
            node.rightDerivative = rightDerivative;
            return static_cast<NodeId>(parameterCount() + _replayCursor++);
        }
        if (parameterCount() + _nodes.size() >= INVALID_NODE) {
            assert(false);
            std::abort();
        }
        _nodes.push_back(
            {leftNode, rightNode, leftDerivative, rightDerivative,
             INVALID_NODE, 0, 0, NodeKind::BINARY});
        _recorded = true;
        return static_cast<NodeId>(parameterCount() + _nodes.size() - 1);
    }

    NodeId edgePosition() const
    {
        auto position = _replaying ? _replayEdgeCursor : _edges.size();
        if (position >= INVALID_NODE) {
            assert(false);
            std::abort();
        }
        return static_cast<NodeId>(position);
    }

    void addEdge(NodeId node, double derivative)
    {
        if (_replaying) {
            if (_replayEdgeCursor >= _edges.size() ||
                _edges[_replayEdgeCursor].node != node) {
                assert(false);
                std::abort();
            }
            _edges[_replayEdgeCursor++].derivative = derivative;
            return;
        }
        if (_edges.size() >= INVALID_NODE) {
            assert(false);
            std::abort();
        }
        _edges.push_back({node, derivative});
        _recorded = true;
    }

    NodeId addLinearNode(NodeId edgeBegin, size_t edgeCount,
                         size_t edgeStride)
    {
        if (edgeCount >= INVALID_NODE || edgeStride >= INVALID_NODE) {
            assert(false);
            std::abort();
        }
        Node node;
        node.edgeBegin = edgeBegin;
        node.edgeCount = static_cast<NodeId>(edgeCount);
        node.edgeStride = static_cast<NodeId>(edgeStride);
        node.kind = NodeKind::EDGES;
        if (_replaying) {
            if (_replayCursor >= _nodes.size()) {
                assert(false);
                std::abort();
            }
            const auto &recorded = _nodes[_replayCursor];
            if (recorded.edgeBegin != node.edgeBegin ||
                recorded.edgeCount != node.edgeCount ||
                recorded.edgeStride != node.edgeStride ||
                recorded.kind != node.kind) {
                assert(false);
                std::abort();
            }
            return static_cast<NodeId>(parameterCount() + _replayCursor++);
        }
        if (parameterCount() + _nodes.size() >= INVALID_NODE) {
            assert(false);
            std::abort();
        }
        _nodes.emplace_back(node);
        _recorded = true;
        return static_cast<NodeId>(parameterCount() + _nodes.size() - 1);
    }

    NodeId sequentialParameterPosition() const
    {
        if (_sequentialParameterCursor >= INVALID_NODE) {
            assert(false);
            std::abort();
        }
        return static_cast<NodeId>(_sequentialParameterCursor);
    }

    void consumeSequentialParameters(size_t count)
    {
        if (_indexedParameters ||
            _sequentialParameterCursor + count > parameterCount()) {
            assert(false);
            std::abort();
        }
        _sequentialParameterCursor += count;
        _recorded = true;
    }

    NodeId staticDerivativePosition() const
    {
        auto position = _replaying ? _replayStaticDerivativeCursor
                                   : _staticDerivatives.size();
        if (position >= INVALID_NODE) {
            assert(false);
            std::abort();
        }
        return static_cast<NodeId>(position);
    }

    template <size_t N>
    void addStaticDerivatives(
        const std::array<double, N> &derivatives)
    {
        if (_replaying) {
            if (_replayStaticDerivativeCursor + N >
                _staticDerivatives.size()) {
                assert(false);
                std::abort();
            }
            _replayStaticDerivativeCursor += N;
            return;
        }
        if (_staticDerivatives.size() + N >= INVALID_NODE) {
            assert(false);
            std::abort();
        }
        for (size_t index = 0; index < N; ++index) {
            _staticDerivatives.emplace_back(derivatives[index]);
        }
        _recorded = true;
    }

    NodeId addStaticLinearNode(NodeId parameterBegin,
                               NodeId derivativeBegin,
                               size_t derivativeCount,
                               size_t derivativeStride)
    {
        if (derivativeCount >= INVALID_NODE ||
            derivativeStride >= INVALID_NODE) {
            assert(false);
            std::abort();
        }
        Node node;
        node.leftNode = parameterBegin;
        node.edgeBegin = derivativeBegin;
        node.edgeCount = static_cast<NodeId>(derivativeCount);
        node.edgeStride = static_cast<NodeId>(derivativeStride);
        node.kind = NodeKind::STATIC_PARAMETERS;
        if (_replaying) {
            if (_replayCursor >= _nodes.size()) {
                assert(false);
                std::abort();
            }
            const auto &recorded = _nodes[_replayCursor];
            if (recorded.leftNode != node.leftNode ||
                recorded.edgeBegin != node.edgeBegin ||
                recorded.edgeCount != node.edgeCount ||
                recorded.edgeStride != node.edgeStride ||
                recorded.kind != node.kind) {
                assert(false);
                std::abort();
            }
            return static_cast<NodeId>(parameterCount() +
                                       _replayCursor++);
        }
        if (parameterCount() + _nodes.size() >= INVALID_NODE) {
            assert(false);
            std::abort();
        }
        _nodes.emplace_back(node);
        _recorded = true;
        return static_cast<NodeId>(parameterCount() +
                                   _nodes.size() - 1);
    }

private:
    std::vector<Node> _nodes;
    std::vector<Edge> _edges;
    std::vector<double> _staticDerivatives;
    std::vector<uint8_t> _parameterBound;
    std::vector<double> _adjoints;
    size_t _sequentialParameterCursor = 0;
    size_t _replayCursor = 0;
    size_t _replayEdgeCursor = 0;
    size_t _replayStaticDerivativeCursor = 0;
    bool _replaying = false;
    bool _recorded = false;
    bool _indexedParameters = false;
};

} // namespace navi_sched
