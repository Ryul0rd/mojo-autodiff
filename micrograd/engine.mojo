from collections import Set
from math import log


@value
@register_passable('trivial')
struct Value:
    var data: Pointer[Float32]
    var grad: Pointer[Float32]
    var _prev: Pointer[Value]
    var _n_prev: Int
    var _backward: fn(prev: Pointer[Value], grad: Float32) -> None

    fn __init__(inout self, data: Float32):
        self.data = pointer_init(data)
        self.grad = pointer_init[Float32](0)
        self._prev = Pointer[Value].get_null()
        self._n_prev = 0
        self._backward = self.noop

    @staticmethod
    fn noop(prev: Pointer[Value], grad: Float32):
        pass

    fn backward(inout self):
        var topo = List[Value]()
        var visited = List[Value]()
        var stack = List[Value]()
        stack.append(self)

        while len(stack):
            var node = stack[-1]
            if node._is_in(visited):
                _ = stack.pop_back()
                topo.append(node)
                continue
            visited.append(node)
            for i in range(node._n_prev):
                var child = node._prev[i]
                if not child._is_in(visited) and not child._is_in(stack):
                    stack.append(child)

        topo.reverse()
        self.grad[] = 1
        for node in topo:
            node[]._backward(node[]._prev, node[].grad[])

    fn _is_in(self, list: List[Value]) -> Bool:
        for e in list:
            if (
                e[].data == self.data
                and e[].grad == self.grad
                and int(e[]._prev) == int(self._prev)
                and e[]._n_prev == self._n_prev
            ):
                return True
        return False

    fn __add__(self, rhs: Value) -> Value:
        fn backward(prev: Pointer[Value], grad: Float32):
            prev[0].grad[] += grad
            prev[1].grad[] += grad

        var val = Value(self.data[] + rhs.data[])
        val._prev = pointer_init(self, rhs)
        val._n_prev = 2
        val._backward = backward
        return val

    fn __sub__(self, rhs: Value) -> Value:
        fn backward(prev: Pointer[Value], grad: Float32):
            prev[0].grad[] += grad
            prev[1].grad[] -= grad

        var val = Value(self.data[] - rhs.data[])
        val._prev = pointer_init(self, rhs)
        val._n_prev = 2
        val._backward = backward
        return val

    fn __mul__(self, rhs: Value) -> Value:
        fn backward(prev: Pointer[Value], grad: Float32):
            prev[0].grad[] += grad * prev[1].data[]
            prev[1].grad[] += grad * prev[0].data[]

        var val = Value(self.data[] * rhs.data[])
        val._prev = pointer_init(self, rhs)
        val._n_prev = 2
        val._backward = backward
        return val

    fn __div__(self, rhs: Value) -> Value:
        fn backward(prev: Pointer[Value], grad: Float32):
            prev[0].grad[] += grad / prev[1].data[]
            prev[1].grad[] += grad * prev[0].data[]

        var val = Value(self.data[] / rhs.data[])
        val._prev = pointer_init(self, rhs)
        val._n_prev = 2
        val._backward = backward
        return val

    fn __pow__(self, rhs: Value) -> Value:
        fn backward(prev: Pointer[Value], grad: Float32):
            var base = prev[0].data[]
            var exponent = prev[1].data[]
            prev[0].grad[] += grad * exponent * base ** (exponent - 1)
            prev[1].grad[] += grad * base ** exponent * log(base)

        var val = Value(self.data[] ** rhs.data[])
        val._prev = pointer_init(self, rhs)
        val._n_prev = 2
        val._backward = backward
        return val

    fn __pow__(self, rhs: Float32) -> Value:
        return self ** Value(rhs)

    fn min(self, x: Value) -> Value:
        fn backward(prev: Pointer[Value], grad: Float32):
            if prev[0].data[] < prev[1].data[]:
                prev[0].grad[] += grad
            else:
                prev[1].data[] += grad

        var val = Value(self.data[]) if self < x else Value(x.data[])
        val._prev = pointer_init(self, x)
        val._n_prev = 2
        val._backward = backward
        return val

    fn min(self, x: Float32) -> Value:
        return self.min(Value(x))

    fn max(self, x: Value) -> Value:
        fn backward(prev: Pointer[Value], grad: Float32):
            if prev[0].data[] > prev[1].data[]:
                prev[0].grad[] += grad
            else:
                prev[1].grad[] += grad
        
        var val = Value(self.data[]) if self > x else Value(x.data[])
        val._prev = pointer_init(self, x)
        val._n_prev = 2
        val._backward = backward
        return val

    fn max(self, x: Float32) -> Value:
        return self.max(Value(x))

    fn __str__(self) -> String:
        return String('Value(data=') + self.data[] + ', grad=' + self.grad[] + ')'

    fn __hash__(self) -> Int:
        var h1 = hash(int(self.data))
        var h2 = hash(int(self.grad))
        var h3 = hash(int(self._prev))
        return h1 * 31 + h2 * 37 + h3 * 101

    fn __eq__(self, other: Value) -> Bool:
        return self.data[] == other.data[]

    fn __ne__(self, other: Value) -> Bool:
        return self.data[] != other.data[]

    fn __gt__(self, rhs: Value) -> Bool:
        return self.data[] > rhs.data[]

    fn __gt__(self, rhs: Float32) -> Bool:
        return self.data[] > rhs

    fn __lt__(self, rhs: Value) -> Bool:
        return self.data[] < rhs.data[]

    fn __lt__(self, rhs: Float32) -> Bool:
        return self.data[] < rhs


@value
struct SGD:
    var params: List[Value]
    var learning_rate: Float32

    fn step(self):
        for i in range(len(self.params)):
            self.params[i].data[] -= self.learning_rate * self.params[i].grad[]

    fn zero_grad(self):
        for i in range(len(self.params)):
            self.params[i].grad[] = 0


fn pointer_init[T: AnyRegType](*args: T) -> Pointer[T]:
    var ptr = Pointer[T].alloc(len(args))
    for i in range(len(args)):
        ptr[i] = args[i]
    return ptr


fn mse(pred: Value, true: Value) -> Value:
    return (pred - true) ** 2
