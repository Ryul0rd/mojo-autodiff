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
            if list_contains(visited, node):
                _ = stack.pop_back()
                topo.append(node)
                continue
            visited.append(node)
            for i in range(node._n_prev):
                var child = node._prev[i]
                if not list_contains(visited, child):
                    stack.append(child)

        topo.reverse()
        self.grad[] = 1
        for node in topo:
            node[]._backward(node[]._prev, node[].grad[])

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

    fn __str__(self) -> String:
        return String('Value(data=') + self.data[] + ', grad=' + self.grad[] + ')'

    fn __hash__(self) -> Int:
        var h1 = hash(int(self.data))
        var h2 = hash(int(self.grad))
        var h3 = hash(int(self._prev))
        return h1 * 31 + h2 * 37 + h3 * 101

    fn __eq__(self, other: Self) -> Bool:
        return hash(self) == hash(other)

    fn __ne__(self, other: Self) -> Bool:
        return hash(self) != hash(other)


@value
struct SGD:
    var params: List[Pointer[Value]]
    var learning_rate: Float32

    fn step(self):
        for i in range(len(self.params)):
            self.params[i][].data[] -= self.learning_rate * self.params[i][].grad[]

    fn zero_grad(self):
        for i in range(len(self.params)):
            self.params[i][].grad[] = 0


fn pointer_init[T: AnyRegType](*args: T) -> Pointer[T]:
    var ptr = Pointer[T].alloc(len(args))
    for i in range(len(args)):
        ptr[i] = args[i]
    return ptr


fn list_contains[T: KeyElement](list: List[T], val: T) -> Bool:
    for e in list:
        if e[] == val:
            return True
    return False


# fn mse(pred: Value, true: Value) -> Value:
#     var error = (pred - true)
#     return error * error


fn mse(pred: Value, true: Value) -> Value:
    return (pred - true) ** 2
