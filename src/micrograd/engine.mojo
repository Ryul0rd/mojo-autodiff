from math import log, exp


@value
struct Value:
    var data: AnyPointer[Float32]
    var grad: AnyPointer[Float32]
    var _prev: AnyPointer[Value]
    var _n_prev: Int
    var _reference_count: AnyPointer[Int]
    var _backward: fn(prev: AnyPointer[Value], grad: Float32) -> None

    fn __init__(inout self, data: Float32):
        self.data = pointer_init(data)
        self.grad = pointer_init[Float32](0)
        self._prev = AnyPointer[Value]()
        self._n_prev = 0
        self._reference_count = pointer_init(1)
        self._backward = self.noop

    fn __copyinit__(inout self, existing: Self):
        existing._reference_count[] += 1
        self.data = existing.data
        self.grad = existing.grad
        self._prev = existing._prev
        self._n_prev = existing._n_prev
        self._reference_count = existing._reference_count
        self._backward = existing._backward

    fn __del__(owned self):
        self._reference_count[] -= 1
        if self._reference_count[] == 0:
            for i in range(self._n_prev):
                _ = (self._prev + i).take_value()
            self.data.free()
            self.grad.free()
            self._prev.free()
            self._reference_count.free()

    @staticmethod
    fn noop(prev: AnyPointer[Value], grad: Float32):
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
        fn backward(prev: AnyPointer[Value], grad: Float32):
            prev[0].grad[] += grad
            prev[1].grad[] += grad

        var val = Value(self.data[] + rhs.data[])
        val._prev = pointer_init(self, rhs)
        val._n_prev = 2
        val._backward = backward
        return val

    fn __sub__(self, rhs: Value) -> Value:
        fn backward(prev: AnyPointer[Value], grad: Float32):
            prev[0].grad[] += grad
            prev[1].grad[] -= grad

        var val = Value(self.data[] - rhs.data[])
        val._prev = pointer_init(self, rhs)
        val._n_prev = 2
        val._backward = backward
        return val

    fn __neg__(self) -> Value:
        return Value(0) - self

    fn __mul__(self, rhs: Value) -> Value:
        fn backward(prev: AnyPointer[Value], grad: Float32):
            prev[0].grad[] += grad * prev[1].data[]
            prev[1].grad[] += grad * prev[0].data[]

        var val = Value(self.data[] * rhs.data[])
        val._prev = pointer_init(self, rhs)
        val._n_prev = 2
        val._backward = backward
        return val

    fn __truediv__(self, rhs: Value) -> Value:
        fn backward(prev: AnyPointer[Value], grad: Float32):
            prev[0].grad[] += grad / prev[1].data[]
            prev[1].grad[] += grad * prev[0].data[]

        var val = Value(self.data[] / rhs.data[])
        val._prev = pointer_init(self, rhs)
        val._n_prev = 2
        val._backward = backward
        return val

    fn __pow__(self, rhs: Value) -> Value:
        fn backward(prev: AnyPointer[Value], grad: Float32):
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
        fn backward(prev: AnyPointer[Value], grad: Float32):
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
        fn backward(prev: AnyPointer[Value], grad: Float32):
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

    fn log(self) -> Value:
        fn backward(prev: AnyPointer[Value], grad: Float32):
            prev[0].grad[] += 1 / prev[0].data[] * grad

        var val = Value(log(self.data[]))
        val._prev = pointer_init(self)
        val._n_prev = 1
        val._backward = backward
        return val

    fn exp(self) -> Value:
        fn backward(prev: AnyPointer[Value], grad: Float32):
            prev[0].grad[] += exp(prev[0].data[]) * grad

        var val = Value(exp(self.data[]))
        val._prev = pointer_init(self)
        val._n_prev = 1
        val._backward = backward
        return val

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


fn argmax(vec: List[Value]) -> Int:
    var highest_index = -1
    var highest_value = Float32.MIN_FINITE
    for i in range(len(vec)):
        if vec[i] > highest_value:
            highest_value = vec[i].data[]
            highest_index = i
    return highest_index


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


fn pointer_init[T: CollectionElement](*args: T) -> AnyPointer[T]:
    var ptr = AnyPointer[T].alloc(len(args))
    for i in range(len(args)):
        ptr[i] = args[i]
    return ptr
