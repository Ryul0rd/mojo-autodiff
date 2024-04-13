from engine import Value


@value
struct Neuron:
    var weights: List[Value]
    var bias: Value

    fn __init__(inout self, in_features: Int):
        self.bias = Value(0)
        self.weights = List[Value]()
        for _ in range(in_features):
            self.weights.append(Value(0))
        
    fn __call__(self, x: List[Value]) -> Value:
        return self.forward(x)

    fn forward(self, x: List[Value]) -> Value:
        var running_sum = self.weights[0] * x[0]
        for i in range(1, len(x)):
            running_sum = running_sum + self.weights[i] * x[i]
        return running_sum + self.bias

    fn parameters(self) -> List[Value]:
        var params = List[Value](capacity=len(self.weights)+1)
        params.extend(self.weights)
        params.append(self.bias)
        return params


@value
struct Linear:
    var neurons: List[Neuron]
    var in_features: Int
    var out_features: Int

    fn __init__(inout self, in_features: Int, out_features: Int):
        self.in_features = in_features
        self.out_features = out_features
        self.neurons = List[Neuron]()
        for _ in range(out_features):
            self.neurons.append(Neuron(in_features))
        
    fn __call__(self, x: List[Value]) -> List[Value]:
        return self.forward(x)

    fn forward(self, x: List[Value]) -> List[Value]:
        var result = List[Value](capacity=len(self.neurons))
        for neuron in self.neurons:
            result.append(neuron[](x))
        return result

    fn parameters(self) -> List[Value]:
        var params = List[Value](capacity=self.in_features*self.out_features+self.out_features)
        for neuron in self.neurons:
            params.extend(neuron[].parameters())
        return params


@value
struct MLP:
    var layers: List[Linear]
    var activation: fn(pre_act: List[Value]) -> List[Value]

    fn __init__(
        inout self,
        in_features: Int,
        hidden_sizes: List[Int],
        out_features: Int,
        activation: fn(pre_act: List[Value]) -> List[Value] = relu
    ):
        self.activation = activation
        self.layers = List[Linear]()
        if len(hidden_sizes) == 0:
            self.layers.append(Linear(in_features, out_features))
            return
        self.layers.append(Linear(in_features, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.layers.append(Linear(hidden_sizes[-1], out_features))

    fn __call__(self, x: List[Value]) -> List[Value]:
        return self.forward(x)

    fn forward(self, x: List[Value]) -> List[Value]:
        var current = x
        for i in range(len(self.layers)):
            current = self.layers[i](current)
            if i+1 == len(self.layers):
                current = relu(current)
        return current

    fn parameters(self) -> List[Value]:
        var params = List[Value]()
        for layer in self.layers:
            params.extend(layer[].parameters())
        return params


fn relu(x: Value) -> Value:
    return x.max(0)


fn relu(x: List[Value]) -> List[Value]:
    var result = List[Value](capacity=len(x))
    for i in range(len(x)):
        result.append(relu(x[i]))
    return result


fn mse(pred: Value, true: Value) -> Value:
    return (pred - true) ** 2
