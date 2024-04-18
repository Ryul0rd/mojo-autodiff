from engine import Value, argmax
from math import sqrt
from random import randn_float64


@value
struct Neuron:
    var weights: List[Value]
    var bias: Value

    fn __init__(inout self, in_features: Int):
        self.bias = Value(0)
        self.weights = List[Value]()
        var kaiming_variance = sqrt(2 / in_features)
        for _ in range(in_features):
            self.weights.append(Value(randn_float64(0, kaiming_variance)))
        
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
            if i+1 != len(self.layers):
                current = relu(current)
        return current

    fn parameters(self) -> List[Value]:
        var params = List[Value]()
        for layer in self.layers:
            params.extend(layer[].parameters())
        return params
    
    fn print_params(self):
        for i_layer in range(len(self.layers)):
            print(String('Layer ') + i_layer)
            for i_neuron in range(len(self.layers[i_layer].neurons)):
                var neuron = self.layers[i_layer].neurons[i_neuron]
                for i_w in range(len(neuron.weights)):
                    print(String('Neuron ') + i_neuron + ' weight ' + i_w + ': ' + neuron.weights[i_w])
                print(String('Neuron ') + i_neuron + ' bias: ' + neuron.bias)
            print()


fn relu(x: Value) -> Value:
    return x.max(0)


fn relu(x: List[Value]) -> List[Value]:
    var result = List[Value](capacity=len(x))
    for i in range(len(x)):
        result.append(relu(x[i]))
    return result


fn mse(pred: Value, true: Value) -> Value:
    return (pred - true) ** 2


fn cross_entropy(logits: List[Value], true: Int) -> Value:
    var max_logit = logits[argmax(logits)]
    var log_sum_exp = Value(0)
    for logit in logits:
        log_sum_exp = log_sum_exp + (logit[] - max_logit).exp()
    log_sum_exp = max_logit + log_sum_exp.log()
    var xe = -logits[true] + log_sum_exp
    return xe


fn log_softmax(logits: List[Value]) -> List[Value]:
    var sm = softmax(logits)
    var log_sm = List[Value](capacity=len(sm))
    for val in sm:
        log_sm.append(val[].log())
    return log_sm


fn softmax(logits: List[Value]) -> List[Value]:
    var exp_values = List[Value](capacity=len(logits))
    for logit in logits:
        exp_values.append(logit[].exp())
    var exp_sum = Value(0)
    for exp_val in exp_values:
        exp_sum = exp_sum + exp_val[]
    var probs = List[Value](capacity=len(exp_values))
    for exp_val in exp_values:
        probs.append(exp_val[] / exp_sum)
    return probs


fn neg_log_likelihood(pred_probs: List[Value], true: Int) -> Value:
    return -pred_probs[true].log()
