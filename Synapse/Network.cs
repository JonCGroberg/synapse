namespace ML;

/// <summary>
///     Please initialize with data before using
///     <code>Network.Init(inputData,outputData)</code>
///     <remarks> input and output data both default to a matrix1x1 with value [[1]] </remarks>
/// </summary>
public class Network
{
    private int _nHiddenNodes = 3,
                _nHiddens,
                _nInputs,
                _nOutputs;

    private Matrix _inputData  = new(new[] { new[] { 1.0 } });
    private Matrix _outputData = new(new[] { new[] { 1.0 } });

    public int[]    Layers;
    public Matrix[] Weights;

    /// <param name = "nInputs" >number of inputs</param>
    /// <param name = "nOutputs" >number of outputs</param>
    /// <param name = "nHiddens" >number of hidden layers</param>
    public Network(int nInputs, int nOutputs, int nHiddens = 3)
    {
        _nInputs  = nInputs;
        _nOutputs = nOutputs;
        _nHiddens = nHiddens;

        //init array and matrix sizes
        Layers  = new int[1                + _nHiddens + 1];
        Weights = new Matrix[Layers.Length - 1];

        //create layers
        Layers[0]  = _nInputs;
        Layers[^1] = _nOutputs;
        for (int i = 1; i < Layers.Length - 1; i++) Layers[i] = _nHiddenNodes;

        //create weights
        for (int m = 0; m < Weights.Length; m++) Weights[m] = new Matrix(Layers[m], Layers[m + 1]);
    }


    public void Init(double[][] inputData, double[][] outputData) =>
        Init(new Matrix(inputData), new Matrix(outputData));

    public void Init(Matrix inputData, Matrix outputData)
    {
        _outputData = outputData.Columns != _nOutputs
                          ? throw new Exception("out data does not match network settings")
                          : outputData;
        _inputData = inputData.Columns != _nInputs
                         ? throw new Exception("input data does not match network settings")
                         : inputData;
        InitWeights();
    }

    public void InitWeights(Random value) => Weights = Weights.Select(x => x.Fill(value.NextDouble)).ToArray();


    private void InitWeights(double value) => Weights = Weights.Select(x => x.Fill(value)).ToArray();


    private void InitWeights()
    {
        NormalDist normalDist = new();
        InitWeights(normalDist);
        Xavierize();
    }

    public Matrix Error()   => _outputData - Predict();
    public Matrix Predict() => ForwardProp();

    ///Recursive forward propagation starting at Input Layer
    public Matrix ForwardProp() => ForwardProp(_inputData);

    private Matrix ForwardProp(Matrix firstLayer) =>
        Weights.Aggregate(firstLayer, (prevLayer, currentWeights) => (prevLayer * currentWeights).Map(Math.Tanh));

    // public double TanH(double x)
    // {
    //     return (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x));
    // }

    /// Normalize by dividing the weights by their corresponding layer size
    private void Xavierize() => Weights = Weights.Zip(Layers, (a, b) => a / b).ToArray();

    public override string ToString() => Weights.Select(x => x.ToString()).Aggregate((a, b) => a + "\n" + b);
}

/// <summary>
///     Each Layer consists of a the weights <see cref = "Matrix" /> that precede it, and the layers activation function.
///     <returns>The activated weights</returns>
/// </summary>
/// <seealso cref = "Matrix" />
public class Layer : Matrix
{
//because of the way that matrices work the amount of nodes in a layer happens to be the amount of columns
    public int Nodes => Columns;

    //Calling without a activation function defaults to identity function (returns the weights just the way they are); ie input data is a plain matrix
    //Can create a Layer with a matrix or directly with data (which gets converted to a matrix).
    public Layer(Matrix     matrix) : base(matrix) => ActivationFunction = x => x;
    public Layer(double[][] data) : base(data) => ActivationFunction = x => x;

    public Layer(Func<double, double> activationFunction, double[][] data) : this(activationFunction, new Matrix(data))
    {
    }

    public Layer(Func<double, double> activationFunction, Matrix matrix) : base(matrix) =>
        ActivationFunction = activationFunction;

    public Func<double, double> ActivationFunction { get; }

    // activate w1 and then multiply with w2, also change activation function from a to b.
    public static Layer operator *(Layer l1, Layer l2) =>
        new Layer(l2.ActivationFunction, ElementWiseOperation(l1, l2, (w1, w2) => l1.ActivationFunction(w1) * w2));
}