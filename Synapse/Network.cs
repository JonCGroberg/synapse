sing System.Runtime.CompilerServices;
using Function = System.Func<double, double>;

namespace Synapse;

/// <summary>
///     Please initialize with data before using
///     <code>Network.Init(inputData,outputData)</code>
///     <remarks> input and output data both default to a matrix1x1 with value [[1]] </remarks>
/// </summary>
public class Network
{
    private int _nHiddenNodes = 4,
                _nHiddens,
                _nInputs,
                _nOutputs;

    private Matrix _inputData  = new(new[] { new[] { 1.0 } });
    private Matrix _outputData = new(new[] { new[] { 1.0 } });

    public Layer [] Layers;
//   public Matrix[] Weights;

    /// <param name = "nInputs" >number of inputs</param>
    /// <param name = "nOutputs" >number of outputs</param>
    /// <param name = "nHiddens" >number of hidden layers</param>
    public Network(int nInputs, int nOutputs, int nHiddens = 3)
    {
        _nInputs  = nInputs;
        _nOutputs = nOutputs;
        _nHiddens = nHiddens;

        //init array and matrix size
        Layers  = new Layer[1                + _nHiddens + 1];
//        Weights = new Matrix[Layers.Length - 1];

        //create layers
//        Layers[0]._nInputs;
//        Layers[^1] = _nOutputs;
//        for (int i = 1; i < Layers.Length - 1; i++) Layers[i] = _nHiddenNodes;
        //create weights+layers
        for (int m = 1; m < Layers.Length-1; m++) Layers[m].Weights = new Matrix(Layers[m-1].Columns, nHiddens);
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
        Layers[0].ActivatedWeightedInputs = new Matrix(_inputData);
        Layers[^1].Weights                = new Matrix(Layers[^2].Columns,outputData.Columns);
        InitWeights();
    }

    public void InitWeights(Random value)
    { 
        foreach (Layer layer in Layers)
            layer.Weights.Fill(value.NextDouble);
    }
//    private void InitWeights(double value) => Weights = Weights.Select(x => x.Fill(value)).ToArray();

    private void InitWeights()
    {
        NormalDist normalDist = new();
        InitWeights(normalDist);
        Xavierize();
    }

    private void Xavierize()
    {
        foreach (var layer in Layers)
        {
           layer.Weights/=layer
        }
        Layers(Layers.Length, (a, b) => a / b).ToArray();
    }

    public  Function CostFunction = x => x * x * 0.5;
    public  Function CostFunctionPrime = x => x;
    private double   TanH(double       x) => Math.Tanh(x);
    private double   TanH_Prime(double x) => Math.Pow(Math.Tanh(x), 2);

    public  Matrix Error()                   => (_outputData - Predict()).Map(CostFunction);
    public  Matrix Predict()                 => ForwardProp();
    public  Matrix ForwardProp()             => ForwardProp(_inputData);
    private Matrix ForwardProp(Matrix first) => Weights.Aggregate(first, (prev, current) => (prev * current).Map(TanH));
    private Matrix BackProp(Matrix    last)  => Weights.Reverse().Aggregate(last, (prev, current) => prev);

    public override string ToString() => Weights.Select(x => x.ToString()).Aggregate((a, b) => a + "\n" + b);
}

/// <summary>
///     Each Layer consists of a 'weights <see cref = "Matrix" />' that precede it and the layers activation function.
///     <returns>The activated weights</returns>
/// </summary>
/// <seealso cref = "Matrix" />
public class Layer : Matrix
{
    public Function ActivationFunction      { get; set; }
    public Function ActivationFunctionPrime { get; set; }
    public Function LossFunction            { get; set; }
    public Function LossFunctionPrime       { get; set; }
    public int      Nodes                   => Columns;
    public Matrix   Weights;
    public Matrix   WeightedInputs;
    public Matrix   ActivatedWeightedInputs;

    public Layer(Matrix matrix, Function activationFunction, Function activationFunctionPrime, Function lossFunction,
                 Function lossFunctionPrime) : base(matrix)
    {
        ActivationFunction      = activationFunction;
        ActivationFunctionPrime = activationFunction;
        LossFunction            = lossFunction;
        LossFunctionPrime       = lossFunctionPrime;
        Weights                 = matrix;
    }
    //Calling without a activation function defaults to identity function (returns the weights just the way they are); ie input data is a plain matrix
    //Can create a Layer with a matrix or directly with data (which gets converted to a matrix).
//    public Layer(Matrix     matrix) : base(matrix) => ActivationFunction = x => x;
//    public Layer(double[][] data) : base(data) => ActivationFunction = x => x;
//    public Layer(Func<double, double> activationFunction, double[][] data) : base(new Matrix(data)) =>
//        ActivationFunction = activationFunction;
//    public Layer(Func<double, double> activationFunction, Matrix matrix) : base(matrix) =>
//        ActivationFunction = activationFunction;

    // activate weightedInput and then multiply with w2, also change activation function from a to b.
    public static Layer operator *(Layer firstLayer, Layer secondLayer)
    {
        secondLayer.WeightedInputs          = firstLayer.ActivatedWeightedInputs *secondLayer.Weights;
        secondLayer.ActivatedWeightedInputs = secondLayer.ActivatedWeightedInputs.Map(secondLayer.ActivationFunction);
        return secondLayer;
    }
//        new Layer(l2.ActivationFunction, ElementWiseOperation(l1, l2, (w1, w2) => l1.ActivationFunction(w1) * w2));
}
