using static ML.FunctionalMethods2D;

namespace ML;

/// <summary>
///     <para>
///         Class <c>Matrix</c>s supports addition, division, dot product, scaler multiplication, and subtraction. (all
///         values converted to double)
///     </para>
///     <example>
///         <code>Matrix (4,3)</code> Constructs a matrix with 4 rows and 3 columns
///         <code>Matrix.Fill(x)</code> Fills matrix with x (value or number generator)
///         <code>Ones</code> Fills matrix with 1
///     </example>
/// </summary>
public class Matrix
{
    private double[][] _matrix;
    public  int        Rows    { get; }
    public  int        Columns { get; }
    public  (int, int) Size    => (Rows, Columns); //convenient for the user

    //construct using rows and columns OR an Matrix OR array or arrays 
    public Matrix(double[][] data) => (Rows, Columns, _matrix) = (data.Length, data[0].Length, data);
    public Matrix(Matrix     matrix) => (Rows, Columns, _matrix) = (matrix.Rows, matrix.Columns, matrix._matrix);

    //syntax gymnastics -> C# wont let us define an array of arrays as [Rows][Columns]
    public Matrix(int r, int c) =>
        (Rows, Columns, _matrix) = (Rows = r    < 1 ? 1 : r,
                                    Columns = c < 1 ? 1 : c,
                                    _matrix = new double[Rows][]
                                             .Select(_ => new double[Columns]).ToArray());

    //operators * / - + and dot product (rolled into *)  
    public static Matrix operator -(Matrix a,    Matrix b)    => ElementWiseOperation(a, b,    (x, y) => x - y);
    public static Matrix operator +(Matrix a,    Matrix b)    => ElementWiseOperation(a, b,    (x, y) => x + y);
    public static Matrix operator *(Matrix a,    double elem) => ElementWiseOperation(a, elem, (x, y) => x * y);
    public static Matrix operator *(double elem, Matrix b)    => b * elem;
    public static Matrix operator /(Matrix a,    double elem) => ElementWiseOperation(a, elem, (x, y) => x / y);
    public static Matrix operator *(Matrix a,    Matrix b)    => DotProduct(a, b);

    //--Matrix Initializers-- 
    //you can fill with ones, other value, or even fill with your own function (like random)
    public Matrix Ones()                       => Fill(1);
    public Matrix Fill(double       value)     => ElementWiseOperation(() => value);
    public Matrix Fill(Func<double> operation) => ElementWiseOperation(operation);

    //--matrix math magic--
    // can perform element-wise operations using 2-dimensional map or zip
    public  Matrix Map(Func<double, double>          operation) => new Matrix(Map2D(_matrix, operation));
    private Matrix ElementWiseOperation(Func<double> operation) => new Matrix(Map2D(_matrix, operation));

    internal static Matrix ElementWiseOperation(Matrix a, Matrix b, Func<double, double, double> operation) =>
        a.Size != b.Size
            ? throw new Exception("can not combine matrices of different sizes")
            : new Matrix(ZipWith2D(a._matrix, b._matrix, operation));

    private static Matrix ElementWiseOperation(Matrix a, double num, Func<double, double, double> operation) =>
        new(Map2D(a._matrix, num, operation));

    // returns transposed copy of matrix 
    public Matrix Transpose()
    {
        Matrix result = new(Columns, Rows);
        for (int row = 0; row < Rows; row++)
            for (int column = 0; column < Columns; column++)
                result._matrix[column][row] = _matrix[row][column];

        return result;
    }

    /// <summary>
    ///     Method <c>DotProduct</c> returns a matrix product; also know as a matrix of dot products for each row x in first
    ///     matrix and the corresponding column x in matrix2
    ///     <remarks>
    ///         only 3-deep loop because we can use r2 as the index for c1; by definition columns in matrix 1 = rows in
    ///         matrix2
    ///     </remarks>
    ///     <example><code>result._matrix[r1][c2]</code> The dot product of row x in first matrix and column x in second matrix</example>
    /// </summary>
    private static Matrix DotProduct(Matrix matrix1, Matrix matrix2)
    {
        if (matrix1.Columns != matrix2.Rows) throw new Exception("columns and rows do not match up");
        Matrix result = new(matrix1.Rows, matrix2.Columns);
        for (int r1 = 0; r1 < matrix1.Rows; r1++)
            for (int c2 = 0; c2 < matrix2.Columns; c2++)
                for (int r2 = 0; r2 < matrix2.Rows; r2++)
                {
                    double elemProduct =
                        matrix1._matrix[r1][r2] *
                        matrix2._matrix[r2][c2];
                    result._matrix[r1][c2] += elemProduct;
                }

        return result;
    }

    // converts array into string representation 
    private string StringifyData(string array = "")
    {
        for (int row = 0; row < Rows; row++)
            for (int column = 0; column < Columns; column++)
            {
                double elem            = _matrix[row][column];
                if (column == 0) array += "\n[";
                array += column != Columns - 1 ? $"{elem}, " : $"{elem}]";
            }

        return array;
    }

    public override string ToString() => $"matrix {Rows}x{Columns} {StringifyData()}";
}