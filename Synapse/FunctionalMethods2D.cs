namespace ML;

/// <summary>
///     Class <c>FunctionalMethods2D</c> contains 2D versions of classic functional methods Zip and Map, Zip2D and Map2D
/// </summary>
public static class FunctionalMethods2D
{
    ///<summary>Method <c>ZipWith2D</c> combines two 2DArrays of type T using a function that can combine two T values</summary>
    public static T[][] ZipWith2D<T>(IEnumerable<T[]> lvl1A, IEnumerable<T[]> lvl1B, Func<T, T, T> operation)
        => lvl1A.Zip(lvl1B, (lvl2A, lvl2B) => lvl2A.Zip(lvl2B, operation).ToArray()).ToArray();

    ///<summary>Method <c>Map2D</c> Combines a T value with 2DArray of type T using a function that can combine two T values</summary>
    public static T[][] Map2D<T>(IEnumerable<T[]> lvl1, T value, Func<T, T, T> operation)
        => lvl1.Select(l2 => l2.Select(elem => operation(elem, value)).ToArray()).ToArray();

    ///<summary>Method <c>Map2D</c> Applies function to each T value in 2DArray of type T</summary>
    public static T[][] Map2D<T>(IEnumerable<T[]> lvl1, Func<T, T> operation) =>
        lvl1.Select(lvl2 => lvl2.Select(operation).ToArray()).ToArray();

    ///<summary>Method <c>Map2D</c> Replaces each T value in 2DArray of Type T with the result of the function</summary>
    public static T[][] Map2D<T>(IEnumerable<T[]> lvl1, Func<T> operation) =>
        lvl1.Select(lvl2 => lvl2.Select(_ => operation()).ToArray()).ToArray();
}