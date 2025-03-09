# Matrix Operations for Neural Networks

## Methods on Matrix Struct

1. **Basic Access & Management**
   - `init(allocator, rows, cols)` - Create a new matrix
   - `deinit()` - Free matrix memory
   - `get(row, col)` - Get element value
   - `set(row, col, value)` - Set element value
   - `clone()` - Create a deep copy of the matrix

2. **Simple Operations**
   - `fill(value)` - Fill entire matrix with a value
   - `transpose()` - Transpose the matrix
   - `scalarMultiply(scalar)` - Multiply all elements by a scalar
   - `elementWiseAdd(scalar)` - Add scalar to all elements
   - `randomize(min, max)` - Fill with random values in range
   - `zeros()` - Fill with zeros
   - `ones()` - Fill with ones

3. **Element-wise Operations**
   - `apply(function)` - Apply a function to each element
   - `sigmoid()` - Apply sigmoid activation to each element
   - `relu()` - Apply ReLU activation to each element
   - `tanh()` - Apply tanh activation to each element
   - `derivativeSigmoid()` - Apply sigmoid derivative
   - `derivativeRelu()` - Apply ReLU derivative
   - `derivativeTanh()` - Apply tanh derivative

## Independent Functions

1. **Matrix-Matrix Operations**
   - `matrixMultiply(a, b)` - Matrix multiplication (most critical operation!)
   - `elementWiseMultiply(a, b)` - Hadamard product (element-wise multiplication)
   - `elementWiseAdd(a, b)` - Element-wise addition
   - `elementWiseSubtract(a, b)` - Element-wise subtraction
   - `concatenate(a, b, axis)` - Concatenate matrices along an axis

2. **Neural Network Specific**
   - `softmax(matrix)` - Apply softmax function (column-wise)
   - `crossEntropyLoss(predictions, targets)` - Calculate loss
   - `meanSquaredError(predictions, targets)` - Calculate MSE loss
   - `batchNormalize(matrix)` - Apply batch normalization

3. **Advanced Operations**
   - `outerProduct(a, b)` - Compute outer product of vectors
   - `sumAxis(matrix, axis)` - Sum along an axis
   - `meanAxis(matrix, axis)` - Mean along an axis
   - `argmax(matrix, axis)` - Index of maximum value along axis

4. **Utility Functions**
   - `createFromArray(allocator, array, rows, cols)` - Create matrix from array
   - `identity(allocator, size)` - Create identity matrix
   - `compareMatrices(a, b, epsilon)` - Compare with tolerance for floating point
