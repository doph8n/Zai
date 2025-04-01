const std = @import("std");

pub fn Matrix(comptime T: type) type {
    return struct {
        rows: usize,
        cols: usize,
        data: []T,
        allocator: std.mem.Allocator,

        // Basic Access & Management
        pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !@This() {
            const data = try allocator.alloc(T, rows * cols);
            @memset(data, 0);

            return @This(){
                .rows = rows,
                .cols = cols,
                .data = data,
                .allocator = allocator,
            };
        }
        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.data);
        }
        pub fn get(self: @This(), row: usize, col: usize) T {
            return self.data[row * self.cols + col];
        }
        pub fn set(self: *@This(), row: usize, col: usize, value: T) void {
            self.data[row * self.cols + col] = value;
        }
        pub fn clone(self: @This()) !@This() {
            const newMatrix = try @This().init(self.allocator, self.rows, self.cols);
            std.mem.copy(T, newMatrix.data, self.data);
            return newMatrix;
        }
        // Simple Operations
        pub fn fill(self: *@This(), value: T) void {
            for (self.data) |*item| {
                item.* = value;
            }
        }
        pub fn transpose(self: @This()) !@This() {
            var result = try @This().init(self.allocator, self.cols, self.rows);
            for (0..self.rows) |i| {
                for (0..self.cols) |j| {
                    result.set(j, i, self.get(i, j));
                }
            }
            return result;
        }
        pub fn scalarMultiply(self: @This(), scalar: T) !@This() {
            const result = try self.clone();
            for (result.data) |*item| {
                item.* *= scalar;
            }
            return result;
        }
        pub fn scalarElementWiseAdd(self: @This(), scalar: T) !@This() {
            const result = try self.clone();

            for (result.data) |*item| {
                item.* += scalar;
            }
            return result;
        }
        pub fn randomize(self: *@This(), min: T, max: T) !@This() {
            var prng = std.rand.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
            const random = prng.random();
            for (0..self.data.len) |i| {
                if (T == f32 or T == f64) {
                    self.data[i] = min + (max - min) * random.float(T);
                } else {
                    self.data[i] = random.intRangeAtMost(T, @intFromFloat(min), @intFromFloat(max));
                }
            }
        }
        pub fn zeros(self: *@This()) void {
            self.fill(0);
        }
        pub fn ones(self: *@This()) void {
            self.fill(1);
        }

        // Element-wise Operations
        pub fn apply(self: @This(), func: fn (T) T) !@This() {
            const result = try self.clone();
            for (result.data) |*item| {
                item.* = func(item.*);
            }
            return result;
        }
        pub fn sigmoid(x: T) T {
            return 1.0 / (1.0 + std.math.exp(-x));
        }
        pub fn dvSigmoid(x: T) T {
            const s = sigmoid(x);
            return s * (1 - s);
        }
        pub fn relu(x: T) T {
            return if (x < 0) 0 else x;
        }
        pub fn dvRelu(x: T) T {
            return if (x > 0) 1 else 0;
        }
        pub fn tanh(x: T) T {
            return (std.math.exp(x) - std.math.exp(-x)) / (std.math.exp(x) + std.math.exp(-x));
        }
        pub fn dvTanh(x: T) T {
            const t = tanh(x);
            return 1.0 - t * t;
        }

        // Debug Purpose
        pub fn print(self: @This()) void {
            std.debug.print("Matrix ({d}x{d}):\n", .{ self.rows, self.cols });
            var i: usize = 0;
            while (i < self.rows) : (i += 1) {
                var j: usize = 0;
                while (j < self.cols) : (j += 1) {
                    std.debug.print("{d} ", .{self.get(i, j)});
                }
                std.debug.print("\n", .{});
            }
        }
    };
}

pub fn matrixMultiply(comptime T: type, a: Matrix(T), b: Matrix(T)) !Matrix(T) {
    if (a.cols != b.rows) {
        return error.DimensionMismatch;
    }
    var result = try Matrix(T).init(a.allocator, a.rows, b.cols);
    for (0..a.rows) |i| {
        for (0..b.cols) |j| {
            var sum: T = 0;
            for (0..a.cols) |k| {
                sum += a.get(i, k) * b.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
}

// Matrix-Matrix Operations
pub fn elementWiseMultiply(comptime T: type, a: Matrix(T), b: Matrix(T)) !Matrix(T) {
    if (a.rows != b.rows or a.cols != b.cols) {
        return error.DimensionMismatch;
    }
    var result = try Matrix(T).init(a.allocator, a.rows, a.cols);
    for (0..a.data.len) |i| {
        result.data[i] = a.data[i] * b.data[i];
    }
    return result;
}
pub fn elementWiseAdd(comptime T: type, a: Matrix(T), b: Matrix(T)) !Matrix(T) {
    if (a.rows != b.rows or a.cols != b.cols) {
        return error.DimensionMismatch;
    }
    var result = try Matrix(T).init(a.allocator, a.rows, a.cols);
    for (0..a.data.len) |i| {
        result.data[i] = a.data[i] + b.data[i];
    }
}
pub fn elementWiseSub(comptime T: type, a: Matrix(T), b: Matrix(T)) !Matrix(T) {
    if (a.rows != b.rows or a.cols != b.cols) {
        return error.DimensionMismatch;
    }
    var result = try Matrix(T).init(a.allocator, a.rows, a.cols);
    for (0..a.data.len) |i| {
        result.data[i] = a.data[i] - b.data[i];
    }
}
pub fn concatenate(comptime T: type, a: Matrix(T), b: Matrix(T), axis: usize) !Matrix(T) {
    var result: Matrix(T) = undefined;
    if (axis == 0) {
        if (a.cols != b.cols) {
            return error.DimensionMismatch;
        }
        result = try Matrix(T).init(a.allocator, a.rows + b.rows, a.cols);
        var dest_index: usize = 0;
        const a_size = a.rows * a.cols;
        @memcpy(result.data[0..a_size], a.data);
        dest_index += a_size;
    } else if (axis == 1) {
        if (a.rows != b.rows) {
            return error.DimensionMismatch;
        }
        result = try Matrix(T).init(a.allocator, a.rows, a.cols + b.cols);
        for (0..a.rows) |i| {
            const row_offset = i * result.cols;
            @memcpy(result.data[row_offset..(row_offset + a.cols)], a.data[(i * a.cols)..((i + 1) * a.cols)]);
            @memcpy(result.data[(row_offset + a.cols)..(row_offset + a.cols + b.cols)], b.data[(i * b.cols)..((i + 1) * b.cols)]);
        }
    } else {
        return error.InvalidAxis;
    }
    return result;
}

// Neural Network Specific
pub fn softmax(comptime T: type, a: Matrix(T)) !Matrix(T) {
    var result = try Matrix(T).init(a.allocator, a.rows, a.cols);
    for (0..a.rows) |i| {
        var max_val: T = std.math.neginf(T);
        for (0..a.cols) |j| {
            max_val = @max(max_val, a.get(i, j));
        }
        var sum: T = 0;
        for (0..a.cols) |j| {
            const exp_val = std.math.exp(a.get(i, j) - max_val);
            result.set(i, j, exp_val);
            sum += exp_val;
        }
        for (0..a.cols) |j| {
            result.set(i, j, result.get(i, j) / sum);
        }
    }
    return result;
}
pub fn crossEntropyLoss(comptime T: type, predictions: Matrix(T), targets: Matrix(T)) !T {
    if (predictions.rows != targets.rows or predictions.cols != targets.cols) {
        return error.DimensionMismatch;
    }
    var loss: T = 0;
    const epsilon: T = 1e-15;
    for (0..predictions.rows) |i| {
        for (0..predictions.cols) |j| {
            var pred = predictions.get(i, j);
            pred = @max(epsilon, @min(1.0 - epsilon, pred));
            loss -= targets.get(i, j) * std.math.log(pred);
        }
    }
    return loss / @as(T, @floatFromInt(predictions.rows));
}
pub fn meanSquaredError(comptime T: type, predictions: Matrix(T), targets: Matrix(T)) !T {
    if (targets.len != predictions.len) {
        return error.LengthMismatch;
    }
    if (predictions.len == 0) {
        return error.EmptyInput;
    }
    var sum: T = 0;
    for (predictions, targets) |pred, tar| {
        const diff = pred - tar;
        sum += diff * diff;
    }
    return sum / @as(T, @floatFromInt(predictions.len));
}
pub fn batchNormalize(comptime T: type, matrix: Matrix(T), epsilon: T) !Matrix(T) {
    var result = try matrix.clone();
    const batch_size = matrix.rows;
    if (batch_size <= 1) {
        return result;
    }
    for (0..matrix.cols) |j| {
        var mean: T = 0;
        for (0..batch_size) |i| {
            mean += matrix.get(i, j);
        }
        mean /= @as(T, @floatFromInt(batch_size));
        var variance: T = 0;
        for (0..batch_size) |i| {
            const diff = matrix.get(i, j) - mean;
            variance += diff * diff;
        }
        variance /= @as(T, @floatFromInt(batch_size));
        for (0..batch_size) |i| {
            const normalized = (matrix.get(i, j) - mean) / std.math.sqrt(variance + epsilon);
            result.set(i, j, normalized);
        }
    }
    return result;
}

// Advanced Operations
pub fn outerProduct(comptime T: type, a: Matrix(T), b: Matrix(T)) !Matrix(T) {
    if (a.cols != 1 or b.cols != 1) {
        return error.InvalidDimensions;
    }
    var result = try Matrix(T).init(a.allocator, a.rows, b.rows);
    for (0..a.rows) |i| {
        for (0..b.rows) |j| {
            result.set(i, j, a.get(i, 0) * b.get(j, 0));
        }
    }
    return result;
}
pub fn sumAxis(comptime T: type, matrix: Matrix(T), axis: usize) !Matrix(T) {
    var result: Matrix(T) = undefined;
    if (axis == 0) {
        result = try Matrix(T).init(matrix.allocator, 1, matrix.cols);
        for (0..matrix.cols) |j| {
            var sum: T = 0;
            for (0..matrix.rows) |i| {
                sum += matrix.get(i, j);
            }
            result.set(0, j, sum);
        }
    } else if (axis == 1) {
        result = try Matrix(T).init(matrix.allocator, matrix.rows, 1);
        for (0..matrix.rows) |i| {
            var sum: T = 0;
            for (0..matrix.cols) |j| {
                sum += matrix.get(i, j);
            }
            result.set(i, 0, sum);
        }
    } else {
        return error.InvalidAxis;
    }
    return result;
}
pub fn meanAxis(comptime T: type, matrix: Matrix(T), axis: usize) !Matrix(T) {
    var result = try sumAxis(T, matrix, axis);
    if (axis == 0) {
        const scale: T = 1.0 / @as(T, @floatFromInt(matrix.rows));
        for (0..result.cols) |j| {
            result.set(0, j, result.get(0, j) * scale);
        }
    } else if (axis == 1) {
        const scale: T = 1.0 / @as(T, @floatFromInt(matrix.cols));
        for (0..result.rows) |i| {
            result.set(i, 0, result.get(i, 0) * scale);
        }
    }
    return result;
}
pub fn argmax(comptime T: type, matrix: Matrix(T), axis: usize) !Matrix(T) {
    var result: Matrix(T) = undefined;
    if (axis == 0) {
        result = try Matrix(T).init(matrix.allocator, 1, matrix.cols);
        for (0..matrix.cols) |j| {
            var max_val: T = std.math.neginf(T);
            var max_idx: T = 0;
            for (0..matrix.rows) |i| {
                const val = matrix.get(i, j);
                if (val > max_val) {
                    max_val = val;
                    max_idx = @floatFromInt(i);
                }
            }
            result.set(0, j, max_idx);
        }
    } else if (axis == 1) {
        result = try Matrix(T).init(matrix.allocator, matrix.rows, 1);
        for (0..matrix.rows) |i| {
            var max_val: T = std.math.neginf(T);
            var max_idx: T = 0;
            for (0..matrix.cols) |j| {
                const val = matrix.get(i, j);
                if (val > max_val) {
                    max_val = val;
                    max_idx = @floatFromInt(j);
                }
            }
            result.set(i, 0, max_idx);
        }
    } else {
        return error.InvalidAxis;
    }
    return result;
}

// Utility Functions
pub fn createFromArray(comptime T: type, allocator: std.mem.Allocator, array: []const T, rows: usize, cols: usize) !Matrix(T) {
    if (array.len != rows * cols) {
        return error.DimensionMismatch;
    }
    const matrix = try Matrix(T).init(allocator, rows, cols);
    std.mem.copy(T, matrix.data, array);
    return matrix;
}
pub fn identity(comptime T: type, allocator: std.mem.Allocator, size: usize) !Matrix(T) {
    var matrix = try Matrix(T).init(allocator, size, size);
    for (0..size) |i| {
        matrix.set(i, i, 1);
    }
    return matrix;
}
pub fn compareMatrices(comptime T: type, a: Matrix(T), b: Matrix(T), epsilon: T) bool {
    if (a.rows != b.rows or a.cols != b.cols) {
        return false;
    }
    for (0..a.data.len) |i| {
        if (std.math.fabs(a.data[i] - b.data[i]) > epsilon) {
            return false;
        }
    }
    return true;
}
