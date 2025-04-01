const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;
const Activation = @import("activation.zig").Activation;

pub fn DenseLayer(comptime T: type) type {
    return struct {
        input_size: usize,
        output_size: usize,
        weights: Matrix(T),
        biases: Matrix(T),
        activation: Activation(T),
        inputs: ?Matrix(T),
        z_values: ?Matrix(T),
        outputs: ?Matrix(T),
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize, activation: Activation(T)) !DenseLayer(T) {}
        pub fn deinit(self: *DenseLayer(T)) void {}
        pub fn clone(self: DenseLayer(T)) !DenseLayer(T) {}
        pub fn forward(self: *DenseLayer(T), inputs: Matrix(T), is_training: bool) !Matrix(T) {}
        pub fn backward(self: *DenseLayer(T), d_outputs: Matrix(T), learning_rate: T) !Matrix(T) {}
        pub fn calculateGradients(self: DenseLayer(T), d_outputs: Matrix(T)) !struct { d_weights: Matrix(T), d_biases: Matrix(T), d_inputs: Matrix(T) } {}
        pub fn initializeWeights(self: *DenseLayer(T), method: InitMethod) !void {}
        pub fn xavierInitialization(self: *DenseLayer(T)) !void {}
        pub fn heInitialization(self: *DenseLayer(T)) !void {}
        pub fn print(self: DenseLayer(T)) void {}
        pub fn reset(self: *DenseLayer(T)) !void {}
    };
}

pub fn BatchNormLayer(comptime T: type) type {
    return struct {
        gamma: Matrix(T),
        beta: Matrix(T),
        running_mean: Matrix(T),
        running_var: Matrix(T),
        epsilon: T,
        momentum: T,
        inputs: ?Matrix(T),
        normalized: ?Matrix(T),
        std_dev: ?Matrix(T),
        batch_mean: ?Matrix(T),
        batch_var: ?Matrix(T),
        is_training: bool,
        feature_size: usize,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, feature_size: usize, epsiolon: T, momentum: T) !BatchNormLayer(T) {}
        pub fn deinit(self: *BatchNormLayer(T)) void {}
        pub fn forward(self: *BatchNormLayer(T), inputs: Matrix(T), is_training: bool) !Matrix(T) {}
        pub fn backword(self: *BatchNormLayer(T), d_outputs: Matrix(T), learning_rate: T) !Matrix(T) {}
    };
}
