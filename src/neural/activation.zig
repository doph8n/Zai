const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;

pub fn Activation(comptime T: type) type {
    return struct {
        forward: *const fn (T) T,
        backward: *const fn (T) T,
        name: []const u8,
        pub fn applyForward(self: @This(), matrix: Matrix(T)) !Matrix(T) {
            return matrix.apply(self.forward);
        }
        pub fn applyBackward(self: @This(), matrix: Matrix(T)) !Matrix(T) {
            return matrix.apply(self.backward);
        }
    };
}

pub fn sigmoid(comptime T: type, x: T) T {
    return 1.0 / (1.0 + std.math.exp(-x));
}

pub fn sigmoidDerivative(comptime T: type, x: T) T {
    const s = sigmoid(T, x);
    return s * (1.0 - s);
}

pub fn createSigmoid(comptime T: type) Activation(T) {
    return Activation(T){
        .forward = struct {
            fn f(x: T) T {
                return sigmoid(T, x);
            }
        }.f,
        .backward = struct {
            fn f(x: T) T {
                return sigmoidDerivative(T, x);
            }
        }.f,
        .name = "sigmoid",
    };
}

// ReLU Activation
pub fn relu(comptime T: type, x: T) T {
    return if (x > 0) x else 0;
}
pub fn reluDerivative(comptime T: type, x: T) T {
    return if (x > 0) 1 else 0;
}
pub fn createRelu(comptime T: type) Activation(T) {
    return Activation(T){
        .forward = struct {
            fn f(x: T) T {
                return relu(T, x);
            }
        }.f,
        .backward = struct {
            fn f(x: T) T {
                return reluDerivative(T, x);
            }
        }.f,
        .name = "relu",
    };
}

// Leaky ReLU Activation
pub fn leakyRelu(comptime T: type, alpha: T, x: T) T {
    return if (x > 0) x else alpha * x;
}
pub fn leakyReluDerivative(comptime T: type, alpha: T, x: T) T {
    return if (x > 0) 1 else alpha;
}
pub fn createLeakyRelu(comptime T: type, alpha: T) Activation(T) {
    return Activation(T){
        .forward = struct {
            const a = alpha;
            fn f(x: T) T {
                return leakyRelu(T, a, x);
            }
        }.f,
        .backward = struct {
            const a = alpha;
            fn f(x: T) T {
                return leakyReluDerivative(T, a, x);
            }
        }.f,
        .name = "leaky_relu",
    };
}

// Tanh Activation
pub fn tanh(comptime T: type, x: T) T {
    return std.math.tanh(x);
}
pub fn tanhDerivative(comptime T: type, x: T) T {
    const t = std.math.tanh(x);
    return 1.0 - t * t;
}
pub fn createTanh(comptime T: type) Activation(T) {
    return Activation(T){
        .forward = struct {
            fn f(x: T) T {
                return tanh(T, x);
            }
        }.f,
        .backward = struct {
            fn f(x: T) T {
                return tanhDerivative(T, x);
            }
        }.f,
        .name = "tanh",
    };
}

// Linear Activation (Identity)
pub fn linear(comptime T: type, x: T) T {
    return x;
}
pub fn createLinear(comptime T: type) Activation(T) {
    return Activation(T){
        .forward = struct {
            fn f(x: T) T {
                return linear(T, x);
            }
        }.f,
        .backward = struct {
            fn f() T {
                return 1;
            }
        }.f,
        .name = "linear",
    };
}

// ELU (Exponential Linear Unit) Activation
pub fn elu(comptime T: type, alpha: T, x: T) T {
    return if (x > 0) x else alpha * (std.math.exp(x) - 1.0);
}
pub fn eluDerivative(comptime T: type, alpha: T, x: T) T {
    return if (x > 0) 1 else alpha * std.math.exp(x);
}
pub fn createElu(comptime T: type, alpha: T) Activation(T) {
    return Activation(T){
        .forward = struct {
            const a = alpha;
            fn f(x: T) T {
                return elu(T, a, x);
            }
        }.f,
        .backward = struct {
            const a = alpha;
            fn f(x: T) T {
                return eluDerivative(T, a, x);
            }
        }.f,
        .name = "elu",
    };
}

// SELU (Scaled Exponential Linear Unit) Activation
const selu_alpha: f32 = 1.6732632423543772848170429916717;
const selu_scale: f32 = 1.0507009873554804934193349852946;
pub fn selu(comptime T: type, x: T) T {
    if (x > 0) {
        return selu_scale * x;
    } else {
        return selu_scale * selu_alpha * (std.math.exp(x) - 1.0);
    }
}

/// Derivative of SELU
pub fn seluDerivative(comptime T: type, x: T) T {
    if (x > 0) {
        return selu_scale;
    } else {
        return selu_scale * selu_alpha * std.math.exp(x);
    }
}
pub fn createSelu(comptime T: type) Activation(T) {
    return Activation(T){
        .forward = struct {
            fn f(x: T) T {
                return selu(T, x);
            }
        }.f,
        .backward = struct {
            fn f(x: T) T {
                return seluDerivative(T, x);
            }
        }.f,
        .name = "selu",
    };
}

// Softmax (special case - operates on entire matrix)
pub fn softmaxForward(comptime T: type, matrix: Matrix(T)) !Matrix(T) {
    return @import("matrix.zig").softmax(T, matrix);
}
pub fn softmaxDerivative(comptime T: type, softmax_output: Matrix(T)) !Matrix(T) {
    return softmax_output.clone();
}

// Utility functions
pub const ActivationType = enum {
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
    Linear,
    ELU,
    SELU,
};

/// Get activation function by type
pub fn getActivation(comptime T: type, activation_type: ActivationType) Activation(T) {
    return switch (activation_type) {
        .Sigmoid => createSigmoid(T),
        .ReLU => createRelu(T),
        .LeakyReLU => createLeakyRelu(T, 0.01), // Default alpha = 0.01
        .Tanh => createTanh(T),
        .Linear => createLinear(T),
        .ELU => createElu(T, 1.0), // Default alpha = 1.0
        .SELU => createSelu(T),
    };
}

/// Get activation function by name
pub fn getActivationByName(comptime T: type, name: []const u8) ?Activation(T) {
    if (std.mem.eql(u8, name, "sigmoid")) {
        return createSigmoid(T);
    } else if (std.mem.eql(u8, name, "relu")) {
        return createRelu(T);
    } else if (std.mem.eql(u8, name, "leaky_relu")) {
        return createLeakyRelu(T, 0.01);
    } else if (std.mem.eql(u8, name, "tanh")) {
        return createTanh(T);
    } else if (std.mem.eql(u8, name, "linear")) {
        return createLinear(T);
    } else if (std.mem.eql(u8, name, "elu")) {
        return createElu(T, 1.0);
    } else if (std.mem.eql(u8, name, "selu")) {
        return createSelu(T);
    } else {
        return null;
    }
}

// Pre-defined common activation functions for convenience
pub const SIGMOID_F32 = createSigmoid(f32);
pub const RELU_F32 = createRelu(f32);
pub const TANH_F32 = createTanh(f32);
pub const LINEAR_F32 = createLinear(f32);
pub const LEAKY_RELU_F32 = createLeakyRelu(f32, 0.01);
pub const ELU_F32 = createElu(f32, 1.0);
pub const SELU_F32 = createSelu(f32);
