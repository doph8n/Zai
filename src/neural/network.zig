const std = @import("std");
const Layer = @import("layer.zig").Layer;

pub fn NeuralNetwork(comptime T: type) type {
    return struct {
        layers: std.Arraylist(*Layer(T)),
    };
}
