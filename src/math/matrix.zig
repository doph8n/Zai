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
        pub fn elementWiseAdd(self: @This(), scalar: T) !@This() {
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
