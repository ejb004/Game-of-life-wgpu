// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct InstanceInput {
    @location(2) position: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

struct Params {
    window_size: u32,
    cells: u32,
}

@group(0) @binding(0)
var<uniform> params: Params;

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;

    let zero_one = vec2<f32>(-1.5,1.5);

    // var cell_size = map_range(f32(params.window_size) / f32(params.cells), vec2<f32>(0.0,f32(params.window_size)), zero_one);
    let cell_size = 1.5 / f32(params.cells);

    let dist_correct = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    let inst_pos = vec4<f32>(map_range(instance.position.x, vec2<f32>(0.0, f32(params.cells)), zero_one), 
        map_range(instance.position.y, vec2<f32>(0.0, f32(params.cells)), zero_one), 0.0, 1.0);

    let c = instance.position[3];
    // out.color = model.color * c;
    out.color = vec3<f32>(c,c,c);
    out.clip_position = vec4<f32>(model.position * cell_size, 1.0) + inst_pos + dist_correct;
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}

fn map_range(value: f32, inRange: vec2<f32>, outRange: vec2<f32>) -> f32 {
    // Ensure the input value is within the input range
    let clampedValue = clamp(value, inRange.x, inRange.y);

    // Calculate the normalized value in the input range
    let normalizedValue = (clampedValue - inRange.x) / (inRange.y - inRange.x);

    // Map the normalized value to the output range
    let mappedValue = outRange.x + normalizedValue * (outRange.y - outRange.x);

    return mappedValue;
}