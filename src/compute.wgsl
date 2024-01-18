struct Params {
    window_size: u32,
    cells: u32,
}

struct InstanceInput {
    @location(2) position: vec4<f32>,
}

fn getIndex(x: u32, y: u32) -> u32 {
    let h = params.cells;
    let w = h;

    return (y % h) * w + (x % w);
}

fn countNeighbors(x: u32, y: u32) -> u32 {
  return u32(getCell(x - 1u, y - 1u) + getCell(x, y - 1u) + getCell(x + 1u, y - 1u) + 
         getCell(x - 1u, y) +                         getCell(x + 1u, y) + 
         getCell(x - 1u, y + 1u) + getCell(x, y + 1u) + getCell(x + 1u, y + 1u));
}

fn getCell(x: u32, y: u32) -> f32 {
  return current[getIndex(x, y)].position[3];
}

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read_write> current: array<InstanceInput>;

@group(0) @binding(2)
var<storage, read_write> next: array<InstanceInput>;

@compute @workgroup_size(8,8,4)
fn cs_main(@builtin(global_invocation_id) grid: vec3<u32>) {
    let x = grid.x;
    let y = grid.y; 
    let w = params.cells;
    let h = w;

    let n = countNeighbors(x, y);

    next[getIndex(x, y)].position[3] = select(f32(n == 3u), f32(n == 2u || n == 3u), f32(getCell(x, y)) == 1.0);
 
}