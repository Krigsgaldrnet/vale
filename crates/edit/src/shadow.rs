//! Recomputing a tile's `MCSH` shadow maps from what stands on it.
//!
//! ## What `MCSH` is
//!
//! The 1.12.1 client casts no shadows at run time. The ground's shadows are a
//! 64x64 bitmap per chunk, baked offline, on the same texel grid as an alpha
//! map, and drawn by darkening the sun term where a bit is set.
//! `vale_assets::world::adt` reads the bits. An edit that moves ground or
//! places a model leaves the old shadow in place until the tile is rebaked.
//!
//! ## What the shipped maps were cast from
//!
//! `vale bake <Map> <x> <y> shadow` rebakes a tile under several occluders
//! and scores each against the shipped map. The occluder that reproduces the
//! shipped maps is:
//!
//! * each placed doodad's drawn triangles, with the triangles of every
//!   non-opaque batch tested against the alpha of their texture
//!   (`vale_assets::world::collision::shadow_casters_near`);
//! * each placed building's collision hull. Its drawn triangles score the
//!   same on the tiles measured and are ten times as many;
//! * no terrain. Adding [`Ground`] lowers the score on every tile measured;
//! * the tile and its eight neighbours, cut to [`casting_box`], since a
//!   shadow crosses tile seams;
//! * the sun at [`vale_assets::tables::light::SHADOW_SUN`], and a ray length
//!   of [`REACH`].
//!
//! ```text
//!                                   texels agreeing   shipped shadow found
//! Azeroth_32_49                          97.7%              99.8%
//! Azeroth_34_51                          96.1%              99.5%
//! Azeroth_31_49                          97.5%              99.5%
//! Kalimdor_32_26                         98.9%             100.0%
//! ```
//!
//! Casting from collision hulls instead, as this module did before, finds
//! 20.5% of the shipped shadow on `Azeroth_32_49`: a tree's hull is its trunk
//! and a rough block for the crown.
//!
//! What the rebake adds that the shipped map lacks is mostly a one- or
//! two-texel fringe of leaf detail and the shadows of small doodads in open
//! ground. What it misses is mostly chunks with holes, where there is no
//! ground to sample.
//!
//! ## Where the geometry comes from
//!
//! This crate opens no archives. The caller supplies an [`Occluder`], which
//! answers one question: is there anything between this point and the sun?
//! [`Casters`] is the occluder for a set of triangles; the caller fills it
//! from the archives.

use crate::adt::file::{AdtFile, Region, SubChunk};
use crate::adt::header::at;
use crate::adt::heights;
use vale_assets::world::collision::{AlphaMask, Cutout, ShadowCasters};
use vale_assets::world::adt::{
    decode_shadow_map, encode_shadow_map, mcnk_flags, ALPHA_LEN, ALPHA_SIDE, CHUNKS_PER_SIDE, CHUNK_SIZE,
};

/// How far along a ray to look for something in the way, in yards.
///
/// 800 yards along a ray at [`vale_assets::tables::light::SHADOW_SUN`]'s
/// 46.7° is 549 yards across the ground, which is past the far edge of the
/// neighbouring tiles, so every caster the bake is given is within reach.
/// At 200 yards, `Kalimdor_32_26` misses 7% of its shipped shadow: the shadows
/// of Ashenvale's tallest trees are longer than that.
pub const REACH: f32 = 800.0;

/// The texture alpha at or above which a cut-out triangle casts.
///
/// The score against the shipped maps changes by under 0.5% between 1 and
/// 240, so this is the midpoint.
pub const ALPHA_TEST: u8 = 128;

/// How far above the ground a ray starts.
///
/// Without it, a bake whose occluder includes the terrain has every ray leave
/// from a point on a triangle of the terrain and immediately hit it, and the
/// whole tile comes out in shadow. A tenth of a yard is far below what a texel
/// resolves (a texel is 0.52 yards) and far above the float error in a ground
/// height.
pub const LIFT: f32 = 0.1;

/// The box a tile's shadows are cast from: the tile's own square widened by
/// `reach` on every side, at every height.
///
/// A ray leaves a texel of the tile and travels `reach` at most, so nothing
/// wholly outside this box can block one. A caller passes it to
/// `vale_assets::world::collision::shadow_casters_near` to keep the
/// neighbouring tiles' placements to the ones that can matter. The neighbours
/// are needed: a building is placed by the tile its origin is on, and its
/// walls can stand across the seam.
pub fn casting_box(tile_x: u32, tile_y: u32, reach: f32) -> [[f32; 3]; 2] {
    use vale_assets::world::adt::TILE_SIZE;
    let centre = vale_assets::world::terrain::tile_centre(tile_x, tile_y);
    let half = TILE_SIZE / 2.0 + reach;
    [
        [centre[0] - half, centre[1] - half, f32::MIN],
        [centre[0] + half, centre[1] + half, f32::MAX],
    ]
}

/// Whether anything is between a point and the sun.
///
/// `from` is a point a little above the ground, `toward` is a unit vector at
/// the sun, and `max` is how far to look.
pub trait Occluder {
    fn blocked(&self, from: [f32; 3], toward: [f32; 3], max: f32) -> bool;
}

/// An occluder with nothing in it, which bakes an unshadowed tile.
///
/// Baking with it clears a tile's shadows. A caller that has no geometry yet
/// passes it rather than skipping the bake and leaving the old bits.
pub struct Nothing;

impl Occluder for Nothing {
    fn blocked(&self, _from: [f32; 3], _toward: [f32; 3], _max: f32) -> bool {
        false
    }
}

impl<F> Occluder for F
where
    F: Fn([f32; 3], [f32; 3], f32) -> bool,
{
    fn blocked(&self, from: [f32; 3], toward: [f32; 3], max: f32) -> bool {
        self(from, toward, max)
    }
}

/// The tile's own ground as an occluder: terrain shading terrain.
///
/// The shipped `MCSH` maps do not contain terrain self-shadow. With the
/// casters described in the module header, adding this occluder lowers the
/// share of texels that agree with the shipped map on every tile measured:
///
/// ```text
///                     casters only   casters and ground
/// Azeroth_32_49           97.7%            96.3%
/// Azeroth_34_51           96.1%            95.1%
/// Kalimdor_32_26          89.1%            89.1%   (at a 200-yard reach)
/// ```
///
/// The client darkens a slope by its `MCNR` normal against the sun, so baking
/// terrain shadow into `MCSH` as well would darken it twice. The editor leaves
/// this out by default and offers it as an option for maps made for this
/// client.
///
/// ## How it is traced
///
/// The ground is a height field, so a ray is marched: each step along the ray
/// compares the ray's height with the ground's. The ground is sampled once
/// onto a regular grid at the alpha map's resolution, 1024 samples across a
/// tile, a little over half a yard apart, which is 1M height lookups and about
/// a tenth of a second.
///
/// ## Limits
///
/// Only the tile's own ground is sampled. A cliff on a neighbouring tile casts
/// nothing on this one, so a bake with this occluder is wrong in a band along
/// any edge with high ground beyond it. Sampling the nine tiles into one field
/// would fix it.
pub struct Ground {
    /// Samples across the tile, in both axes.
    side: usize,
    /// The tile's maximum corner, which is where sample `(0, 0)` sits.
    origin: [f32; 2],
    /// Yards between samples.
    step: f32,
    /// `side * side` heights, row-major: rows run along decreasing world x and
    /// columns along decreasing world y, the same frame as everything else here.
    z: Vec<f32>,
}

impl Ground {
    /// Sample a tile's ground onto a regular grid.
    ///
    /// Taken before the bake because [`bake`] holds the tile mutably and this
    /// needs it shared.
    pub fn of(tile: &AdtFile) -> Ground {
        let side = CHUNKS_PER_SIDE * ALPHA_SIDE;
        let step = CHUNK_SIZE / ALPHA_SIDE as f32;
        let mut z = vec![f32::MIN; side * side];
        let mut origin = [f32::MIN, f32::MIN];

        for (index, chunk) in tile.chunks.iter().enumerate().take(CHUNKS_PER_SIDE * CHUNKS_PER_SIDE)
        {
            let at = chunk.head().position();
            if index == 0 {
                origin = [at[0], at[1]];
            }
            let ground = heights::ground(chunk);
            let (cx, cy) = (index % CHUNKS_PER_SIDE, index / CHUNKS_PER_SIDE);
            for ty in 0..ALPHA_SIDE {
                for tx in 0..ALPHA_SIDE {
                    let x = at[0] - (ty as f32 + 0.5) * step;
                    let y = at[1] - (tx as f32 + 0.5) * step;
                    if let Some(h) = ground.height_at(x, y) {
                        z[(cy * ALPHA_SIDE + ty) * side + cx * ALPHA_SIDE + tx] = h;
                    }
                }
            }
        }
        Ground {
            side,
            origin,
            step,
            z,
        }
    }

    /// The ground at a world position, or `None` off the tile.
    fn at(&self, x: f32, y: f32) -> Option<f32> {
        let row = ((self.origin[0] - x) / self.step).floor();
        let col = ((self.origin[1] - y) / self.step).floor();
        if row < 0.0 || col < 0.0 {
            return None;
        }
        let (row, col) = (row as usize, col as usize);
        if row >= self.side || col >= self.side {
            return None;
        }
        match self.z[row * self.side + col] {
            v if v == f32::MIN => None,
            v => Some(v),
        }
    }
}

impl Occluder for Ground {
    fn blocked(&self, from: [f32; 3], toward: [f32; 3], max: f32) -> bool {
        // A level or descending ray is not tested: the sun is above the
        // horizon, and such a ray would march without climbing.
        if toward[2] <= 0.0 {
            return false;
        }
        let mut travelled = self.step;
        while travelled <= max {
            let x = from[0] + toward[0] * travelled;
            let y = from[1] + toward[1] * travelled;
            let z = from[2] + toward[2] * travelled;
            match self.at(x, y) {
                // Off the tile: no ground is known beyond it.
                None => return false,
                Some(ground) if ground > z => return true,
                Some(_) => {}
            }
            travelled += self.step;
        }
        false
    }
}

/// World-space triangles that cast a shadow, bucketed on a grid over x and y.
///
/// A ray walks the grid cells its x and y cross, nearest first, and stops at
/// the first triangle it meets. A cell is skipped when the ray is already above
/// the highest triangle in it, and the walk stops when the ray is above every
/// triangle. A tile with its neighbours' trees is a few million triangles,
/// which this answers in about a second per million rays on one core.
pub struct Casters {
    triangles: Vec<[[f32; 3]; 3]>,
    /// Parallel to `triangles`; see `vale_assets::world::collision::Cutout`.
    cutouts: Vec<Option<Cutout>>,
    masks: Vec<AlphaMask>,
    /// The alpha at or above which a cut-out triangle casts.
    threshold: u8,
    /// The grid's minimum corner in x and y.
    min: [f32; 2],
    cols: usize,
    rows: usize,
    /// `starts[c]..starts[c + 1]` indexes `items` for cell `c = row * cols + col`.
    starts: Vec<u32>,
    /// Triangle indices, bucketed by cell.
    items: Vec<u32>,
    /// The highest z of any triangle in each cell, or `f32::MIN` for an empty one.
    top: Vec<f32>,
    /// The highest z of any triangle.
    highest: f32,
}

impl Casters {
    /// The side of a grid cell, in yards.
    pub const CELL: f32 = 2.0;

    /// Bucket `casters`. A cut-out triangle casts where its texel's alpha is
    /// at least `threshold`.
    pub fn new(casters: ShadowCasters, threshold: u8) -> Casters {
        let ShadowCasters {
            triangles,
            mut cutouts,
            masks,
        } = casters;
        cutouts.resize(triangles.len(), None);
        let mut lo = [f32::MAX; 2];
        let mut hi = [f32::MIN; 2];
        let mut highest = f32::MIN;
        for t in &triangles {
            for v in t {
                for axis in 0..2 {
                    lo[axis] = lo[axis].min(v[axis]);
                    hi[axis] = hi[axis].max(v[axis]);
                }
                highest = highest.max(v[2]);
            }
        }
        if triangles.is_empty() {
            lo = [0.0; 2];
            hi = [0.0; 2];
        }
        let cols = (((hi[0] - lo[0]) / Self::CELL).floor() as usize + 1).max(1);
        let rows = (((hi[1] - lo[1]) / Self::CELL).floor() as usize + 1).max(1);
        let span = |t: &[[f32; 3]; 3]| {
            let (mut a, mut b) = ([f32::MAX; 2], [f32::MIN; 2]);
            for v in t {
                for axis in 0..2 {
                    a[axis] = a[axis].min(v[axis]);
                    b[axis] = b[axis].max(v[axis]);
                }
            }
            let cell = |v: f32, axis: usize, n: usize| (((v - lo[axis]) / Self::CELL).floor().max(0.0) as usize).min(n - 1);
            (cell(a[0], 0, cols), cell(b[0], 0, cols), cell(a[1], 1, rows), cell(b[1], 1, rows))
        };

        let mut counts = vec![0u32; cols * rows + 1];
        for t in &triangles {
            let (c0, c1, r0, r1) = span(t);
            for r in r0..=r1 {
                for c in c0..=c1 {
                    counts[r * cols + c] += 1;
                }
            }
        }
        let mut starts = vec![0u32; cols * rows + 1];
        for c in 0..cols * rows {
            starts[c + 1] = starts[c] + counts[c];
        }
        let mut fill = starts.clone();
        let mut items = vec![0u32; starts[cols * rows] as usize];
        let mut top = vec![f32::MIN; cols * rows];
        for (index, t) in triangles.iter().enumerate() {
            let z = t[0][2].max(t[1][2]).max(t[2][2]);
            let (c0, c1, r0, r1) = span(t);
            for r in r0..=r1 {
                for c in c0..=c1 {
                    let cell = r * cols + c;
                    items[fill[cell] as usize] = index as u32;
                    fill[cell] += 1;
                    top[cell] = top[cell].max(z);
                }
            }
        }
        Casters {
            triangles,
            cutouts,
            masks,
            threshold,
            min: lo,
            cols,
            rows,
            starts,
            items,
            top,
            highest,
        }
    }

    /// How many triangles cast.
    pub fn len(&self) -> usize {
        self.triangles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    /// Whether the ray meets any triangle in one cell between `0` and `max`.
    fn hits_in(&self, cell: usize, from: [f32; 3], toward: [f32; 3], max: f32) -> bool {
        let range = self.starts[cell] as usize..self.starts[cell + 1] as usize;
        self.items[range].iter().any(|&t| {
            let Some((d, u, v)) = ray_meets(from, toward, &self.triangles[t as usize]) else {
                return false;
            };
            if d > max {
                return false;
            }
            match &self.cutouts[t as usize] {
                None => true,
                Some(cut) => {
                    let w = 1.0 - u - v;
                    let uv = [
                        w * cut.uvs[0][0] + u * cut.uvs[1][0] + v * cut.uvs[2][0],
                        w * cut.uvs[0][1] + u * cut.uvs[1][1] + v * cut.uvs[2][1],
                    ];
                    self.masks
                        .get(cut.mask as usize)
                        .is_none_or(|m| m.at(uv) >= self.threshold)
                }
            }
        })
    }
}

impl Occluder for Casters {
    fn blocked(&self, from: [f32; 3], toward: [f32; 3], max: f32) -> bool {
        if self.triangles.is_empty() || from[2] > self.highest && toward[2] >= 0.0 {
            return false;
        }
        let size = [self.cols as f32 * Self::CELL, self.rows as f32 * Self::CELL];
        // Clip the ray to the grid's square.
        let (mut enter, mut leave) = (0.0f32, max);
        for axis in 0..2 {
            let (o, d) = (from[axis] - self.min[axis], toward[axis]);
            if d.abs() < 1.0e-9 {
                if o < 0.0 || o >= size[axis] {
                    return false;
                }
                continue;
            }
            let (a, b) = ((0.0 - o) / d, (size[axis] - o) / d);
            enter = enter.max(a.min(b));
            leave = leave.min(a.max(b));
        }
        if enter > leave {
            return false;
        }

        // Walk the cells from `enter` to `leave`.
        let at = |t: f32, axis: usize| from[axis] + toward[axis] * t - self.min[axis];
        let n = [self.cols, self.rows];
        let mut cell = [0usize; 2];
        let mut step = [0isize; 2];
        let mut next = [f32::MAX; 2];
        let mut delta = [f32::MAX; 2];
        for axis in 0..2 {
            let p = at(enter, axis);
            cell[axis] = ((p / Self::CELL).floor().max(0.0) as usize).min(n[axis] - 1);
            let d = toward[axis];
            if d > 1.0e-9 {
                step[axis] = 1;
                delta[axis] = Self::CELL / d;
                next[axis] = enter + ((cell[axis] + 1) as f32 * Self::CELL - p) / d;
            } else if d < -1.0e-9 {
                step[axis] = -1;
                delta[axis] = -Self::CELL / d;
                next[axis] = enter + (cell[axis] as f32 * Self::CELL - p) / d;
            }
        }
        let mut t = enter;
        loop {
            let exit = next[0].min(next[1]).min(leave);
            // The ray's lowest point inside this cell.
            let low = from[2] + toward[2] * if toward[2] >= 0.0 { t } else { exit };
            if toward[2] >= 0.0 && low > self.highest {
                return false;
            }
            let index = cell[1] * self.cols + cell[0];
            if low <= self.top[index] && self.hits_in(index, from, toward, max) {
                return true;
            }
            if exit >= leave {
                return false;
            }
            let axis = if next[0] < next[1] { 0 } else { 1 };
            let moved = cell[axis] as isize + step[axis];
            if moved < 0 || moved as usize >= n[axis] {
                return false;
            }
            cell[axis] = moved as usize;
            t = next[axis];
            next[axis] += delta[axis];
        }
    }
}

/// The distance along a ray to a triangle, if it meets it in front of its
/// start, and the barycentric weights of the second and third corners at the
/// hit (Möller–Trumbore).
fn ray_meets(origin: [f32; 3], dir: [f32; 3], t: &[[f32; 3]; 3]) -> Option<(f32, f32, f32)> {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    };
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let e1 = sub(t[1], t[0]);
    let e2 = sub(t[2], t[0]);
    let h = cross(dir, e2);
    let a = dot(e1, h);
    if a.abs() < 1.0e-12 {
        return None;
    }
    let f = 1.0 / a;
    let s = sub(origin, t[0]);
    let u = f * dot(s, h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(s, e1);
    let v = f * dot(dir, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let d = f * dot(e2, q);
    (d > 1.0e-4).then_some((d, u, v))
}

/// Two occluders as one: a ray is blocked when either blocks it.
pub struct Either<'a>(pub &'a (dyn Occluder + Sync), pub &'a (dyn Occluder + Sync));

impl Occluder for Either<'_> {
    fn blocked(&self, from: [f32; 3], toward: [f32; 3], max: f32) -> bool {
        self.0.blocked(from, toward, max) || self.1.blocked(from, toward, max)
    }
}

/// What a bake changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Baked {
    /// Chunks whose bits are not what they were.
    pub changed: usize,
    /// Chunks of the 256 with any texel in shadow.
    pub shadowed: usize,
}

/// Recompute one chunk's `MCSH`. Returns the 512 packed bytes.
///
/// The texel grid is the alpha map's: row `ty` runs along decreasing world x
/// and column `tx` along decreasing world y, from the chunk's origin, which is
/// its maximum corner in both. This is the layout `Adt::alpha_atlas` assumes,
/// the one `vale textures`' seam check pins, and the one `vale sun` reads
/// shadows through. A bake under any other layout would land rotated or
/// mirrored.
///
/// Texel 63 of each axis is computed but not read. The 1.12.1 client stretches
/// a 64x64 map so texels 0 and 62 sit on the chunk's edges, and
/// `decode_shadow_map` copies 62 into 63 when reading. The bake writes the
/// value the geometry gives for 63 anyway.
pub(crate) fn bake_chunk(
    tile: &AdtFile,
    chunk_index: usize,
    sun: [f32; 3],
    occluder: &dyn Occluder,
    reach: f32,
) -> Vec<u8> {
    let mut texels = vec![0u8; ALPHA_LEN];
    let Some(chunk) = tile.chunks.get(chunk_index) else {
        return encode_shadow_map(&texels);
    };
    let origin = chunk.head().position();
    let ground = heights::ground(chunk);
    let texel = CHUNK_SIZE / ALPHA_SIDE as f32;
    let sun = normalise(sun);

    for ty in 0..ALPHA_SIDE {
        for tx in 0..ALPHA_SIDE {
            // The texel's centre, in the chunk's own decreasing axes.
            let x = origin[0] - (ty as f32 + 0.5) * texel;
            let y = origin[1] - (tx as f32 + 0.5) * texel;
            let Some(z) = ground.height_at(x, y) else {
                continue;
            };
            let from = [x, y, z + LIFT];
            if occluder.blocked(from, sun, reach) {
                texels[ty * ALPHA_SIDE + tx] = 255;
            }
        }
    }
    encode_shadow_map(&texels)
}

/// Recompute every chunk's `MCSH` and set each chunk's `HAS_MCSH` flag to
/// match.
///
/// A chunk with no texel in shadow loses its `MCSH` region and the flag,
/// rather than keeping 64 bytes of zeroes. The two draw the same, the shipped
/// files leave unshadowed chunks without the region, and keeping zeroes would
/// make a tile larger after every bake. [`crate::adt::colours`] does the same
/// for `MCCV`.
///
/// A chunk that gains shadow gains the region and the flag. The reader keys
/// off the flag, so a region without it is not drawn.
pub fn bake(
    tile: &mut AdtFile,
    sun: [f32; 3],
    occluder: &(dyn Occluder + Sync),
    reach: f32,
) -> Baked {
    bake_watching(tile, sun, occluder, reach, &|_| {})
}

/// [`bake`], reporting progress.
///
/// The chunks are baked on one thread per core and written back in order
/// afterwards. `watch` is called once per finished chunk with the number
/// finished so far, 1 up to the chunk count, one call at a time.
/// `vale_ide::map::tiles` runs a bake on a task pool and shows the count on
/// the status line.
pub fn bake_watching(
    tile: &mut AdtFile,
    sun: [f32; 3],
    occluder: &(dyn Occluder + Sync),
    reach: f32,
    watch: &(dyn Fn(usize) + Sync),
) -> Baked {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    let count = tile.chunks.len().min(CHUNKS_PER_SIDE * CHUNKS_PER_SIDE);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(count.max(1));
    let next = AtomicUsize::new(0);
    let done = Mutex::new(0usize);
    let mut results: Vec<Option<Vec<u8>>> = vec![None; count];
    {
        let shared: &AdtFile = tile;
        let finished: Vec<Vec<(usize, Vec<u8>)>> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| {
                        let mut mine = Vec::new();
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            if index >= count {
                                break;
                            }
                            mine.push((index, bake_chunk(shared, index, sun, occluder, reach)));
                            let mut done = done.lock().unwrap_or_else(|e| e.into_inner());
                            *done += 1;
                            watch(*done);
                        }
                        mine
                    })
                })
                .collect();
            workers.into_iter().map(|w| w.join().unwrap_or_default()).collect()
        });
        for (index, bits) in finished.into_iter().flatten() {
            results[index] = Some(bits);
        }
    }

    let mut baked = Baked::default();
    for (index, bits) in results.into_iter().enumerate() {
        let bits = bits.unwrap_or_else(|| encode_shadow_map(&[0u8; ALPHA_LEN]));
        let any = bits.iter().any(|&b| b != 0);
        if any {
            baked.shadowed += 1;
        }
        if set_shadow(tile, index, any.then_some(bits.as_slice())) {
            baked.changed += 1;
        }
    }
    baked
}

/// How many chunks a bake walks, for a caller sizing a progress bar.
pub const CHUNKS: usize = CHUNKS_PER_SIDE * CHUNKS_PER_SIDE;

/// Put a chunk's `MCSH` bits in place, or take the region away.
///
/// Returns whether anything moved, so a caller can skip writing a tile nothing
/// changed on.
pub(crate) fn set_shadow(tile: &mut AdtFile, chunk_index: usize, bits: Option<&[u8]>) -> bool {
    let Some(chunk) = tile.chunks.get_mut(chunk_index) else {
        return false;
    };
    let was = chunk
        .region(Region::Shadow)
        .map(|sub| sub.data.clone());
    let now = bits.map(|b| b.to_vec());
    let flags = u32_at(&chunk.header, at::FLAGS);
    let wanted_flags = match now.is_some() {
        true => flags | mcnk_flags::HAS_MCSH,
        false => flags & !mcnk_flags::HAS_MCSH,
    };
    if was == now && flags == wanted_flags {
        return false;
    }

    chunk.regions[Region::Shadow as usize] = now.map(SubChunk::new);
    put_u32(&mut chunk.header, at::FLAGS, wanted_flags);
    true
}

/// A chunk's `MCSH` as one byte per texel, `255` in shadow, decoded by
/// `vale_assets`' reader.
///
/// `None` when the chunk declares no shadow. That draws the same as an
/// all-zero map but is a different file.
pub fn shadow_texels(tile: &AdtFile, chunk_index: usize) -> Option<Vec<u8>> {
    let chunk = tile.chunks.get(chunk_index)?;
    if u32_at(&chunk.header, at::FLAGS) & mcnk_flags::HAS_MCSH == 0 {
        return None;
    }
    Some(decode_shadow_map(&chunk.region(Region::Shadow)?.data))
}

fn normalise(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    match len > 1.0e-6 {
        true => [v[0] / len, v[1] / len, v[2] / len],
        false => [0.0, 0.0, 1.0],
    }
}

fn u32_at(buf: &[u8], at: usize) -> u32 {
    let mut word = [0u8; 4];
    word.copy_from_slice(&buf[at..at + 4]);
    u32::from_le_bytes(word)
}

fn put_u32(buf: &mut [u8], at: usize, value: u32) {
    buf[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adt::blank::{blank_tile, chunk_origin};
    use vale_assets::tables::light::SHADOW_SUN as SUN_TOWARD;

    const GRASS: &str = r"Tileset\Elwynn\ElwynnGrassBase.blp";

    fn tile() -> AdtFile {
        blank_tile(32, 48, GRASS, 100.0, 0)
    }

    /// An occluder that blocks everything: the whole tile is in shadow.
    struct Everything;
    impl Occluder for Everything {
        fn blocked(&self, _: [f32; 3], _: [f32; 3], _: f32) -> bool {
            true
        }
    }

    /// With nothing casting, no chunk has an `MCSH` region or flag. A blank
    /// tile has neither to begin with, so this also checks that a bake over
    /// empty ground leaves the file unchanged.
    #[test]
    fn an_unoccluded_tile_carries_no_shadow_at_all() {
        let mut tile = tile();
        let baked = bake(&mut tile, SUN_TOWARD, &Nothing, REACH);
        assert_eq!(baked.shadowed, 0);
        assert_eq!(baked.changed, 0, "nothing should have moved");
        for chunk in &tile.chunks {
            assert!(chunk.region(Region::Shadow).is_none());
            assert_eq!(chunk.head().flags() & mcnk_flags::HAS_MCSH, 0);
        }
    }

    /// With everything casting, every chunk has the region, the flag and every
    /// bit set.
    #[test]
    fn a_fully_occluded_tile_is_all_shadow() {
        let mut tile = tile();
        let baked = bake(&mut tile, SUN_TOWARD, &Everything, REACH);
        assert_eq!(baked.shadowed, 256);
        assert_eq!(baked.changed, 256);
        for (i, chunk) in tile.chunks.iter().enumerate() {
            let sub = chunk.region(Region::Shadow).expect("region");
            assert_eq!(sub.data.len(), vale_assets::world::adt::SHADOW_MAP_BYTES);
            assert!(sub.data.iter().all(|&b| b == 0xFF), "chunk {i}");
            assert_ne!(chunk.head().flags() & mcnk_flags::HAS_MCSH, 0);
        }
    }

    /// Baking with nothing after baking with everything removes the regions
    /// and flags again, so a tile whose casters were deleted is written without
    /// shadow rather than with 16 KB of zeroes.
    #[test]
    fn clearing_a_bake_removes_the_region_and_the_flag() {
        let mut tile = tile();
        bake(&mut tile, SUN_TOWARD, &Everything, REACH);
        let baked = bake(&mut tile, SUN_TOWARD, &Nothing, REACH);
        assert_eq!(baked.changed, 256);
        assert_eq!(baked.shadowed, 0);
        assert!(tile.chunks.iter().all(|c| c.region(Region::Shadow).is_none()));
        assert!(tile
            .chunks
            .iter()
            .all(|c| c.head().flags() & mcnk_flags::HAS_MCSH == 0));
        // The file is the one the tile started as.
        assert_eq!(tile.write(), self::tests::tile().write());
    }

    /// A second bake with the same occluder changes nothing, so a caller can
    /// rebake without rewriting an unchanged file.
    #[test]
    fn baking_twice_is_idempotent() {
        let mut tile = tile();
        bake(&mut tile, SUN_TOWARD, &Everything, REACH);
        let again = bake(&mut tile, SUN_TOWARD, &Everything, REACH);
        assert_eq!(again.changed, 0);
    }

    /// The bits land on the texels the occluder covers. An occluder that blocks
    /// only rays starting below a world x shadows the rows the texel mapping
    /// puts there. A transposed or mirrored map would still look like a
    /// plausible shadow, so this is checked directly.
    #[test]
    fn a_half_blocked_chunk_shadows_the_right_half() {
        let mut tile = tile();
        // Chunk 0's origin is its maximum corner; x decreases along rows.
        let origin = chunk_origin(32, 48, 0);
        let cut = origin[0] - CHUNK_SIZE / 2.0;
        let half = move |from: [f32; 3], _: [f32; 3], _: f32| from[0] < cut;
        bake(&mut tile, SUN_TOWARD, &half, REACH);

        let texels = shadow_texels(&tile, 0).expect("chunk 0 has a shadow");
        for ty in 0..ALPHA_SIDE {
            for tx in 0..ALPHA_SIDE {
                let lit = texels[ty * ALPHA_SIDE + tx] == 0;
                // Rows past the middle are the ones with the smaller x.
                let expected_lit = ty < ALPHA_SIDE / 2;
                assert_eq!(lit, expected_lit, "texel ({tx}, {ty})");
            }
        }
    }

    /// The same check along world y and the texel columns.
    #[test]
    fn a_half_blocked_chunk_shadows_the_right_columns() {
        let mut tile = tile();
        let origin = chunk_origin(32, 48, 0);
        let cut = origin[1] - CHUNK_SIZE / 2.0;
        let half = move |from: [f32; 3], _: [f32; 3], _: f32| from[1] < cut;
        bake(&mut tile, SUN_TOWARD, &half, REACH);

        let texels = shadow_texels(&tile, 0).unwrap();
        for ty in 0..ALPHA_SIDE {
            for tx in 0..ALPHA_SIDE {
                let lit = texels[ty * ALPHA_SIDE + tx] == 0;
                assert_eq!(lit, tx < ALPHA_SIDE / 2, "texel ({tx}, {ty})");
            }
        }
    }

    /// The bake packs bits in the order the reader unpacks them. A bake packed
    /// most significant bit first would decode as stripes with a period of
    /// eight texels, so this compares against the reader's own decode.
    #[test]
    fn the_packed_bits_are_the_order_the_reader_unpacks() {
        let mut tile = tile();
        // One column in eight, so a mirrored byte shows.
        let every_eighth = |from: [f32; 3], _: [f32; 3], _: f32| {
            let origin = chunk_origin(32, 48, 0);
            let texel = CHUNK_SIZE / ALPHA_SIDE as f32;
            let tx = ((origin[1] - from[1]) / texel) as usize;
            tx % 8 == 0
        };
        bake(&mut tile, SUN_TOWARD, &every_eighth, REACH);

        let raw = tile.write();
        let adt = vale_assets::world::adt::Adt::parse(&raw).unwrap();
        let shadow = &adt.chunks[0].shadow;
        assert_eq!(shadow.len(), ALPHA_SIDE * ALPHA_SIDE);
        // Column 0 of every row is shadowed and column 1 is not. The reader's
        // `fix_alpha_edge` rewrites the last row and column, so the check stops
        // short of them.
        for ty in 0..ALPHA_SIDE - 1 {
            assert_ne!(shadow[ty * ALPHA_SIDE], 0, "row {ty} column 0");
            assert_eq!(shadow[ty * ALPHA_SIDE + 1], 0, "row {ty} column 1");
            assert_ne!(shadow[ty * ALPHA_SIDE + 8], 0, "row {ty} column 8");
        }
    }

    /// A baked tile parses with `vale_assets`' reader and with this crate's.
    #[test]
    fn a_baked_tile_still_parses() {
        let mut tile = tile();
        bake(&mut tile, SUN_TOWARD, &Everything, REACH);
        let raw = tile.write();
        let adt = vale_assets::world::adt::Adt::parse(&raw).unwrap();
        assert_eq!(adt.chunks.len(), 256);
        for chunk in &adt.chunks {
            assert_eq!(chunk.shadow.len(), ALPHA_SIDE * ALPHA_SIDE);
            assert!(chunk.shadow.iter().all(|&t| t == 255));
        }
        // This crate reads it back byte for byte.
        assert_eq!(AdtFile::parse(&raw).unwrap().write(), raw);
    }

    /// Flat ground casts no shadow on itself: every sample is at the height of
    /// the ray's start and the ray climbs away from it.
    #[test]
    fn flat_ground_casts_no_shadow_on_itself() {
        let mut tile = tile();
        let ground = Ground::of(&tile);
        let baked = bake(&mut tile, SUN_TOWARD, &ground, REACH);
        assert_eq!(baked.shadowed, 0, "flat ground shadowed itself");
    }

    /// A step in the ground shadows the low side next to it. One half of the
    /// tile is raised 300 yards; the chunk at the foot of the step, on the side
    /// away from the sun, must be shadowed. This checks that the march walks
    /// toward the sun.
    #[test]
    fn a_step_in_the_ground_shadows_what_is_behind_it() {
        let mut tile = tile();
        // The sun is at azimuth 45 from +x toward +y, so ground at higher x
        // shadows ground at lower x.
        for index in 0..tile.chunks.len() {
            let origin = chunk_origin(32, 48, index);
            // Chunks in the half nearer the tile's maximum x get raised.
            if origin[0] > chunk_origin(32, 48, 128)[0] {
                let at = tile.chunks[index].head().position();
                tile.chunks[index]
                    .head_mut()
                    .set_position([at[0], at[1], at[2] + 300.0]);
            }
        }
        let ground = Ground::of(&tile);
        let baked = bake(&mut tile, SUN_TOWARD, &ground, REACH);
        assert!(
            baked.shadowed > 0,
            "a 300-yard step cast nothing at all"
        );
        // Chunk 128 is the first chunk of row 8, the first unraised row, with
        // the 300-yard wall along its high-x edge.
        let low = shadow_texels(&tile, 128).expect("the chunk under the step has a shadow");
        let shadowed = low.iter().filter(|&&t| t != 0).count();
        assert!(
            shadowed > low.len() / 4,
            "only {shadowed} of {} texels under a 300-yard wall are shadowed",
            low.len()
        );
        // The raised half, which nothing stands over, is lit.
        assert!(
            shadow_texels(&tile, 0).is_none(),
            "the top of the step shadowed itself"
        );
    }

    /// A ray off the tile is not blocked by the ground occluder, which knows
    /// no ground there.
    #[test]
    fn a_ray_off_the_tile_is_not_blocked() {
        let tile = tile();
        let ground = Ground::of(&tile);
        // Well outside the tile, aimed away from it.
        let outside = [-20000.0, -20000.0, 500.0];
        assert!(!ground.blocked(outside, normalise(SUN_TOWARD), REACH));
    }

    /// `Either` blocks a ray when either occluder does.
    #[test]
    fn either_is_blocked_when_either_is() {
        let yes = Everything;
        let no = Nothing;
        let from = [0.0, 0.0, 0.0];
        let up = [0.0, 0.0, 1.0];
        assert!(Either(&yes, &no).blocked(from, up, 10.0));
        assert!(Either(&no, &yes).blocked(from, up, 10.0));
        assert!(!Either(&no, &no).blocked(from, up, 10.0));
    }

    /// The watcher is called once per chunk with 1, 2, … up to the chunk
    /// count, in order, although the chunks are baked on several threads.
    #[test]
    fn a_bake_reports_every_chunk_once() {
        let mut tile = tile();
        let seen = std::sync::Mutex::new(Vec::new());
        bake_watching(&mut tile, SUN_TOWARD, &Nothing, REACH, &|done| {
            seen.lock().unwrap().push(done);
        });
        let seen = seen.into_inner().unwrap();
        assert_eq!(seen.len(), CHUNKS);
        assert_eq!(seen.first(), Some(&1));
        assert_eq!(seen.last(), Some(&CHUNKS));
        assert!(seen.windows(2).all(|w| w[1] == w[0] + 1), "out of order");
    }

    /// Rays start `LIFT` above the ground. Starting on it, a bake that includes
    /// the terrain would have every ray hit the triangle it starts on.
    #[test]
    fn rays_start_above_the_ground() {
        let tile = tile();
        let seen = std::cell::RefCell::new(Vec::new());
        let record = |from: [f32; 3], _: [f32; 3], _: f32| {
            seen.borrow_mut().push(from[2]);
            false
        };
        bake_chunk(&tile, 0, SUN_TOWARD, &record, REACH);
        let seen = seen.into_inner();
        assert_eq!(seen.len(), ALPHA_SIDE * ALPHA_SIDE);
        assert!(seen.iter().all(|&z| (z - (100.0 + LIFT)).abs() < 0.001));
    }

    /// A horizontal square of two triangles, centred on `(x, y)` at height
    /// `z`, `half` yards from centre to edge.
    fn square(x: f32, y: f32, z: f32, half: f32) -> Vec<[[f32; 3]; 3]> {
        let (a, b, c, d) = (
            [x - half, y - half, z],
            [x + half, y - half, z],
            [x + half, y + half, z],
            [x - half, y + half, z],
        );
        vec![[a, b, c], [a, c, d]]
    }

    fn casters(triangles: Vec<[[f32; 3]; 3]>) -> Casters {
        Casters::new(
            ShadowCasters {
                triangles,
                ..Default::default()
            },
            128,
        )
    }

    /// A ray straight up is blocked by a square above it and not by one beside
    /// it.
    #[test]
    fn casters_block_what_is_overhead() {
        let up = [0.0, 0.0, 1.0];
        let over = casters(square(10.0, 10.0, 20.0, 1.0));
        assert!(over.blocked([10.0, 10.0, 0.0], up, REACH));
        assert!(!over.blocked([20.0, 10.0, 0.0], up, REACH));
        // Above the square, the ray meets nothing.
        assert!(!over.blocked([10.0, 10.0, 30.0], up, REACH));
        // Shorter than the distance to the square, the ray meets nothing.
        assert!(!over.blocked([10.0, 10.0, 0.0], up, 10.0));
    }

    /// The grid walk gives the same answer as testing every triangle, over a
    /// scattered set and slanted rays that cross many cells.
    #[test]
    fn casters_agree_with_testing_every_triangle() {
        let mut seed = 12345u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let mut triangles = Vec::new();
        for _ in 0..300 {
            let (x, y, z) = (next() * 200.0 - 100.0, next() * 200.0 - 100.0, next() * 40.0);
            let half = 0.5 + next() * 3.0;
            triangles.extend(square(x, y, z, half));
        }
        let grid = casters(triangles.clone());
        let sun = normalise(SUN_TOWARD);
        let (mut hits, mut tested) = (0, 0);
        for _ in 0..4000 {
            let from = [next() * 240.0 - 120.0, next() * 240.0 - 120.0, next() * 10.0 - 5.0];
            let every = triangles
                .iter()
                .any(|t| ray_meets(from, sun, t).is_some_and(|(d, _, _)| d <= REACH));
            assert_eq!(grid.blocked(from, sun, REACH), every, "from {from:?}");
            hits += usize::from(every);
            tested += 1;
        }
        assert!(hits > tested / 20, "only {hits} of {tested} rays hit anything");
    }

    /// A cut-out triangle casts where its texture is opaque and not where it
    /// is transparent.
    #[test]
    fn a_cut_out_casts_only_where_its_texture_is_opaque() {
        let triangles = square(0.0, 0.0, 10.0, 1.0);
        // Two texels across: the left column transparent, the right opaque.
        let mask = AlphaMask {
            width: 2,
            height: 1,
            alpha: vec![0, 255],
        };
        // u runs with world x across the square.
        let uv = |p: [f32; 3]| [(p[0] + 1.0) / 2.0, 0.5];
        let cutouts = triangles
            .iter()
            .map(|t| {
                Some(Cutout {
                    mask: 0,
                    uvs: [uv(t[0]), uv(t[1]), uv(t[2])],
                })
            })
            .collect();
        let grid = Casters::new(
            ShadowCasters {
                triangles,
                cutouts,
                masks: vec![mask],
            },
            128,
        );
        let up = [0.0, 0.0, 1.0];
        assert!(!grid.blocked([-0.5, 0.0, 0.0], up, REACH), "the transparent half cast");
        assert!(grid.blocked([0.5, 0.0, 0.0], up, REACH), "the opaque half did not cast");
    }
}
