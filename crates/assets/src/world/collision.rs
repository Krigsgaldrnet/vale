//! Collision hulls for buildings, doodads and game objects, and the floor and
//! wall queries the local movement simulation makes against them.
//!
//! The server is authoritative and runs its own collision against the vmaps its
//! extractor built from these same files, so nothing here changes the server's
//! state. This module serves the local simulation. Without it the character
//! falls through Stormwind to the terrain under the city, and walks into a wall
//! until the server moves them back.
//!
//! ## WMO hulls: solidity is decided per triangle, not per batch
//!
//! [`crate::world::wmo::collides`] is the rule, and it reads `MOPY`, a chunk the
//! render path does not use. As a result the collision hull is not the drawn
//! geometry:
//!
//! * a WMO carries invisible collision triangles (`MOPY` `COLLISION` with no
//!   `RENDER` bit), which no batch references and which `WmoModel::assemble`
//!   therefore never sees;
//! * it carries drawn triangles that are not solid (`DETAIL`: grass laid over
//!   a floor, trim laid over a wall), which would add to the hull without
//!   blocking anything;
//! * `assemble` drops batches whose material does not resolve and groups that
//!   are antiportals, so a hull built from its output would have holes where a
//!   room failed to texture.
//!
//! [`CollisionMesh::build`] therefore walks the groups again from
//! `MOVT`/`MOVI`/`MOPY` and ignores the draw list. It is the same pass vmangos'
//! `WMOGroup::ConvertToVMAPGroupWmo` makes over the same three chunks.
//!
//! ## M2 hulls: the `BoundingTriangles` block
//!
//! A doodad (a tree, a fence, a crate, the furniture inside a building)
//! carries its own hull in the `BoundingTriangles` block, which
//! [`crate::world::m2::M2::collision`] reads and the render path does not use.
//! That block has no `MOPY` and no per-triangle flag. It exists only for
//! collision, so every triangle in it is solid. A model with no such block is
//! not an error: it is a model the character walks through, which covers most
//! grass, every bird and every flame. vmangos' `Model::open` drops the same
//! models before they reach a vmap.
//!
//! ## Game object hulls
//!
//! A game object (a door, a portcullis, a chest, a mailbox) is not in any
//! `MDDF` or `MODF` row. The server spawns it as an entity with a
//! `GAMEOBJECT_DISPLAYID`, a position and a facing, and can remove it during
//! the session. Its hull is the `BoundingTriangles` block of its M2, as for a
//! doodad, but its lifetime is not a tile's, so it is held apart from the
//! tiles, keyed by GUID, in [`Objects`].
//!
//! A game object is also solid in only some of its states.
//! [`game_object_is_solid`] is that rule, taken from vmangos: a closed
//! Deadmines door blocks a stride, and the same door open does not.
//!
//! The code after this point is shared by all three sources: [`Collider`] does
//! not depend on which file a hull came from.
//!
//! ## Queries
//!
//! [`CollisionWorld::floor`] returns the surface under the character, and
//! [`CollisionWorld::step`] returns where a stride ends. The session thread
//! calls both twenty times a second, which sets the design: a placed building
//! keeps its triangles in world space in a uniform grid over the XY footprint,
//! so a query touches one cell rather than Stormwind's quarter-million
//! triangles.
//!
//! ## Step height and wall probe heights
//!
//! [`STEP_UP`] and [`PROBE_HEIGHTS`] are not in any file. They decide whether
//! stairs work: a stair riser is a vertical wall a few inches high, so a wall
//! probe at ankle height stops the character at the bottom of every staircase.
//! The probes therefore start above the height a character can climb, and the
//! floor query lifts the character over everything below that. Wrong values do
//! not produce an error; they produce a client that cannot enter a building.
//! The numbers are named constants for that reason.

use crate::world::wmo::WmoGroup;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// How far above the feet a surface can be and still be the one stood on.
///
/// This is the client's step height: the character is lifted onto anything
/// within it without jumping, and anything above it is a ledge they are under
/// rather than a floor they are on. It is also the ceiling on the floor query,
/// which keeps a character walking under a bridge from being lifted onto its
/// deck.
pub const STEP_UP: f32 = 1.0;

/// Heights above the feet at which a stride is tested for walls.
///
/// The lowest is above [`STEP_UP`]. A staircase is a run of vertical risers,
/// and a probe below the step height hits the first one and stops the
/// character at the bottom. Geometry below the lowest probe is walked over and
/// then lifted by the floor query, which is what the 1.12.1 client does with a
/// kerb.
pub const PROBE_HEIGHTS: [f32; 2] = [STEP_UP + 0.2, 1.9];

/// How close to a wall the character is allowed to get, in yards.
///
/// The stride is tested this much further than it actually goes, so the
/// character stops a body's width short instead of ending the frame with their
/// centre exactly on the surface. It is also the half-width the stride is swept
/// at — see [`PROBE_LANES`].
pub const PLAYER_RADIUS: f32 = 0.5;

/// Sideways offsets, in yards, of the lines a stride is tested along. See
/// `wall_ahead` for the reasoning.
///
/// The centre and both shoulders. A single ray along the centre axis stops a
/// stride only when something is directly ahead of the character's centre: it
/// misses a post, a trunk, a door jamb approached at an angle, and any corner.
pub(crate) const PROBE_LANES: [f32; 3] = [0.0, -PLAYER_RADIUS, PLAYER_RADIUS];

/// How many times one stride may have a wall's own direction taken out of it
/// before the answer is to stand still.
///
/// Two walls is a doorway and is walked through; three is a corner, and no
/// direction is left for the stride. When the attempts run out,
/// [`CollisionWorld::step`] stands still rather than applying the remaining
/// motion.
const CANCELS: usize = 3;

/// A triangle whose normal is this far from vertical is a wall, and blocks;
/// anything flatter is a floor, and is left to the floor query.
///
/// 0.5 is 60° from horizontal — steeper than any ramp or staircase the game
/// asks a character to walk up, and shallower than any wall.
pub(crate) const WALL_NORMAL_Z: f32 = 0.5;

/// Target size of a grid cell, in yards. The grid is capped at
/// [`GRID_MAX`] cells per axis, so a building larger than
/// `GRID_MAX * CELL_SIZE` gets coarser cells rather than a bigger grid.
const CELL_SIZE: f32 = 8.0;

/// Cap on a collider's grid, per axis. Stormwind's hull spans hundreds of
/// yards; without a cap the grid, not the geometry, would be the memory cost.
const GRID_MAX: usize = 192;

// ---------------------------------------------------------------------------
// The hull, in the file's own space
// ---------------------------------------------------------------------------

/// A WMO's solid triangles in its own model space — the same space `MOVT` is
/// in, so [`adt::placement_matrix`] places it unchanged.
///
/// [`adt::placement_matrix`]: crate::world::adt::placement_matrix
#[derive(Debug, Clone, Default)]
pub struct CollisionMesh {
    pub positions: Vec<[f32; 3]>,
    /// Triangle list. Vertices are copied per hull rather than shared with the
    /// render buffers, because the hull keeps a different subset of them.
    pub indices: Vec<u32>,
    /// How many triangles the groups declared in total, solid or not. Kept only
    /// so `vale collision` can report the ratio. A hull that is 100% of the
    /// geometry means the `DETAIL` term was dropped, and a hull that is 0% means
    /// `MOPY` was not read. Neither fault shows up anywhere else.
    pub declared_triangles: usize,
    /// Which `MOGP` group each solid triangle came from, as an index into
    /// [`Self::group_flags`] — one entry per triangle, or empty for a hull that
    /// has no groups (an M2's `BoundingTriangles`, and the tests' invented ones).
    ///
    /// Used to find which group's floor a character is standing on, which
    /// decides whether they are outdoors. See [`Collider::building_floor`], and
    /// [`crate::world::wmo::group_flags::OUTDOOR`] for the bit read from that
    /// group.
    pub triangle_groups: Vec<u16>,
    /// The `MOGP` flag word of each group in [`Self::triangle_groups`], in the
    /// order [`Self::build`] walked them.
    pub group_flags: Vec<u32>,
}

impl CollisionMesh {
    /// Build the hull from a WMO's groups.
    ///
    /// Groups vmangos' extractor skips are skipped here for its reasons
    /// ([`WmoGroup::is_uncollidable`]), and within a group a triangle is kept
    /// only if [`crate::world::wmo::collides`] says so. Vertices are compacted, so a group
    /// that contributes no solid triangle costs nothing.
    pub fn build(groups: &[WmoGroup]) -> CollisionMesh {
        let mut mesh = CollisionMesh::default();
        for group in groups {
            if group.is_uncollidable() {
                continue;
            }
            let triangles = group.indices.len() / 3;
            mesh.declared_triangles += triangles;
            // A group past the 65,536th cannot be told apart from the ones
            // before it, and no shipped building has more than 306 — see
            // `wmo::MAX_GROUPS`, which refuses a root long before this.
            let group_index = u16::try_from(mesh.group_flags.len()).unwrap_or(u16::MAX);
            mesh.group_flags.push(group.flags);

            // Compacted per group: a room's solid triangles reference a handful
            // of its vertices, and copying the whole `MOVT` for each would put
            // Stormwind's 844,627 vertices in the hull twice over.
            let mut remap: HashMap<u16, u32> = HashMap::new();
            for triangle in 0..triangles {
                if !group.triangle_collides(triangle) {
                    continue;
                }
                let corners: [u16; 3] = [
                    group.indices[triangle * 3],
                    group.indices[triangle * 3 + 1],
                    group.indices[triangle * 3 + 2],
                ];
                // A triangle indexing past `MOVT` is a damaged tail. Checked
                // before any corner is pushed, because emitting one or two of
                // three would shift every later triangle and produce a hull
                // that does not match the building.
                if corners.iter().any(|&c| c as usize >= group.positions.len()) {
                    continue;
                }
                for local in corners {
                    let next = mesh.positions.len() as u32;
                    let index = *remap.entry(local).or_insert(next);
                    if index == next {
                        mesh.positions.push(group.positions[local as usize]);
                    }
                    mesh.indices.push(index);
                }
                mesh.triangle_groups.push(group_index);
            }
        }
        mesh
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

// ---------------------------------------------------------------------------
// One placement, in world space
// ---------------------------------------------------------------------------

/// One `MODF` placement's hull, transformed into world space and indexed.
///
/// World space rather than model space with an inverse transform per query,
/// because a placement is built once and queried for as long as its tile is
/// loaded, and inverting a matrix with a 180° term in it is another place
/// where an axis error would go unnoticed.
#[derive(Debug)]
pub struct Collider {
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
    bounds: [[f32; 3]; 2],
    cols: usize,
    rows: usize,
    cell: [f32; 2],
    /// CSR: `starts[c]..starts[c + 1]` indexes `cell_triangles`.
    starts: Vec<u32>,
    /// Triangle indices (not index-buffer offsets), bucketed by cell.
    cell_triangles: Vec<u32>,
    /// [`CollisionMesh::triangle_groups`] and [`CollisionMesh::group_flags`],
    /// carried through unchanged: a placement moves the triangles and not the
    /// groups they belong to.
    triangle_groups: Vec<u16>,
    group_flags: Vec<u32>,
}

impl Collider {
    /// Place a hull with a `MODF` placement matrix (column-major, as
    /// [`adt::placement_matrix`] returns it).
    ///
    /// [`adt::placement_matrix`]: crate::world::adt::placement_matrix
    pub fn place(mesh: &CollisionMesh, matrix: &[f32; 16]) -> Collider {
        let positions: Vec<[f32; 3]> = mesh
            .positions
            .iter()
            .map(|v| transform_point(matrix, *v))
            .collect();
        Self::from_positions(
            positions,
            mesh.indices.clone(),
            mesh.triangle_groups.clone(),
            mesh.group_flags.clone(),
        )
    }

    /// The same hull moved by `delta`, a column-major matrix in world space:
    /// every position through it, the bounds and the grid rebuilt. The
    /// triangles, their groups and the flags are the mesh's and do not
    /// change. This is how a hull follows a placement that is moved while its
    /// tile is on screen, without the model it was placed from.
    pub fn moved(&self, delta: &[f32; 16]) -> Collider {
        let positions: Vec<[f32; 3]> = self
            .positions
            .iter()
            .map(|v| transform_point(delta, *v))
            .collect();
        Self::from_positions(
            positions,
            self.indices.clone(),
            self.triangle_groups.clone(),
            self.group_flags.clone(),
        )
    }

    /// Build the hull over world-space positions: the bounds, the grid and
    /// the cell index.
    fn from_positions(
        positions: Vec<[f32; 3]>,
        indices: Vec<u32>,
        triangle_groups: Vec<u16>,
        group_flags: Vec<u32>,
    ) -> Collider {
        let mut bounds = [[f32::MAX; 3], [f32::MIN; 3]];
        for v in &positions {
            for axis in 0..3 {
                bounds[0][axis] = bounds[0][axis].min(v[axis]);
                bounds[1][axis] = bounds[1][axis].max(v[axis]);
            }
        }
        if positions.is_empty() {
            bounds = [[0.0; 3]; 2];
        }

        let extent = [
            (bounds[1][0] - bounds[0][0]).max(0.0),
            (bounds[1][1] - bounds[0][1]).max(0.0),
        ];
        let cols = grid_axis(extent[0]);
        let rows = grid_axis(extent[1]);
        // A cell is never zero-sized: a flat building would divide by zero, and
        // an axis with no extent still needs one column.
        let cell = [
            (extent[0] / cols as f32).max(f32::MIN_POSITIVE),
            (extent[1] / rows as f32).max(f32::MIN_POSITIVE),
        ];

        let mut collider = Collider {
            positions,
            indices,
            bounds,
            cols,
            rows,
            cell,
            starts: vec![0; cols * rows + 1],
            cell_triangles: Vec::new(),
            triangle_groups,
            group_flags,
        };
        collider.index_cells();
        collider
    }

    /// Bucket every triangle into the cells its XY footprint overlaps, as CSR:
    /// count, prefix-sum, then fill. A `Vec<Vec<u32>>` would be one allocation
    /// per cell and Stormwind's grid is 36,864 of them.
    fn index_cells(&mut self) {
        let cells = self.cols * self.rows;
        let triangles = self.indices.len() / 3;
        let mut scratch = Vec::new();

        let mut counts = vec![0u32; cells];
        for triangle in 0..triangles {
            self.cells_of(triangle, &mut scratch);
            for &cell in &scratch {
                counts[cell] += 1;
            }
        }
        let mut total = 0u32;
        for (cell, count) in counts.iter().enumerate() {
            self.starts[cell] = total;
            total += count;
        }
        self.starts[cells] = total;

        self.cell_triangles = vec![0; total as usize];
        let mut cursor: Vec<u32> = self.starts[..cells].to_vec();
        for triangle in 0..triangles {
            self.cells_of(triangle, &mut scratch);
            for &cell in &scratch {
                self.cell_triangles[cursor[cell] as usize] = triangle as u32;
                cursor[cell] += 1;
            }
        }
    }

    /// Every grid cell a triangle's XY bounding box touches.
    ///
    /// Fills a caller-owned buffer rather than allocating, because the two
    /// passes of [`Self::index_cells`] must agree exactly — a triangle counted
    /// in one pass but not filled in the other corrupts the whole CSR silently
    /// — and the cheapest way to guarantee that is one function called twice.
    fn cells_of(&self, triangle: usize, out: &mut Vec<usize>) {
        out.clear();
        let [a, b, c] = self.triangle(triangle);
        let lo = [a[0].min(b[0]).min(c[0]), a[1].min(b[1]).min(c[1])];
        let hi = [a[0].max(b[0]).max(c[0]), a[1].max(b[1]).max(c[1])];
        let (col0, row0) = self.cell_of(lo[0], lo[1]);
        let (col1, row1) = self.cell_of(hi[0], hi[1]);
        for row in row0..=row1 {
            for col in col0..=col1 {
                out.push(row * self.cols + col);
            }
        }
    }

    fn cell_of(&self, x: f32, y: f32) -> (usize, usize) {
        let col = ((x - self.bounds[0][0]) / self.cell[0]) as isize;
        let row = ((y - self.bounds[0][1]) / self.cell[1]) as isize;
        (
            col.clamp(0, self.cols as isize - 1) as usize,
            row.clamp(0, self.rows as isize - 1) as usize,
        )
    }

    /// The three world-space corners of one solid triangle.
    ///
    /// Public so a check can walk the hull it built: `vale collision` passes
    /// triangle centroids back through [`Self::floor`] and
    /// [`CollisionWorld::step`], which tests the grid against shipped buildings
    /// rather than invented ones.
    pub fn triangle(&self, triangle: usize) -> [[f32; 3]; 3] {
        let i = triangle * 3;
        [
            self.positions[self.indices[i] as usize],
            self.positions[self.indices[i + 1] as usize],
            self.positions[self.indices[i + 2] as usize],
        ]
    }

    fn cell_slice(&self, col: usize, row: usize) -> &[u32] {
        let cell = row * self.cols + col;
        let start = self.starts[cell] as usize;
        let end = self.starts[cell + 1] as usize;
        &self.cell_triangles[start..end]
    }

    /// Unit normal of one solid triangle, or zero for a degenerate one.
    pub fn triangle_normal(&self, triangle: usize) -> [f32; 3] {
        normal_of(&self.triangle(triangle))
    }

    /// Is this triangle one [`CollisionWorld::step`] would block on?
    pub fn triangle_is_wall(&self, triangle: usize) -> bool {
        let normal = self.triangle_normal(triangle);
        normal != [0.0; 3] && normal[2].abs() < WALL_NORMAL_Z
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Every world-space vertex of the hull.
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }

    /// World-space bounding box of the hull.
    pub fn bounds(&self) -> [[f32; 3]; 2] {
        self.bounds
    }

    /// The highest solid surface under `(x, y)` that is at or below `ceiling`,
    /// or `None` where the hull has nothing over that point.
    ///
    /// `ceiling` stops the query from returning the roof while the character
    /// stands on the ground floor; the caller sets it to the feet plus
    /// [`STEP_UP`].
    pub fn floor(&self, x: f32, y: f32, ceiling: f32) -> Option<f32> {
        self.surface(x, y, ceiling).map(|(z, _)| z)
    }

    /// The surface [`Self::floor`] returns, with its slope as an upward unit
    /// normal.
    ///
    /// The normal is that of the triangle [`Self::floor`] took the height from.
    /// This is one function with two callers rather than two lookups because a
    /// model tilted by one triangle while standing at another's height leans
    /// the wrong way on every step of a staircase.
    pub fn surface(&self, x: f32, y: f32, ceiling: f32) -> Option<(f32, [f32; 3])> {
        if x < self.bounds[0][0]
            || x > self.bounds[1][0]
            || y < self.bounds[0][1]
            || y > self.bounds[1][1]
            || ceiling < self.bounds[0][2]
        {
            return None;
        }
        let (col, row) = self.cell_of(x, y);
        let mut best: Option<(f32, usize)> = None;
        for &triangle in self.cell_slice(col, row) {
            let triangle = triangle as usize;
            let corners = self.triangle(triangle);
            let Some(z) = height_at(&corners, x, y) else {
                continue;
            };
            if z <= ceiling && best.is_none_or(|(b, _)| z > b) {
                best = Some((z, triangle));
            }
        }
        best.map(|(z, triangle)| {
            // `normal_of` returns zero for a degenerate triangle and otherwise
            // follows the file's winding, which in a `BoundingTriangles` hull
            // does not always face up. Both cases return an upward normal here.
            let n = self.triangle_normal(triangle);
            let n = match n[2] {
                _ if n == [0.0; 3] => [0.0, 0.0, 1.0],
                z if z < 0.0 => [-n[0], -n[1], -n[2]],
                _ => n,
            };
            (z, n)
        })
    }

    /// The highest floor under `(x, y)` at or below `ceiling`, and the `MOGP`
    /// flag word of the group it belongs to. `None` where the hull has nothing
    /// there, or has no groups (an M2).
    ///
    /// This is vmangos' `VMapManager2::getAreaInfo`: a ray straight down from
    /// the probe, and the group of the first triangle it meets. The server uses
    /// it to decide `IsOutdoors`. The 1.12.1 client decides the same question
    /// by which WMO group contains the unit, a containment test this client
    /// does not have. In every building measured, the floor under the feet
    /// belongs to the room the feet are in, which is why the server's method
    /// works.
    pub fn building_floor(&self, x: f32, y: f32, ceiling: f32) -> Option<(f32, u32)> {
        if self.triangle_groups.is_empty()
            || x < self.bounds[0][0]
            || x > self.bounds[1][0]
            || y < self.bounds[0][1]
            || y > self.bounds[1][1]
            || ceiling < self.bounds[0][2]
        {
            return None;
        }
        let (col, row) = self.cell_of(x, y);
        let mut best: Option<(f32, usize)> = None;
        for &triangle in self.cell_slice(col, row) {
            let triangle = triangle as usize;
            let Some(z) = height_at(&self.triangle(triangle), x, y) else {
                continue;
            };
            if z <= ceiling && best.is_none_or(|(b, _)| z > b) {
                best = Some((z, triangle));
            }
        }
        let (z, triangle) = best?;
        let group = *self.triangle_groups.get(triangle)?;
        Some((z, *self.group_flags.get(usize::from(group))?))
    }

    /// The nearest surface of any slope a ray meets, as a distance.
    ///
    /// [`Self::wall`] serves the mover, which needs what would stop a stride,
    /// and skips floors, because a probe from inside a room always meets the
    /// floor it stands on. The camera needs every surface between the
    /// character and the eye, floors and ceilings included, because the most
    /// common surface a third-person camera ends up behind is a roof.
    pub fn ray(&self, origin: [f32; 3], dir: [f32; 3], max: f32) -> Option<f32> {
        let end = [
            origin[0] + dir[0] * max,
            origin[1] + dir[1] * max,
            origin[2] + dir[2] * max,
        ];
        // The same three-axis rejection `wall` documents: `cell_of` clamps, so a
        // hull the ray misses entirely would otherwise resolve to its nearest
        // corner cell and test every triangle in it.
        for axis in 0..3 {
            if origin[axis].min(end[axis]) > self.bounds[1][axis]
                || origin[axis].max(end[axis]) < self.bounds[0][axis]
            {
                return None;
            }
        }
        let (col0, row0) = self.cell_of(origin[0].min(end[0]), origin[1].min(end[1]));
        let (col1, row1) = self.cell_of(origin[0].max(end[0]), origin[1].max(end[1]));

        let mut best: Option<f32> = None;
        let mut seen: HashSet<u32> = HashSet::new();
        for row in row0..=row1 {
            for col in col0..=col1 {
                for &triangle in self.cell_slice(col, row) {
                    if !seen.insert(triangle) {
                        continue;
                    }
                    let corners = self.triangle(triangle as usize);
                    let Some(distance) = ray_triangle(origin, dir, &corners) else {
                        continue;
                    };
                    if (0.0..=max).contains(&distance) && best.is_none_or(|b| distance < b) {
                        best = Some(distance);
                    }
                }
            }
        }
        best
    }

    /// The nearest wall a ray meets, as `(distance, outward normal)`.
    ///
    /// "Outward" means facing the ray. A WMO's winding says which side of a
    /// wall is outside, and without this a wall met from inside a room would
    /// push the character further in. Only walls are reported: every probe hits
    /// the floor underfoot, and floors belong to the floor query.
    fn wall(&self, origin: [f32; 3], dir: [f32; 3], max: f32) -> Option<(f32, [f32; 3])> {
        if origin[2] + dir[2] * max < self.bounds[0][2]
            || origin[2].min(origin[2] + dir[2] * max) > self.bounds[1][2]
        {
            return None;
        }
        // The ray is one stride long and covers a few cells, so taking them
        // from its bounding box rather than walking it is simpler and, at this
        // length, no more work.
        let end = [origin[0] + dir[0] * max, origin[1] + dir[1] * max];

        // The same rejection in XY, which the z test does not give.
        // [`Self::cell_of`] clamps, so a ray a hundred yards away resolves to
        // the nearest corner cell and every triangle in it is ray-tested for
        // nothing. With a doodad hull per tree that is thousands of hulls
        // tested twenty times a second, which takes the whole frame.
        // [`Self::floor`] rejects the same way.
        if origin[0].min(end[0]) > self.bounds[1][0]
            || origin[0].max(end[0]) < self.bounds[0][0]
            || origin[1].min(end[1]) > self.bounds[1][1]
            || origin[1].max(end[1]) < self.bounds[0][1]
        {
            return None;
        }
        let (col0, row0) = self.cell_of(origin[0].min(end[0]), origin[1].min(end[1]));
        let (col1, row1) = self.cell_of(origin[0].max(end[0]), origin[1].max(end[1]));

        let mut best: Option<(f32, [f32; 3])> = None;
        let mut seen: HashSet<u32> = HashSet::new();
        for row in row0..=row1 {
            for col in col0..=col1 {
                for &triangle in self.cell_slice(col, row) {
                    if !seen.insert(triangle) {
                        continue;
                    }
                    let corners = self.triangle(triangle as usize);
                    let normal = normal_of(&corners);
                    // A degenerate triangle has no normal, and its zero would
                    // read as perfectly vertical.
                    if normal == [0.0; 3] || normal[2].abs() >= WALL_NORMAL_Z {
                        continue;
                    }
                    let Some(distance) = ray_triangle(origin, dir, &corners) else {
                        continue;
                    };
                    if distance < 0.0 || distance > max {
                        continue;
                    }
                    if best.is_none_or(|(d, _)| distance < d) {
                        // Face the ray, whichever way the triangle is wound.
                        let facing = if dot(normal, dir) > 0.0 {
                            [-normal[0], -normal[1], -normal[2]]
                        } else {
                            normal
                        };
                        best = Some((distance, facing));
                    }
                }
            }
        }
        best
    }
}

/// How many cells one axis of a hull's grid gets.
fn grid_axis(extent: f32) -> usize {
    ((extent / CELL_SIZE).ceil() as usize).clamp(1, GRID_MAX)
}

// ---------------------------------------------------------------------------
// Every loaded placement
// ---------------------------------------------------------------------------

/// The colliders the client currently holds, grouped by the tile that owns
/// them.
///
/// Keyed by tile because a building lives as long as its tile: the renderer
/// loads a 3x3 around the character and despawns what falls outside it, and a
/// collider left behind for a despawned building is an invisible wall.
/// Internally locked, as [`crate::Terrain`] is and for the same reason: the
/// writer is the renderer's main thread and the reader is the session thread,
/// twenty times a second.
#[derive(Default)]
pub struct CollisionWorld {
    inner: RwLock<Inner>,
}

/// Tiles [`CollisionWorld::retain_tiles`] took out, for the caller to drop
/// wherever freeing them costs nothing that matters.
pub struct Retired(Vec<Tile>);

impl Retired {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[derive(Default)]
struct Inner {
    /// Which map these colliders belong to. A far teleport invalidates every
    /// one of them, and tile coordinates repeat across continents, so this is
    /// checked on both the read and the write path rather than left to the
    /// renderer's despawn.
    map_id: Option<u32>,
    tiles: HashMap<(u32, u32), Tile>,
    /// The hulls no tile owns. See [`Objects`].
    objects: Objects,
}

impl Inner {
    /// Every hull the caller's own box test admits, from both populations.
    ///
    /// The only place the two are joined, so none of the five queries can be
    /// written against the tiles alone and miss the game objects.
    fn candidates<'a>(
        &'a self,
        keep: impl Fn(&[[f32; 3]; 2]) -> bool + Copy + 'a,
    ) -> impl Iterator<Item = &'a Arc<Collider>> + 'a {
        self.candidates_of(keep, Surfaces::ALL)
    }

    /// The walk of [`Self::candidates`], narrowed to some of the populations.
    /// See [`Surfaces`]. Separate from [`Self::candidates`] rather than a
    /// parameter on it because every query but one wants all three, and those
    /// five are the mover's and the camera's.
    fn candidates_of<'a>(
        &'a self,
        keep: impl Fn(&[[f32; 3]; 2]) -> bool + Copy + 'a,
        want: Surfaces,
    ) -> impl Iterator<Item = &'a Arc<Collider>> + 'a {
        self.tiles
            .values()
            .flat_map(move |tile| tile.candidates(keep, want))
            .chain(self.objects.candidates(keep, want))
    }

    /// Is there anything at all to test a query against?
    fn nothing_placed(&self) -> bool {
        self.tiles.is_empty() && self.objects.solids.is_empty()
    }
}

/// What one tile holds.
#[derive(Default)]
struct Tile {
    /// The ids held, each with its index into the three vectors below, so a
    /// second insert of the same thing is recognised in O(1) and replaces the
    /// hull rather than being ignored. A scan over the vector would be cheap
    /// for the dozen buildings on a tile and quadratic for its twelve hundred
    /// doodads.
    ids: HashMap<SolidId, usize>,
    solids: Vec<Arc<Collider>>,
    /// Each hull's world-space box, in `solids` order: the value
    /// `Collider::bounds` returns, copied into a contiguous vector for the
    /// scan.
    ///
    /// Every query begins by rejecting hulls on their boxes. Reading the box
    /// through the `Arc` costs a pointer chase into a separately allocated
    /// `Collider` per hull per query, a cache miss each, and that is the whole
    /// cost of a query that rejects everything. With this vector the rejection
    /// is a linear walk of 24 contiguous bytes per hull, and the `Arc` is read
    /// only for a candidate.
    ///
    /// The hull count justifies the second vector: a tile holds a dozen
    /// buildings and up to 3,817 solid doodads once a city's furniture is
    /// counted, the camera casts a ray against all of them every frame, and the
    /// mover tests a floor and three probe heights against them twenty times a
    /// second. Kept in lockstep with `solids` by [`CollisionWorld::insert`],
    /// the only writer of either.
    boxes: Vec<[[f32; 3]; 2]>,
    /// Which population each hull came out of, in `solids` order and kept in
    /// lockstep with it by the same line that writes the box.
    ///
    /// A byte per hull rather than a map, for the reason given on
    /// [`Self::boxes`]: the one caller that reads it filters thousands of hulls
    /// and reads the value beside the box it has just tested. See [`Surfaces`].
    kinds: Vec<Solid>,
    /// The indices into `solids` of the [`Solid::Building`] hulls, ascending.
    ///
    /// [`CollisionWorld::building_floor`] is answered only by a hull with
    /// groups, which on a tile is a `MODF` building, and it is asked for every
    /// unit in the world every frame. Walking `boxes` for it rejects a city
    /// tile's thousands of doodads per unit to find its dozen buildings: 4.5 ms
    /// a frame for 290 units in a debug build. Written by the same two places
    /// as `kinds`.
    buildings: Vec<usize>,
    /// Which `MODF` placements the caller has finished deciding about, by
    /// `unique_id`: hull placed, hull absent, or the file refused.
    ///
    /// This is not the same set as [`Self::ids`] and cannot be derived from
    /// it. A building with no `BoundingTriangles` inserts no hull, so it is
    /// never in `ids`, and a caller waiting for it to appear there would wait
    /// for ever. A placement that was examined and has nothing solid in it is
    /// recorded here as finished.
    ///
    /// Keyed by placement rather than by [`SolidId`] because it describes the
    /// `MODF` row. See [`crate::world::adt::Adt::awaiting_building`], the one
    /// reader.
    settled: HashSet<u32>,
}

impl Tile {
    /// Drop every hull whose id fails `keep`, and the settled mark of every
    /// placement none of the kept hulls belongs to and `keep_settled` refuses.
    /// The three vectors and the index are rebuilt together. Returns how many
    /// hulls went.
    fn retain_hulls(
        &mut self,
        keep: impl Fn(&SolidId) -> bool,
        keep_settled: impl Fn(u32) -> bool,
    ) -> usize {
        let before = self.solids.len();
        let mut kept: Vec<(SolidId, usize)> = self
            .ids
            .iter()
            .filter(|(id, _)| keep(id))
            .map(|(id, index)| (*id, *index))
            .collect();
        if kept.len() == before {
            self.settled.retain(|placement| keep_settled(*placement));
            return 0;
        }
        kept.sort_by_key(|(_, index)| *index);
        let solids = std::mem::take(&mut self.solids);
        let boxes = std::mem::take(&mut self.boxes);
        let kinds = std::mem::take(&mut self.kinds);
        self.ids.clear();
        for (id, index) in kept {
            self.ids.insert(id, self.solids.len());
            self.solids.push(Arc::clone(&solids[index]));
            self.boxes.push(boxes[index]);
            self.kinds.push(kinds[index]);
        }
        self.index_buildings();
        self.settled.retain(|placement| keep_settled(*placement));
        before - self.solids.len()
    }

    /// Rebuild [`Self::buildings`] from `kinds`.
    fn index_buildings(&mut self) {
        self.buildings.clear();
        self.buildings.extend(
            self.kinds.iter().enumerate().filter(|(_, &kind)| kind == Solid::Building).map(|(i, _)| i),
        );
    }

    /// [`Self::candidates`] narrowed to the buildings, walking only their
    /// boxes. See [`Self::buildings`].
    fn building_candidates<'a>(
        &'a self,
        keep: impl Fn(&[[f32; 3]; 2]) -> bool + 'a,
    ) -> impl Iterator<Item = &'a Arc<Collider>> + 'a {
        self.buildings
            .iter()
            .filter_map(move |&i| keep(&self.boxes[i]).then_some(&self.solids[i]))
    }

    /// The hulls whose box the caller's own test admits, as `Arc`s it may then
    /// dereference. `keep` sees only the box.
    fn candidates<'a>(
        &'a self,
        keep: impl Fn(&[[f32; 3]; 2]) -> bool + 'a,
        want: Surfaces,
    ) -> impl Iterator<Item = &'a Arc<Collider>> + 'a {
        self.boxes
            .iter()
            .zip(&self.kinds)
            .zip(&self.solids)
            .filter_map(move |((bounds, &kind), collider)| {
                (want.admits(kind) && keep(bounds)).then_some(collider)
            })
    }
}

/// Which population a hull came out of.
///
/// Every query in this module but one ignores it: the mover treats all hulls
/// alike, and a character standing on a crate stands on it whether or not the
/// crate is a doodad. The exception is a caller choosing a surface to put
/// something on, which must be able to exclude a population. See
/// [`Surfaces`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Solid {
    /// A `MODF` building.
    Building,
    /// A `MDDF` doodad, or a building's own `MODD` furniture.
    ///
    /// One variant and not two, because they are one population everywhere
    /// else: the same pass draws both, the same switch hides both, and a piece
    /// of furniture comes from an M2 just as a tree does.
    Doodad,
    /// A game object the server spawned: a door, a chest, an elevator.
    Object,
}

/// Which populations a query will accept an answer from.
///
/// Every query but one takes [`Surfaces::ALL`]. The exception is a caller
/// placing something in the world, which stands it only on the populations it
/// is drawing: standing something on the roof of a hidden building puts it
/// where the user cannot see it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Surfaces {
    pub buildings: bool,
    pub doodads: bool,
    pub objects: bool,
}

impl Surfaces {
    /// Every population, which is what the mover and the camera ask for.
    pub const ALL: Surfaces = Surfaces {
        buildings: true,
        doodads: true,
        objects: true,
    };

    /// No population: the ground only.
    pub const NONE: Surfaces = Surfaces {
        buildings: false,
        doodads: false,
        objects: false,
    };

    /// Is a hull of this kind one this query may be answered by?
    pub fn admits(&self, kind: Solid) -> bool {
        match kind {
            Solid::Building => self.buildings,
            Solid::Doodad => self.doodads,
            Solid::Object => self.objects,
        }
    }

    /// Whether this admits no population. A query with such a filter would
    /// walk every hull in the world and reject all of them.
    pub fn nothing(&self) -> bool {
        *self == Surfaces::NONE
    }
}

/// Does a segment's own box overlap a hull's, on all three axes?
///
/// The same rejection `Collider::ray` opens with, hoisted out so it can be
/// asked of [`Tile::boxes`] without touching the hull.
fn segment_meets(bounds: &[[f32; 3]; 2], from: [f32; 3], to: [f32; 3]) -> bool {
    (0..3).all(|axis| {
        from[axis].min(to[axis]) <= bounds[1][axis] && from[axis].max(to[axis]) >= bounds[0][axis]
    })
}

/// What one collider is, so the same thing inserted twice is held once.
///
/// Not a bare `u32`. A `MODD` spawn (furniture inside a building) has no id of
/// its own and carries its building's, so Stormwind's three thousand chairs
/// share one number, and a dedup on that number keeps one chair and drops the
/// rest. Two spawns therefore need a second field to tell them apart.
///
/// vmangos uses the same key: its extractor names every placed object
/// `GenerateUniqueObjectId(clientId, clientDoodadId)`, where the doodad id is
/// `0` for anything an `MDDF` or `MODF` row placed directly and the spawn's
/// ordinal within its building otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SolidId {
    /// The `MDDF` or `MODF` unique id of the placement this came from.
    pub placement: u32,
    /// Which spawn inside that placement, 1-based. `0` is the placement itself.
    pub spawn: u32,
}

// ---------------------------------------------------------------------------
// Game objects the server places
// ---------------------------------------------------------------------------

/// `GO_STATE_READY` (`GameObjectDefines.h`), the reset state: a door shut, a
/// portcullis down, a chest with its lid on.
pub const GO_STATE_READY: u8 = 1;

/// Whether a game object in this state blocks movement.
///
/// `GO_STATE_ACTIVE` (0) is the used state, a door standing open, and
/// `GO_STATE_ACTIVE_ALTERNATIVE` (2) is a second used state; only
/// `GO_STATE_READY` is solid. vmangos makes the same test in
/// `GameObject::UpdateCollisionState`:
///
/// ```text
/// bool enabled = GetGoType() == GAMEOBJECT_TYPE_CHEST
///     ? getLootState() == GO_READY : GetGoState() == GO_STATE_READY;
/// m_model->enable(enabled);
/// ```
///
/// `None` means open. vmangos' `Object::_SetCreateBits` omits a field whose
/// value is zero, so a game object that sends no state has state 0, which is
/// `GO_STATE_ACTIVE`. See
/// `vale_protocol::state::objects::Entity::game_object_state`, which returns
/// this `None`.
///
/// One approximation: the chest arm above reads a loot state that is not sent
/// to the client, so a chest is judged by its go state like any other game
/// object. The difference is a looted chest that has not changed state, which
/// stays solid here, as it does in the 1.12.1 client.
pub fn game_object_is_solid(state: Option<u8>) -> bool {
    state == Some(GO_STATE_READY)
}

/// The model-to-world matrix for something the server placed, column-major.
///
/// `position` is world-space and `facing` is the server's orientation, in
/// radians counter-clockwise about +Z with zero pointing north. Every entity
/// in this client is drawn from that pair. The matrix is `T · Rz(facing) · S`
/// with no other term, and in particular no 180° term: that belongs to
/// [`adt::placement_matrix`], where it is the internal-to-world frame flip an
/// `MDDF` row's encoding needs, and a server-placed object's position and
/// facing are not in that frame.
///
/// vmangos builds the same matrix for its dynamic tree in
/// `GameObjectModel::initialize`:
/// `G3D::Matrix3::fromEulerAnglesZYX(pGo->GetOrientation(), 0, 0)` about
/// `Vector3(GetPositionX(), GetPositionY(), GetPositionZ())`, scaled by
/// `GetObjectScale()`. The server also rotates about Z only, and neither it
/// nor this client collides against the four-float `GAMEOBJECT_ROTATION`
/// quaternion sent beside the facing.
///
/// [`adt::placement_matrix`]: crate::world::adt::placement_matrix
pub fn object_matrix(position: [f32; 3], facing: f32, scale: f32) -> [f32; 16] {
    let (s, c) = facing.sin_cos();
    let mut out = [0.0f32; 16];
    // Columns 0..2 are the scaled basis of Rz(facing), column 3 the position.
    out[0] = c * scale;
    out[1] = s * scale;
    out[4] = -s * scale;
    out[5] = c * scale;
    out[10] = scale;
    out[12] = position[0];
    out[13] = position[1];
    out[14] = position[2];
    out[15] = 1.0;
    out
}

/// The placement one game object's hull was built for.
///
/// Held beside the hull so a caller can check whether the hull is current,
/// rather than re-placing every door in view every frame: `Collider::place`
/// walks and re-indexes every triangle, and a portcullis has many. Compared
/// exactly, the way `place_entities` compares a `Transform`: a game object
/// that moved a millimetre needs a new hull, and an epsilon would save only a
/// comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectPlacement {
    pub position: [f32; 3],
    /// The server's own orientation, in radians.
    pub facing: f32,
    /// `OBJECT_FIELD_SCALE_X`.
    pub scale: f32,
}

/// The hulls no tile owns: the game objects the server spawns.
///
/// Three vectors in lockstep and two maps, for these reasons:
///
/// * `placed` is every game object decided about, hull or no hull, for the
///   same reason as [`Tile::settled`]. A model with no `BoundingTriangles`
///   inserts nothing, so it is never in `index`, and a caller that read that
///   as "not decided yet" would re-read the M2, re-place a hull and re-insert
///   nothing, once per frame per campfire.
/// * `index` holds the ones that have a hull, so a removal is O(1) rather than
///   a scan. A door opening and shutting is the common case.
/// * `guids` exists only so `swap_remove` can re-point the entry it moved.
#[derive(Default)]
struct Objects {
    placed: HashMap<u64, ObjectPlacement>,
    index: HashMap<u64, usize>,
    guids: Vec<u64>,
    /// In `solids` order, for the reason given on [`Tile::boxes`].
    boxes: Vec<[[f32; 3]; 2]>,
    solids: Vec<Arc<Collider>>,
}

impl Objects {
    fn candidates<'a>(
        &'a self,
        keep: impl Fn(&[[f32; 3]; 2]) -> bool + 'a,
        want: Surfaces,
    ) -> impl Iterator<Item = &'a Arc<Collider>> + 'a {
        // No kinds vector here: every hull this population holds is a game
        // object, so the whole walk is admitted or none of it is.
        let admitted = want.objects;
        self.boxes
            .iter()
            .zip(&self.solids)
            .filter_map(move |(bounds, collider)| (admitted && keep(bounds)).then_some(collider))
    }

    /// Drop this object's hull, keeping the three vectors and the index in step.
    /// The placement record is not touched: the two callers need opposite
    /// things from it, and each updates it itself.
    fn drop_hull(&mut self, guid: u64) {
        let Some(at) = self.index.remove(&guid) else {
            return;
        };
        self.guids.swap_remove(at);
        self.boxes.swap_remove(at);
        self.solids.swap_remove(at);
        // The entry `swap_remove` moved into the hole now has a different
        // index. Nothing moved when the hole was the last slot.
        if let Some(&moved) = self.guids.get(at) {
            self.index.insert(moved, at);
        }
    }

    fn clear(&mut self) {
        self.placed.clear();
        self.index.clear();
        self.guids.clear();
        self.boxes.clear();
        self.solids.clear();
    }
}

impl SolidId {
    /// A building or a doodad, placed by its own `MODF` / `MDDF` row.
    pub fn placement(unique_id: u32) -> SolidId {
        SolidId {
            placement: unique_id,
            spawn: 0,
        }
    }

    /// One `MODD` spawn inside the building `unique_id`, by its ordinal.
    pub fn spawn(unique_id: u32, ordinal: u32) -> SolidId {
        SolidId {
            placement: unique_id,
            spawn: ordinal.saturating_add(1),
        }
    }
}

impl CollisionWorld {
    pub fn new() -> CollisionWorld {
        CollisionWorld::default()
    }

    /// Add a placed hull (a building, a doodad or a piece of furniture),
    /// dropping everything on a different map first.
    pub fn insert(
        &self,
        map_id: u32,
        tile: (u32, u32),
        id: SolidId,
        kind: Solid,
        collider: Arc<Collider>,
    ) {
        let mut inner = self.write();
        if inner.map_id != Some(map_id) {
            inner.tiles.clear();
            inner.map_id = Some(map_id);
        }
        let placed = inner.tiles.entry(tile).or_default();
        // The placement is settled when its hull arrives, not when the caller
        // decides to place one, so this is here rather than at the call site.
        // A building's `Collider::place` runs on the compute pool and
        // Stormwind's takes 27.6 ms; a caller that marked the placement
        // settled when it started would leave that long a window in which the
        // placement counts as settled and has no hull.
        placed.settled.insert(id.placement);
        match placed.ids.get(&id) {
            // The same id arriving again is the same placement placed again:
            // a building read from a re-read tile after a move, or a new
            // placement minted under an id an undone one had. The new hull
            // replaces the old one in place. Ignoring it would leave the old
            // hull at the old position and no hull at the new one.
            Some(&index) => {
                placed.boxes[index] = collider.bounds();
                placed.solids[index] = collider;
                if placed.kinds[index] != kind {
                    placed.kinds[index] = kind;
                    placed.index_buildings();
                }
            }
            None => {
                // The box and the kind are pushed in lockstep with the hull:
                // `Tile::boxes` and `Tile::kinds` are valid only while they
                // have the same length and order as `solids`, and this and
                // `Tile::retain_hulls` are the only places any of the three
                // is written.
                if kind == Solid::Building {
                    placed.buildings.push(placed.solids.len());
                }
                placed.ids.insert(id, placed.solids.len());
                placed.boxes.push(collider.bounds());
                placed.kinds.push(kind);
                placed.solids.push(collider);
            }
        }
    }

    /// Move every hull of one placement, the building's or the doodad's own
    /// and every furniture spawn inside it, by `delta`, a column-major world
    /// space matrix. Returns how many hulls moved. For a host that moves a
    /// placement while its tile is on screen. The transform is written live,
    /// and a hull left at the old position is an invisible wall there and
    /// leaves the placement with no hull.
    pub fn move_placement(&self, map_id: u32, unique_id: u32, delta: &[f32; 16]) -> usize {
        let mut inner = self.write();
        if inner.map_id != Some(map_id) {
            return 0;
        }
        let mut moved = 0;
        for tile in inner.tiles.values_mut() {
            for (id, &index) in tile.ids.iter() {
                if id.placement != unique_id {
                    continue;
                }
                let collider = Arc::new(tile.solids[index].moved(delta));
                tile.boxes[index] = collider.bounds();
                tile.solids[index] = collider;
                moved += 1;
            }
        }
        moved
    }

    /// Drop every hull of one placement, wherever it is held, and its
    /// settled mark. Returns how many hulls went.
    #[cfg(test)]
    pub(crate) fn remove_placement(&self, map_id: u32, unique_id: u32) -> usize {
        let mut inner = self.write();
        if inner.map_id != Some(map_id) {
            return 0;
        }
        inner
            .tiles
            .values_mut()
            .map(|tile| tile.retain_hulls(|id| id.placement != unique_id, |p| p != unique_id))
            .sum()
    }

    /// Drop every hull under `tile` whose placement is not in `keep`, and the
    /// settled marks of the same placements. Returns how many hulls went.
    ///
    /// For a tile read again: the hulls of its placements were inserted when
    /// it first arrived and a re-read inserts the current ones over them, so
    /// a placement the tile no longer names would keep its hull until the
    /// tile walked out of range. Called with the ids the re-read tile's
    /// placement lists carry, before those are spawned.
    pub fn retain_placements(&self, map_id: u32, tile: (u32, u32), keep: &HashSet<u32>) -> usize {
        // Under a read lock first: the client's own streaming calls this on
        // every tile arrival, where the coordinate holds nothing yet.
        {
            let inner = self.read();
            let empty = inner.map_id != Some(map_id)
                || inner
                    .tiles
                    .get(&tile)
                    .is_none_or(|t| t.ids.is_empty() && t.settled.is_empty());
            if empty {
                return 0;
            }
        }
        let mut inner = self.write();
        match inner.tiles.get_mut(&tile) {
            Some(t) => t.retain_hulls(|id| keep.contains(&id.placement), |p| keep.contains(&p)),
            None => 0,
        }
    }

    /// Record that a placement has been decided about and has no hull: the
    /// file carries no `BoundingTriangles`, or does not load.
    ///
    /// See [`Tile::settled`]. Without this, a point inside a hull-less
    /// building's `MODF` box waits for a hull that never arrives, and a mover
    /// waiting on it never lands.
    pub fn settle(&self, map_id: u32, tile: (u32, u32), placement: u32) {
        let mut inner = self.write();
        if inner.map_id != Some(map_id) {
            inner.tiles.clear();
            inner.map_id = Some(map_id);
        }
        inner.tiles.entry(tile).or_default().settled.insert(placement);
    }

    /// Whether this `MODF` placement has been decided about. See
    /// [`Tile::settled`].
    ///
    /// Checked across every resident tile rather than at one coordinate,
    /// because a building belongs to the tile its origin stands on and a
    /// character under it can be on the next tile. At most a dozen hash
    /// lookups, once per floor query.
    pub fn settled(&self, map_id: u32, placement: u32) -> bool {
        let inner = self.read();
        inner.map_id == Some(map_id)
            && inner.tiles.values().any(|t| t.settled.contains(&placement))
    }

    /// Place a game object's hull, replacing whatever this GUID had before.
    ///
    /// `hull` is `None` for an object that has been decided about and has
    /// nothing solid in it (a campfire, a fishing bobber, a light). That covers
    /// most game objects and is a finished answer, not a failure. See
    /// [`Objects::placed`] for why it is recorded rather than inferred.
    ///
    /// One call rather than the two of [`Self::insert`] and [`Self::settle`],
    /// because unlike for a `MODF` row the caller has the model loaded when it
    /// decides, and so knows both the placement and the hull at once.
    pub fn place_object(
        &self,
        map_id: u32,
        guid: u64,
        at: ObjectPlacement,
        hull: Option<Arc<Collider>>,
    ) {
        let mut inner = self.write();
        if inner.map_id != Some(map_id) {
            inner.tiles.clear();
            inner.objects.clear();
            inner.map_id = Some(map_id);
        }
        inner.objects.drop_hull(guid);
        inner.objects.placed.insert(guid, at);
        if let Some(collider) = hull {
            let objects = &mut inner.objects;
            objects.index.insert(guid, objects.solids.len());
            objects.guids.push(guid);
            // The box is pushed in lockstep with the hull, as in `insert`.
            objects.boxes.push(collider.bounds());
            objects.solids.push(collider);
        }
    }

    /// What this game object's hull was placed for, or `None` if it has
    /// never been decided about on this map.
    ///
    /// The caller uses it to check whether its hull is current: a door that has
    /// not moved returns the placement it was given and needs no work.
    pub fn object_placement(&self, map_id: u32, guid: u64) -> Option<ObjectPlacement> {
        let inner = self.read();
        if inner.map_id != Some(map_id) {
            return None;
        }
        inner.objects.placed.get(&guid).copied()
    }

    /// Whether this game object has a hull in the world, as opposed to only a
    /// placement on record.
    ///
    /// The two are different states and [`Self::place_object`] writes both in
    /// one call, so only this query tells them apart. A `.wmo` transport out of
    /// hull-building range, a model with no `BoundingTriangles` and a model
    /// that failed to load all have a placement and no hull.
    ///
    /// The difference decides whether a passenger has stepped ashore. A
    /// character who is aboard and not standing on any hull has walked off the
    /// deck only if the deck has a hull. With no hull there is no evidence
    /// either way, and treating it as a step ashore strands the passenger. See
    /// `vale_protocol::socket::session::World::platform_hulled`.
    pub fn object_hulled(&self, map_id: u32, guid: u64) -> bool {
        let inner = self.read();
        inner.map_id == Some(map_id) && inner.objects.index.contains_key(&guid)
    }

    /// Forget every game object not in `live`.
    ///
    /// Called with the ones that are both in view and solid, so this removes
    /// both a door that has opened and an object the server no longer
    /// describes. In both cases the GUID has no hull now, and handling them on
    /// one path means an open door and a despawned one cannot be treated
    /// differently by mistake.
    pub fn retain_objects(&self, live: &HashSet<u64>) {
        // Under a read lock first, for the reason given in `retain_tiles`: in
        // the steady state nothing has changed, and taking the write lock
        // every frame would block the session thread's floor query.
        if self.read().objects.placed.keys().all(|g| live.contains(g)) {
            return;
        }
        let mut inner = self.write();
        let gone: Vec<u64> = inner
            .objects
            .placed
            .keys()
            .copied()
            .filter(|g| !live.contains(g))
            .collect();
        for guid in gone {
            inner.objects.drop_hull(guid);
            inner.objects.placed.remove(&guid);
        }
    }

    /// Forget every tile not in `live`. Called with the tiles the renderer
    /// still has, so a building goes when the ground it stands on does.
    ///
    /// The forgotten tiles are returned rather than dropped here. A tile's
    /// hulls are every triangle of its buildings and doodads, and freeing a
    /// block of them took up to 19 ms on the caller's thread, which is the main
    /// thread; the caller drops [`Retired`] on a background task instead. The
    /// write lock is held only to move them out.
    pub fn retain_tiles(&self, live: &HashSet<(u32, u32)>) -> Retired {
        // Checked under a read lock first, because the caller is a Bevy system
        // running every frame and in the steady state nothing has changed.
        // Taking the write lock sixty times a second would block the session
        // thread's floor query.
        if self.read().tiles.keys().all(|k| live.contains(k)) {
            return Retired(Vec::new());
        }
        let mut inner = self.write();
        let gone: Vec<(u32, u32)> = inner
            .tiles
            .keys()
            .copied()
            .filter(|coord| !live.contains(coord))
            .collect();
        Retired(gone.into_iter().filter_map(|coord| inner.tiles.remove(&coord)).collect())
    }

    pub fn clear(&self) {
        let mut inner = self.write();
        inner.tiles.clear();
        inner.objects.clear();
        inner.map_id = None;
    }

    /// Hulls held, and solid triangles across them. For the HUD.
    pub fn counts(&self) -> (usize, usize) {
        let inner = self.read();
        let hulls = inner.tiles.values().map(|t| t.solids.len()).sum::<usize>()
            + inner.objects.solids.len();
        let triangles = inner
            .tiles
            .values()
            .flat_map(|t| &t.solids)
            .chain(&inner.objects.solids)
            .map(|c| c.triangle_count())
            .sum();
        (hulls, triangles)
    }

    /// Every hull within `radius` of `centre`, passed to `visit` one at a time,
    /// for a debug overlay that draws the surfaces the character collides
    /// with.
    ///
    /// The only function in this module that serves drawing rather than a
    /// query. It is here rather than in the renderer because the box
    /// rejection, the two populations and the map check belong to this file,
    /// and a second copy of them in `vale-client` could diverge. The renderer
    /// receives triangles.
    ///
    /// Bounded only by the caller's radius. A loaded world holds ~3,800 hulls
    /// and some are a whole cathedral, so an unbounded visit draws a million
    /// lines in one frame; the overlay is meant to show the ground around the
    /// character.
    ///
    /// A visitor rather than an iterator because the hulls live behind an
    /// `RwLock` this must not hand out.
    pub fn hulls_near(
        &self,
        map_id: u32,
        centre: [f32; 3],
        radius: f32,
        mut visit: impl FnMut(&Collider),
    ) {
        let inner = self.read();
        if inner.map_id != Some(map_id) {
            return;
        }
        let keep = move |bounds: &[[f32; 3]; 2]| {
            (0..3).all(|axis| {
                bounds[0][axis] - radius <= centre[axis] && centre[axis] <= bounds[1][axis] + radius
            })
        };
        for collider in inner.candidates(keep) {
            visit(collider);
        }
    }

    /// The highest building surface under `(x, y)` at or below `ceiling`.
    pub fn floor(&self, map_id: u32, x: f32, y: f32, ceiling: f32) -> Option<f32> {
        self.surface(map_id, x, y, ceiling).map(|(z, _)| z)
    }

    /// The surface [`Self::floor`] returns, with its slope. See
    /// [`Collider::surface`]. One walk of the candidate hulls returns both, so
    /// the height and the slope always come from the same building.
    pub fn surface(&self, map_id: u32, x: f32, y: f32, ceiling: f32) -> Option<(f32, [f32; 3])> {
        let inner = self.read();
        if inner.map_id != Some(map_id) {
            return None;
        }
        let mut best: Option<(f32, [f32; 3])> = None;
        // The box test `Collider::floor` begins with, made against the tile's
        // box vector so a hull the point misses is never dereferenced. The
        // comparisons and the result are the same. See [`Tile::boxes`].
        let over = |bounds: &[[f32; 3]; 2]| {
            x >= bounds[0][0]
                && x <= bounds[1][0]
                && y >= bounds[0][1]
                && y <= bounds[1][1]
                && ceiling >= bounds[0][2]
        };
        for collider in inner.candidates(over) {
            if let Some((z, n)) = collider.surface(x, y, ceiling) {
                if best.is_none_or(|(b, _)| z > b) {
                    best = Some((z, n));
                }
            }
        }
        best
    }

    /// The building floor under a point, and its group's `MOGP` flags. See
    /// [`Collider::building_floor`]. The highest across every building hull
    /// that has one, so over a courtyard spanned by a bridge the answer is the
    /// bridge.
    ///
    /// Only a hull that carries groups answers, which is every `MODF` building
    /// and nothing else: a crate on a street is a surface to stand on but does
    /// not say whether the street is outdoors.
    pub fn building_floor(&self, map_id: u32, x: f32, y: f32, ceiling: f32) -> Option<(f32, u32)> {
        let inner = self.read();
        if inner.map_id != Some(map_id) {
            return None;
        }
        let over = |bounds: &[[f32; 3]; 2]| {
            x >= bounds[0][0]
                && x <= bounds[1][0]
                && y >= bounds[0][1]
                && y <= bounds[1][1]
                && ceiling >= bounds[0][2]
        };
        let mut best: Option<(f32, u32)> = None;
        // A tile's doodads have no groups and cannot answer, so only its
        // buildings are walked. A game object can be a WMO, so they all are.
        let buildings = inner.tiles.values().flat_map(|tile| tile.building_candidates(over));
        for collider in buildings.chain(inner.objects.candidates(over, Surfaces::ALL)) {
            if let Some((z, flags)) = collider.building_floor(x, y, ceiling) {
                if best.is_none_or(|(b, _)| z > b) {
                    best = Some((z, flags));
                }
            }
        }
        best
    }

    /// Which game object's hull is directly under this point, and how high
    /// its surface is. `None` where the highest thing underfoot is terrain or
    /// a building.
    ///
    /// The only query in this file that returns which hull rather than a
    /// surface. A character standing on a moving platform needs to know which
    /// platform, because the movement packet carries `MOVEFLAG_ONTRANSPORT`, a
    /// guid, and a position in that object's frame. See
    /// `vale_protocol::state::movement::Ferry`.
    ///
    /// It tests only the game objects: a tile's buildings and terrain cannot
    /// move, so a surface either of them provides is not a platform. That also
    /// makes it cheap, since the object list is a few dozen entries where a
    /// city tile's hull list is thousands.
    ///
    /// It returns a door or a chest as readily as a lift, because nothing here
    /// knows which objects move. The caller decides: the session asks the
    /// object manager whether the guid carries `UPDATEFLAG_TRANSPORT`, the only
    /// data that marks an object as a transport.
    pub fn object_under(&self, map_id: u32, x: f32, y: f32, ceiling: f32) -> Option<(u64, f32)> {
        let inner = self.read();
        if inner.map_id != Some(map_id) {
            return None;
        }
        let objects = &inner.objects;
        let mut best: Option<(u64, f32)> = None;
        for (n, (bounds, collider)) in objects.boxes.iter().zip(&objects.solids).enumerate() {
            if x < bounds[0][0] || x > bounds[1][0] || y < bounds[0][1] || y > bounds[1][1] {
                continue;
            }
            if ceiling < bounds[0][2] {
                continue;
            }
            let Some(z) = collider.floor(x, y, ceiling) else {
                continue;
            };
            if best.is_none_or(|(_, b)| z > b) {
                // `guids` runs parallel to `solids`; `drop_hull` keeps them
                // parallel by applying `swap_remove` to all three vectors.
                best = Some((objects.guids[n], z));
            }
        }
        best
    }

    /// [`Self::floor`] for many points at once that share one box and one
    /// ceiling.
    ///
    /// The saving is the candidate scan, not the lock. Every query here begins
    /// by walking [`Tile::boxes`] (up to 3,817 hulls on a city tile, nine tiles
    /// loaded) and rejecting the hulls the point misses. Point by point, a
    /// decal grid of eighty-one samples repeats that walk eighty-one times for
    /// a footprint a couple of yards across, and every walk admits the same one
    /// or two hulls. Here the walk runs once, against the grid's box, and each
    /// point is then tested against the hulls it admitted.
    ///
    /// `points` are `(x, y)` in world coordinates and the results are written
    /// parallel to them. A point with nothing over it gets `None`, as in the
    /// single-point form.
    pub fn floor_batch(
        &self,
        map_id: u32,
        points: &[(f32, f32)],
        ceiling: f32,
        out: &mut Vec<Option<f32>>,
    ) {
        out.clear();
        out.resize(points.len(), None);
        let inner = self.read();
        if inner.map_id != Some(map_id) || points.is_empty() {
            return;
        }
        let (mut lo, mut hi) = ((f32::MAX, f32::MAX), (f32::MIN, f32::MIN));
        for &(x, y) in points {
            lo = (lo.0.min(x), lo.1.min(y));
            hi = (hi.0.max(x), hi.1.max(y));
        }
        let over = |bounds: &[[f32; 3]; 2]| {
            hi.0 >= bounds[0][0]
                && lo.0 <= bounds[1][0]
                && hi.1 >= bounds[0][1]
                && lo.1 <= bounds[1][1]
                && ceiling >= bounds[0][2]
        };
        for collider in inner.candidates(over) {
            for (&(x, y), slot) in points.iter().zip(out.iter_mut()) {
                if let Some(z) = collider.floor(x, y, ceiling) {
                    if slot.is_none_or(|b| z > b) {
                        *slot = Some(z);
                    }
                }
            }
        }
    }

    /// How far along `from` -> `to` the first solid surface is, as a fraction of
    /// that segment, or `None` for a clear line.
    ///
    /// Used by the camera only. A fraction rather than a point because the
    /// caller stops short of the hit by a margin, and a fraction can be
    /// reduced directly.
    pub fn ray(&self, map_id: u32, from: [f32; 3], to: [f32; 3]) -> Option<f32> {
        self.ray_of(map_id, from, to, Surfaces::ALL)
    }

    /// [`Self::ray`] narrowed to some of the populations. See [`Surfaces`].
    ///
    /// The camera needs any surface between the eye and the character. A
    /// caller placing something in the world needs the surface it will stand
    /// the thing on, and that surface must belong to a population that is
    /// being drawn: with the buildings hidden, an object placed on a roof
    /// would be out of sight.
    pub fn ray_of(
        &self,
        map_id: u32,
        from: [f32; 3],
        to: [f32; 3],
        want: Surfaces,
    ) -> Option<f32> {
        if want.nothing() {
            return None;
        }
        let inner = self.read();
        if inner.map_id != Some(map_id) || inner.nothing_placed() {
            return None;
        }
        let delta = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
        let length = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        if length < 1.0e-4 {
            return None;
        }
        let dir = [delta[0] / length, delta[1] / length, delta[2] / length];
        let mut best: Option<f32> = None;
        let meets = |bounds: &[[f32; 3]; 2]| segment_meets(bounds, from, to);
        for collider in inner.candidates_of(meets, want) {
            if let Some(d) = collider.ray(from, dir, length) {
                if best.is_none_or(|b| d < b) {
                    best = Some(d);
                }
            }
        }
        best.map(|d| d / length)
    }

    /// Where a stride from `from` to `to` ends.
    ///
    /// Only the horizontal part is decided here; the vertical is decided by
    /// [`CollisionWorld::floor`]. Motion is cancelled, never reversed: each
    /// blocking wall removes the component of the stride heading into it and
    /// keeps the component along it, so a character slides along a wall rather
    /// than sticking to it, and a character inside geometry can always walk
    /// back out. Three walls in one stride is a corner, and the character
    /// stands still.
    pub fn step(&self, map_id: u32, from: [f32; 3], to: [f32; 3]) -> [f32; 3] {
        let inner = self.read();
        if inner.map_id != Some(map_id) || inner.nothing_placed() {
            return to;
        }
        let mut motion = [to[0] - from[0], to[1] - from[1]];
        for attempt in 0..=CANCELS {
            let length = (motion[0] * motion[0] + motion[1] * motion[1]).sqrt();
            if length < 1.0e-4 {
                return [from[0], from[1], to[2]];
            }
            let dir = [motion[0] / length, motion[1] / length, 0.0];
            let Some(normal) = wall_ahead(&inner, from, dir, length + PLAYER_RADIUS) else {
                break;
            };
            // The cancels are spent and something is still in the way: stand
            // still. Applying the remaining motion instead would walk the
            // character out through the corner they are wedged into. The loop
            // runs one pass more than there are cancels so that the last
            // cancel is checked rather than assumed to have worked.
            //
            // Standing still cannot trap the character: motion is cancelled
            // and never reversed, so every direction out of the corner is
            // still available in full on the next stride.
            if attempt == CANCELS {
                return [from[0], from[1], to[2]];
            }
            let into = motion[0] * normal[0] + motion[1] * normal[1];
            // Facing the wall but travelling along it: nothing to cancel, and
            // cancelling zero would loop.
            if into >= 0.0 {
                break;
            }
            motion[0] -= normal[0] * into;
            motion[1] -= normal[1] * into;
        }
        [from[0] + motion[0], from[1] + motion[1], to[2]]
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, Inner> {
        self.inner.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, Inner> {
        self.inner.write().unwrap_or_else(|e| e.into_inner())
    }
}

/// The nearest wall normal any probe meets along `dir`.
///
/// A single ray along the centre axis at each of [`PROBE_HEIGHTS`] stops a
/// stride only when something is directly ahead of the character's centre.
/// It misses everything a shoulder would hit: a fence post, a tree trunk, the
/// jamb of a doorway approached at an angle, the corner of a crate, and a wall
/// met so obliquely that the centre ray leaves through its far edge before the
/// body arrives. Extra heights do not help, because the gap is sideways.
///
/// The sweep therefore uses [`PROBE_LANES`]: the centre line and one line on
/// each side at [`PLAYER_RADIUS`], perpendicular to the stride. Three parallel
/// rays at each height approximate the swept box of a body a yard wide.
///
/// The hulls are scanned once for the whole sweep, not once per ray. The rays
/// differ only in `z` and in a sideways offset, so one box covering all of
/// them rejects for all of them. That keeps the scan over a city's 3,817 hulls
/// to one pass over the tile's contiguous box copies, with no pointer chase
/// per hull (see [`Tile::boxes`]). Each hull that passes is then tested with
/// every ray.
///
/// ## The most head-on wall wins, not the nearest
///
/// With three lanes the ranking decides the result. A shoulder ray passing a
/// barrel meets the barrel's side, whose outward normal is nearly
/// perpendicular to the stride; the centre ray meets its front. Ranked by
/// distance, the side can win, and then [`CollisionWorld::step`] has nothing
/// to cancel, because the component of the stride heading into a wall it runs
/// along is zero. Its `into >= 0.0` arm treats that as travelling along the
/// wall, breaks, and applies the whole stride, so the character walks through
/// the front face.
///
/// `vale collision <map> <tx> <ty>` counts the doodad hulls' vertical faces
/// that stop a stride walked straight into them. Ranked by distance,
/// Ironforge's tile blocks 43,976 of 43,985 and Stormwind's 76,303 of 76,309;
/// ranked by how head-on the wall is, both block all of them. Without this
/// ranking the three-lane sweep blocks fewer faces than a single centre ray.
///
/// Distance breaks a tie, so of two walls with the same orientation the
/// nearer one stops the stride.
fn wall_ahead(inner: &Inner, from: [f32; 3], dir: [f32; 3], distance: f32) -> Option<[f32; 3]> {
    // `(into, distance, normal)`, smallest `into` first. Every normal is
    // already flipped to face the ray, so `into` is at most zero and the most
    // negative is the most opposed.
    let mut best: Option<(f32, f32, [f32; 3])> = None;
    let (low, high) = PROBE_HEIGHTS
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &h| (lo.min(h), hi.max(h)));
    // Perpendicular to the stride, in the ground plane: the axis the lanes are
    // spread along.
    let side = [-dir[1], dir[0]];
    let span_from = [from[0], from[1], from[2] + low];
    let span_to = [
        from[0] + dir[0] * distance,
        from[1] + dir[1] * distance,
        from[2] + high + dir[2] * distance,
    ];
    // The rejection tests the segment against the hull's box grown by the
    // sweep's half-width. The growth goes on the hull's box, not on the
    // segment's ends: `segment_meets` takes the endpoints' axis-aligned
    // extent, so moving one end out and the other in shrinks that extent on
    // every axis the stride travels backwards along. With the endpoints
    // offset, Ironforge's doodad hulls stopped 43,787 of 43,985 vertical faces
    // instead of 43,979, because the hull holding the face was rejected before
    // any ray was cast at it. Ironforge's tile: 5 us a floor+step pair with the
    // endpoints offset, 11 us with the box grown; Stormwind's, over 3,726
    // hulls, 17 us against 14 for a single lane. Both against a 50 ms mover
    // tick.
    let grow = |bounds: &[[f32; 3]; 2]| {
        [
            [
                bounds[0][0] - PLAYER_RADIUS,
                bounds[0][1] - PLAYER_RADIUS,
                bounds[0][2],
            ],
            [
                bounds[1][0] + PLAYER_RADIUS,
                bounds[1][1] + PLAYER_RADIUS,
                bounds[1][2],
            ],
        ]
    };
    let meets = |bounds: &[[f32; 3]; 2]| segment_meets(&grow(bounds), span_from, span_to);
    for collider in inner.candidates(meets) {
        for lane in PROBE_LANES {
            for height in PROBE_HEIGHTS {
                let origin = [
                    from[0] + side[0] * lane,
                    from[1] + side[1] * lane,
                    from[2] + height,
                ];
                if let Some((d, normal)) = collider.wall(origin, dir, distance) {
                    let into = dot(normal, dir);
                    if best.is_none_or(|(bi, bd, _)| (into, d) < (bi, bd)) {
                        best = Some((into, d, normal));
                    }
                }
            }
        }
    }
    best.map(|(_, _, normal)| normal)
}

// ---------------------------------------------------------------------------
// Where a ray meets the floor
// ---------------------------------------------------------------------------

/// How far apart the samples are, in yards.
///
/// A ground-target ray is resolved by marching along it, not by intersecting
/// it, because the two surfaces it can land on are queried differently: the
/// terrain is a heightfield with no triangles (see
/// [`crate::Terrain::height_at`]) and a building is a hull behind a lock. A
/// march uses the one query both support, a floor height at a point.
///
/// Half a yard is under a character's width, and the refinement below reduces
/// the residual to a few centimetres, well inside the yard and a quarter of
/// slack the server's range check allows.
const GROUND_STEP: f32 = 0.5;

/// How fast the step grows with distance, as a reciprocal: the step at `t` is
/// `t / GROUND_STEP_SPREAD` once that exceeds [`GROUND_STEP`].
///
/// A fixed half-yard step suits distances near the camera and is needless
/// precision a thousand yards out, where half a yard is a fraction of a screen
/// pixel. At a 90-degree horizontal field of view over 1,280 pixels one pixel
/// subtends about 0.0012 radians, so a step of `t / 300` is roughly two pixels
/// at any distance. The step size decides only whether the march tunnels
/// through a ridge, a ridge two pixels wide is not a target, and the bisection
/// below provides the precision.
///
/// 300 rather than a rounder number so that `t / 300` reaches [`GROUND_STEP`]
/// at 150 yards, the range the march was previously limited to. Everything
/// inside 150 yards is sampled at the fixed step; only the samples past it
/// are spaced out.
const GROUND_STEP_SPREAD: f32 = 300.0;

/// The interval the bisection below stops at, in yards: 3 cm, the result of
/// four halvings of a half-yard step.
const GROUND_REFINE_TO: f32 = GROUND_STEP / 16.0;

/// The most halvings the bisection may take to reach [`GROUND_REFINE_TO`].
/// Reached only by a step that has grown past 30 yards, which is a
/// `max_distance` of about nine thousand.
const GROUND_REFINE_MAX: u32 = 12;

/// Where a ray meets the ground: the floor point under a pointer, which is
/// where a ground-targeted spell such as Blizzard lands.
///
/// `floor_at(x, y, z)` is the caller's combination of the terrain and the
/// buildings on it, queried at the height the ray has reached. It takes the
/// same three arguments with the same meaning as
/// `world::standing::Standing::floor`, and is passed in rather than rebuilt
/// here so that a placement and a footstep use the same floor. Everything is
/// in WoW's axes, `+Z up`.
///
/// The result is the first sample at which the ray is under the floor,
/// refined by bisection and then interpolated, so a ray that grazes a
/// hillside lands on the hillside rather than at the next sample past it.
/// `None` when the ray runs the whole distance without going below a surface,
/// which is a pointer aimed at the sky and is a valid result, not a failure.
///
/// A ray that starts below the floor returns its origin immediately. That is
/// a camera inside the ground, and marching on until the ray comes back out
/// would point at the far side of the hill.
pub fn ground_under_ray(
    origin: [f32; 3],
    direction: [f32; 3],
    max_distance: f32,
    floor_at: impl Fn(f32, f32, f32) -> Option<f32>,
) -> Option<[f32; 3]> {
    let length = dot(direction, direction).sqrt();
    if length < 1.0e-6 || !max_distance.is_finite() || max_distance <= 0.0 {
        return None;
    }
    let dir = [
        direction[0] / length,
        direction[1] / length,
        direction[2] / length,
    ];
    let at = |t: f32| {
        [
            origin[0] + dir[0] * t,
            origin[1] + dir[1] * t,
            origin[2] + dir[2] * t,
        ]
    };
    // How far the ray is above the floor at `t`. `None` where there is no
    // floor under that column (a hole in the terrain, or past the edge of the
    // loaded map), which is neither above nor below and cannot end the walk.
    let above = |t: f32| {
        let p = at(t);
        floor_at(p[0], p[1], p[2]).map(|floor| p[2] - floor)
    };

    let mut previous: Option<(f32, f32)> = above(0.0).map(|gap| (0.0, gap));
    if let Some((_, gap)) = previous {
        if gap <= 0.0 {
            return Some(at(0.0));
        }
    }
    let mut t = GROUND_STEP;
    while t <= max_distance {
        match above(t) {
            Some(gap) if gap <= 0.0 => {
                // No usable sample before it means no interval to refine: the
                // ray came in over a hole or past the edge of the loaded map,
                // and the first sample with a floor is already under it. The
                // sample itself is the result; inventing a previous one would
                // interpolate against an unmeasured gap.
                let Some((mut lo, mut lo_gap)) = previous else {
                    let mut hit = at(t);
                    if let Some(floor) = floor_at(hit[0], hit[1], hit[2]) {
                        hit[2] = floor;
                    }
                    return Some(hit);
                };
                let mut hi = t;
                for _ in 0..GROUND_REFINE_MAX {
                    if hi - lo <= GROUND_REFINE_TO {
                        break;
                    }
                    let mid = 0.5 * (lo + hi);
                    match above(mid) {
                        Some(gap) if gap <= 0.0 => hi = mid,
                        Some(gap) => {
                            lo = mid;
                            lo_gap = gap;
                        }
                        // No floor under the midpoint: neither half can be
                        // excluded, so refinement stops at the current interval.
                        None => break,
                    }
                }
                // One linear step across the last interval, which is exact for a
                // floor that is planar over three centimetres.
                let hi_gap = above(hi).unwrap_or(0.0);
                let span = lo_gap - hi_gap;
                let cross = if span > 1.0e-6 {
                    lo + (hi - lo) * (lo_gap / span)
                } else {
                    hi
                };
                let mut hit = at(cross);
                // The floor's height rather than the ray's, so the point is on
                // the surface even where the interpolation lands slightly off it.
                if let Some(floor) = floor_at(hit[0], hit[1], hit[2]) {
                    hit[2] = floor;
                }
                return Some(hit);
            }
            Some(gap) => previous = Some((t, gap)),
            None => previous = None,
        }
        // The step grows with distance; see [`GROUND_STEP_SPREAD`]. Out to
        // 150 yards it is [`GROUND_STEP`]. Past that the cost is no longer
        // linear in the range, so a ground-targeted cast can reach the edge of
        // the loaded world rather than stopping at a fixed bound.
        t += (t / GROUND_STEP_SPREAD).max(GROUND_STEP);
    }
    None
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// A column-major 4x4 applied to a point, in the layout
/// [`crate::world::adt::placement_matrix`] returns.
pub fn transform_point(m: &[f32; 16], v: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for (row, o) in out.iter_mut().enumerate() {
        *o = m[row] * v[0] + m[4 + row] * v[1] + m[8 + row] * v[2] + m[12 + row];
    }
    out
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Unit normal of a triangle, or zero for a degenerate one.
fn normal_of(t: &[[f32; 3]; 3]) -> [f32; 3] {
    let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
    let length = dot(n, n).sqrt();
    if length < 1.0e-12 {
        return [0.0; 3];
    }
    [n[0] / length, n[1] / length, n[2] / length]
}

/// The height of a triangle's plane at `(x, y)`, or `None` when the point is
/// outside the triangle's XY projection.
///
/// A triangle seen edge-on, as every wall is, projects to zero area, and a
/// height from it would put the character on top of the wall they stand
/// beside. The area test returns `None` for those.
fn height_at(t: &[[f32; 3]; 3], x: f32, y: f32) -> Option<f32> {
    let (x0, y0) = (t[0][0], t[0][1]);
    let (x1, y1) = (t[1][0], t[1][1]);
    let (x2, y2) = (t[2][0], t[2][1]);
    let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if area.abs() < 1.0e-9 {
        return None;
    }
    let u = ((x - x0) * (y2 - y0) - (x2 - x0) * (y - y0)) / area;
    let v = ((x1 - x0) * (y - y0) - (x - x0) * (y1 - y0)) / area;
    if u < 0.0 || v < 0.0 || u + v > 1.0 {
        return None;
    }
    Some(t[0][2] + u * (t[1][2] - t[0][2]) + v * (t[2][2] - t[0][2]))
}

/// Möller–Trumbore, two-sided, because a WMO's winding does not reliably say
/// which side of a wall the character is on.
fn ray_triangle(origin: [f32; 3], dir: [f32; 3], t: &[[f32; 3]; 3]) -> Option<f32> {
    let edge1 = sub(t[1], t[0]);
    let edge2 = sub(t[2], t[0]);
    let h = cross(dir, edge2);
    let a = dot(edge1, h);
    if a.abs() < 1.0e-9 {
        return None;
    }
    let f = 1.0 / a;
    let s = sub(origin, t[0]);
    let u = f * dot(s, h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(s, edge1);
    let v = f * dot(dir, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    Some(f * dot(edge2, q))
}

/// Every hull a tile places, in world space: its doodads and its buildings.
///
/// Takes a reader rather than an `Assets`, which is this project's convention
/// for anything that loads by name. The two callers hold the archives
/// differently: `vale bake` has an `Assets`, and a host with a `GameAssets`
/// has an `Rc<dyn Fn(&str) -> Option<Vec<u8>>>`. A signature naming either
/// would need a second copy of this function for the other.
///
/// A model that does not open, does not parse, or carries no collision is
/// skipped rather than reported. A model with no collision is not a failure:
/// it is one the character walks through (grass, birds, flames), and that is
/// most of what a tile places.
///
/// Each distinct path is read once however many times it is placed, which for a
/// forest tile is a few hundred reads against a few thousand placements.
pub fn tile_hulls(
    adt: &crate::world::adt::Adt,
    read: impl FnMut(&str) -> Option<Vec<u8>>,
) -> Vec<Collider> {
    hulls_near(&[adt], read, None)
}

/// Every hull placed by any of several tiles that reaches into a box: the
/// same as [`tile_hulls`] over a neighbourhood, with a bound on what is kept.
///
/// A placement is a record in the tile whose `MODF` or `MDDF` carries it, and
/// a building is carried by the tile its origin stands on. The hulls that
/// stand over a tile are therefore not only the hulls its own file places:
/// Menethil's harbour is placed by one tile and its walls cross the seam onto
/// the next, and a bake given only the next tile's file casts no shadow from
/// them. A rebake then shadows the tile the building belongs to and none of
/// the other tiles it stands on. The caller passes the tile and its eight
/// neighbours, and `within` (the tile's box widened by the furthest a shadow
/// can fall) keeps this from building every hull in nine tiles to serve one.
///
/// The neighbourhood needs three things a single tile does not:
///
/// * a placement is kept once, by unique id. A model straddling a seam is
///   recorded by both tiles with the same `unique_id`, and building it twice
///   gives two identical hulls in the same place and twice the ray tests;
/// * a building is boxed before it is read. A WMO is one root and many
///   group files, and the root alone carries `MOHD`'s bounds, so the eight
///   corners of that box through the placement matrix decide whether the
///   groups are opened at all. Stormwind's 830k triangles stay unread for a
///   tile two seams away;
/// * a doodad is boxed after, since an `M2` is one read either way and its
///   hull's world box is not known until it is placed. `None` keeps everything.
///
/// `within` is in world space; only its x and y are tested, because a shadow
/// falls from above and a hull standing over the box at any height casts into
/// it.
pub fn hulls_near(
    adts: &[&crate::world::adt::Adt],
    mut read: impl FnMut(&str) -> Option<Vec<u8>>,
    within: Option<[[f32; 3]; 2]>,
) -> Vec<Collider> {
    use std::collections::{BTreeMap, HashSet};

    let mut out = Vec::new();
    let mut seen: HashSet<u32> = HashSet::new();
    let overlaps = |bounds: [[f32; 3]; 2]| match within {
        None => true,
        Some(keep) => {
            bounds[0][0] <= keep[1][0]
                && bounds[1][0] >= keep[0][0]
                && bounds[0][1] <= keep[1][1]
                && bounds[1][1] >= keep[0][1]
        }
    };

    // The buildings, whose geometry is spread over a root and its groups. The
    // root is read first and alone, because its bounds decide whether the
    // groups are needed.
    let mut roots: BTreeMap<String, Option<crate::world::wmo::WmoRoot>> = BTreeMap::new();
    let mut meshes: BTreeMap<String, CollisionMesh> = BTreeMap::new();
    for adt in adts {
        for building in adt.placed_wmos() {
            if !seen.insert(building.unique_id) {
                continue;
            }
            let root = roots.entry(building.path.clone()).or_insert_with(|| {
                read(&building.path).and_then(|raw| crate::world::wmo::WmoRoot::parse(&raw).ok())
            });
            let Some(root) = root else { continue };
            if !overlaps(placed_box(root.bounds, &building.matrix)) {
                continue;
            }
            if !meshes.contains_key(&building.path) {
                let mut groups = Vec::new();
                for index in 0..root.group_count {
                    if let Some(group) = read(&crate::world::wmo::group_path(&building.path, index))
                        .and_then(|b| crate::world::wmo::WmoGroup::parse(&b).ok())
                    {
                        groups.push(group);
                    }
                }
                meshes.insert(building.path.clone(), CollisionMesh::build(&groups));
            }
            let mesh = &meshes[&building.path];
            if !mesh.is_empty() {
                out.push(Collider::place(mesh, &building.matrix));
            }
        }
    }

    // The doodads, whose hull is one chunk of one file.
    let mut hulls: BTreeMap<String, Option<CollisionMesh>> = BTreeMap::new();
    for adt in adts {
        for doodad in adt.placed_doodads() {
            if !seen.insert(doodad.unique_id) {
                continue;
            }
            let hull = hulls.entry(doodad.path.clone()).or_insert_with(|| {
                read(&doodad.path)
                    .and_then(|b| crate::world::m2::M2::parse(&b).ok())
                    .map(|m| m.collision)
            });
            let Some(hull) = hull else { continue };
            if hull.is_empty() {
                continue;
            }
            let placed = Collider::place(hull, &doodad.matrix);
            if overlaps(placed.bounds()) {
                out.push(placed);
            }
        }
    }

    out
}

/// Which of a building's triangles cast a baked shadow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildingCasts {
    /// The solid triangles: the building's collision hull.
    Hull,
    /// Every drawn triangle, solid or not.
    Drawn,
}

/// A texture's alpha channel, for testing a ray hit against.
#[derive(Debug, Clone)]
pub struct AlphaMask {
    pub width: u32,
    pub height: u32,
    /// One byte per texel, row-major.
    pub alpha: Vec<u8>,
}

impl AlphaMask {
    /// The alpha at a texture coordinate, wrapped into 0..1 as a repeating
    /// sampler wraps it, nearest texel.
    pub fn at(&self, uv: [f32; 2]) -> u8 {
        let wrap = |v: f32, n: u32| {
            let f = v - v.floor();
            ((f * n as f32) as u32).min(n - 1) as usize
        };
        let (x, y) = (wrap(uv[0], self.width), wrap(uv[1], self.height));
        self.alpha[y * self.width as usize + x]
    }
}

/// A triangle that casts only where its texture is opaque: which mask, and
/// the texture coordinate at each corner.
#[derive(Debug, Clone, Copy)]
pub struct Cutout {
    pub mask: u32,
    pub uvs: [[f32; 2]; 3],
}

/// The triangles that cast a baked `MCSH` shadow, in world space.
#[derive(Debug, Clone, Default)]
pub struct ShadowCasters {
    pub triangles: Vec<[[f32; 3]; 3]>,
    /// Parallel to `triangles`: `None` for a triangle that casts everywhere.
    pub cutouts: Vec<Option<Cutout>>,
    pub masks: Vec<AlphaMask>,
}

/// A model-space mesh ready to place: positions, and per triangle its corners
/// and its cutout.
struct CasterMesh {
    positions: Vec<[f32; 3]>,
    triangles: Vec<([u32; 3], Option<Cutout>)>,
}

/// Every triangle that casts a baked `MCSH` shadow into `within`, in world
/// space: each placed doodad's drawn triangles, and each placed building's
/// triangles as `buildings` selects.
///
/// With `cut_out`, each triangle of a doodad batch that is not opaque (blend
/// mode other than 0) carries its texture's alpha and texture coordinates, so
/// a ray can be tested against the texel it hits. Leaf cards are drawn this
/// way, and testing their alpha brings the share of texels agreeing with the
/// shipped map from 90.5–98.2% to 96.1–98.9% on the four tiles listed at
/// `vale_edit::shadow`.
///
/// A doodad casts from its drawn triangles, not from its collision hull. A
/// tree's hull is its trunk and a rough block for the crown, while the shipped
/// shadow maps cover the whole crown: on `Azeroth_32_49` the hulls find 20.5%
/// of the shipped shadow and the drawn triangles 98.5%.
///
/// Placements are kept once by unique id across `adts`, as in [`hulls_near`].
/// `within` is tested on x and y only; `None` keeps everything.
pub fn shadow_casters_near(
    adts: &[&crate::world::adt::Adt],
    mut read: impl FnMut(&str) -> Option<Vec<u8>>,
    within: Option<[[f32; 3]; 2]>,
    buildings: BuildingCasts,
    cut_out: bool,
) -> ShadowCasters {
    use std::collections::{BTreeMap, HashSet};

    let overlaps = |bounds: [[f32; 3]; 2]| match within {
        None => true,
        Some(keep) => {
            bounds[0][0] <= keep[1][0]
                && bounds[1][0] >= keep[0][0]
                && bounds[0][1] <= keep[1][1]
                && bounds[1][1] >= keep[0][1]
        }
    };
    let mut out = ShadowCasters::default();
    let mut seen: HashSet<u32> = HashSet::new();
    let place = |mesh: &CasterMesh, matrix: &[f32; 16], out: &mut ShadowCasters| {
        let placed: Vec<[f32; 3]> = mesh.positions.iter().map(|&v| transform_point(matrix, v)).collect();
        let mut bounds = [[f32::MAX; 3], [f32::MIN; 3]];
        for v in &placed {
            for axis in 0..3 {
                bounds[0][axis] = bounds[0][axis].min(v[axis]);
                bounds[1][axis] = bounds[1][axis].max(v[axis]);
            }
        }
        if placed.is_empty() || !overlaps(bounds) {
            return;
        }
        for (corners, cutout) in &mesh.triangles {
            let corner = |i: u32| placed.get(i as usize).copied();
            if let (Some(a), Some(b), Some(c)) = (corner(corners[0]), corner(corners[1]), corner(corners[2])) {
                out.triangles.push([a, b, c]);
                out.cutouts.push(*cutout);
            }
        }
    };
    let solid = |indices: &[u32]| -> Vec<([u32; 3], Option<Cutout>)> {
        indices.chunks_exact(3).map(|t| ([t[0], t[1], t[2]], None)).collect()
    };

    // Buildings. The root is read first, and its bounds through the placement
    // decide whether the groups are read at all.
    let mut roots: BTreeMap<String, Option<crate::world::wmo::WmoRoot>> = BTreeMap::new();
    let mut meshes: BTreeMap<String, CasterMesh> = BTreeMap::new();
    for adt in adts {
        for building in adt.placed_wmos() {
            if !seen.insert(building.unique_id) {
                continue;
            }
            let root = roots.entry(building.path.clone()).or_insert_with(|| {
                read(&building.path).and_then(|raw| crate::world::wmo::WmoRoot::parse(&raw).ok())
            });
            let Some(root) = root else { continue };
            if !overlaps(placed_box(root.bounds, &building.matrix)) {
                continue;
            }
            if !meshes.contains_key(&building.path) {
                let mut groups = Vec::new();
                for index in 0..root.group_count {
                    if let Some(group) = read(&crate::world::wmo::group_path(&building.path, index))
                        .and_then(|b| crate::world::wmo::WmoGroup::parse(&b).ok())
                    {
                        groups.push(group);
                    }
                }
                let mesh = match buildings {
                    BuildingCasts::Hull => {
                        let hull = CollisionMesh::build(&groups);
                        CasterMesh {
                            triangles: solid(&hull.indices),
                            positions: hull.positions,
                        }
                    }
                    BuildingCasts::Drawn => {
                        let (mut positions, mut indices) = (Vec::new(), Vec::new());
                        for group in groups.iter().filter(|g| !g.is_antiportal()) {
                            let base = positions.len() as u32;
                            positions.extend_from_slice(&group.positions);
                            indices.extend(group.indices.iter().map(|&i| base + u32::from(i)));
                        }
                        CasterMesh {
                            triangles: solid(&indices),
                            positions,
                        }
                    }
                };
                meshes.insert(building.path.clone(), mesh);
            }
            place(&meshes[&building.path], &building.matrix, &mut out);
        }
    }

    // Doodads, from the triangles their batches draw.
    let mut mask_ids: BTreeMap<String, Option<u32>> = BTreeMap::new();
    let mut models: BTreeMap<String, Option<CasterMesh>> = BTreeMap::new();
    for adt in adts {
        for doodad in adt.placed_doodads() {
            if !seen.insert(doodad.unique_id) {
                continue;
            }
            if !models.contains_key(&doodad.path) {
                let m2 = read(&doodad.path).and_then(|raw| crate::world::m2::M2::parse(&raw).ok());
                let mesh = m2.map(|m2| {
                    let mut triangles = Vec::new();
                    let total = m2.indices.len() / 3;
                    for batch in &m2.batches {
                        let mask = match (cut_out, batch.blend) {
                            (true, 1..) => batch
                                .texture
                                .and_then(|t| m2.textures.get(t as usize))
                                .filter(|t| !t.file_name.is_empty())
                                .and_then(|t| {
                                    *mask_ids
                                        .entry(t.file_name.to_ascii_lowercase())
                                        .or_insert_with(|| alpha_mask(&mut read, &t.file_name, &mut out.masks))
                                }),
                            _ => None,
                        };
                        let first = batch.index_start as usize / 3;
                        let last = (first + batch.index_count as usize / 3).min(total);
                        for t in first..last {
                            let c = [0, 1, 2].map(|k| u32::from(m2.indices[t * 3 + k]));
                            let cutout = mask.map(|mask| Cutout {
                                mask,
                                uvs: c.map(|i| m2.uvs.get(i as usize).copied().unwrap_or_default()),
                            });
                            triangles.push((c, cutout));
                        }
                    }
                    CasterMesh {
                        positions: m2.positions,
                        triangles,
                    }
                });
                models.insert(doodad.path.clone(), mesh);
            }
            if let Some(Some(mesh)) = models.get(&doodad.path) {
                place(mesh, &doodad.matrix, &mut out);
            }
        }
    }
    out
}

/// A texture's alpha channel added to `masks`, and its index. `None` for a
/// texture that does not read, or that has no transparent texel and so casts
/// solid.
fn alpha_mask(
    read: &mut impl FnMut(&str) -> Option<Vec<u8>>,
    name: &str,
    masks: &mut Vec<AlphaMask>,
) -> Option<u32> {
    let blp = crate::world::blp::decode(&read(name)?).ok()?;
    let alpha: Vec<u8> = blp.rgba.chunks_exact(4).map(|p| p[3]).collect();
    if alpha.iter().all(|&a| a == 255) {
        return None;
    }
    masks.push(AlphaMask {
        width: blp.width,
        height: blp.height,
        alpha,
    });
    Some(masks.len() as u32 - 1)
}

/// A model-space box's world-space box, through a placement matrix: the eight
/// corners transformed and re-boxed.
fn placed_box(bounds: [[f32; 3]; 2], matrix: &[f32; 16]) -> [[f32; 3]; 2] {
    let mut out = [[f32::MAX; 3], [f32::MIN; 3]];
    for corner in 0..8 {
        let pick = |axis: usize| match corner >> axis & 1 {
            0 => bounds[0][axis],
            _ => bounds[1][axis],
        };
        let v = transform_point(matrix, [pick(0), pick(1), pick(2)]);
        for axis in 0..3 {
            out[0][axis] = out[0][axis].min(v[axis]);
            out[1][axis] = out[1][axis].max(v[axis]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::wmo;

    /// vmangos' rule, case by case. The `DETAIL` term is the one that cannot be
    /// inferred from the flag names, and dropping it doubles the hull in the
    /// places that are queried most.
    #[test]
    fn a_detail_triangle_is_drawn_but_not_solid() {
        use wmo::mopy_flags::*;
        assert!(wmo::collides(RENDER));
        assert!(wmo::collides(COLLISION));
        assert!(wmo::collides(COLLISION | RENDER | DETAIL));
        assert!(!wmo::collides(RENDER | DETAIL));
        assert!(!wmo::collides(0));
        assert!(!wmo::collides(DETAIL));
    }

    /// A unit quad on the XY plane at z = 5, as two triangles.
    fn flat_quad(z: f32) -> CollisionMesh {
        CollisionMesh {
            positions: vec![
                [0.0, 0.0, z],
                [10.0, 0.0, z],
                [10.0, 10.0, z],
                [0.0, 10.0, z],
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            declared_triangles: 2,
            ..CollisionMesh::default()
        }
    }

    /// A wall in the plane x = 5, ten yards wide and four high.
    fn wall() -> CollisionMesh {
        CollisionMesh {
            positions: vec![
                [5.0, 0.0, 0.0],
                [5.0, 10.0, 0.0],
                [5.0, 10.0, 4.0],
                [5.0, 0.0, 4.0],
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            declared_triangles: 2,
            ..CollisionMesh::default()
        }
    }

    const IDENTITY: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];

    /// The overlay's query is bounded by radius and checked against the map,
    /// which makes it safe to run every frame. A radius that admits everything
    /// draws a million lines in one frame, and a hull left over from the map
    /// before a teleport draws triangles where nothing stands.
    #[test]
    fn hulls_near_is_bounded_by_its_radius_and_by_the_map() {
        let world = CollisionWorld::new();
        let far = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            100.0, 0.0, 0.0, 1.0,
        ];
        world.insert(
            0,
            (32, 32),
            SolidId::placement(1),
            Solid::Building,
            Arc::new(Collider::place(&flat_quad(5.0), &IDENTITY)),
        );
        world.insert(
            0,
            (32, 32),
            SolidId::placement(2),
            Solid::Building,
            Arc::new(Collider::place(&flat_quad(5.0), &far)),
        );

        let count = |map: u32, centre: [f32; 3], radius: f32| {
            let mut n = 0;
            world.hulls_near(map, centre, radius, |_| n += 1);
            n
        };

        // Standing on the near quad: one hull, and the one a hundred yards
        // east is not in it.
        assert_eq!(count(0, [5.0, 5.0, 5.0], 20.0), 1);
        // Widen far enough to reach the other and both come back.
        assert_eq!(count(0, [5.0, 5.0, 5.0], 120.0), 2);
        // On a map these hulls are not on, nothing is returned.
        assert_eq!(count(1, [5.0, 5.0, 5.0], 120.0), 0);
    }

    #[test]
    fn a_floor_is_found_under_a_point_over_it_and_not_beside_it() {
        let c = Collider::place(&flat_quad(5.0), &IDENTITY);
        assert_eq!(c.floor(5.0, 5.0, 100.0), Some(5.0));
        assert_eq!(c.floor(-1.0, 5.0, 100.0), None);
        assert_eq!(c.floor(5.0, 11.0, 100.0), None);
    }

    /// The slope is taken from the triangle that supplied the height. That
    /// lets a model stand aligned to a ramp inside a building rather than to
    /// the terrain under it, and keeps it from leaning by one step of a
    /// staircase while standing on the next.
    #[test]
    fn a_surface_answers_the_slope_of_the_triangle_it_took_the_height_from() {
        // Two decks: a flat one at z = 5 and a ramp above it rising in x.
        let ramp = CollisionMesh {
            positions: vec![
                [0.0, 0.0, 10.0],
                [10.0, 0.0, 20.0],
                [10.0, 10.0, 20.0],
                [0.0, 10.0, 10.0],
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            declared_triangles: 2,
            ..CollisionMesh::default()
        };
        let flat = Collider::place(&flat_quad(5.0), &IDENTITY);
        let (z, n) = flat.surface(5.0, 5.0, 100.0).expect("over the deck");
        assert_eq!(z, 5.0);
        assert_eq!(n, [0.0, 0.0, 1.0], "a level deck is straight up");

        let sloped = Collider::place(&ramp, &IDENTITY);
        let (z, n) = sloped.surface(5.0, 5.0, 100.0).expect("over the ramp");
        assert!((z - 15.0).abs() < 1e-4, "{z}");
        // Rising one yard in z per yard in x: the normal leans back along -x at
        // 45°, and it points up whatever the file's winding, which a
        // `BoundingTriangles` hull does not guarantee.
        let root_half = std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (n[0] + root_half).abs() < 1e-3 && n[1].abs() < 1e-3 && (n[2] - root_half).abs() < 1e-3,
            "{n:?}"
        );
        // `floor` returns the same height without the slope, so the mover and
        // the stance cannot pick different surfaces.
        assert_eq!(sloped.floor(5.0, 5.0, 100.0), Some(z));
        assert_eq!(sloped.surface(-1.0, 5.0, 100.0), None);
    }

    /// The ceiling keeps a character walking under a bridge from being lifted
    /// onto its deck, which is why the query takes one.
    #[test]
    fn a_surface_above_the_ceiling_is_not_the_one_stood_on() {
        let c = Collider::place(&flat_quad(5.0), &IDENTITY);
        assert_eq!(c.floor(5.0, 5.0, 4.9), None);
        assert_eq!(c.floor(5.0, 5.0, 5.0), Some(5.0));
    }

    /// Two decks stacked: standing on the lower one must not return the upper,
    /// and standing on the upper must not return the lower.
    #[test]
    fn the_highest_floor_below_the_ceiling_wins() {
        let mut mesh = flat_quad(0.0);
        let upper = flat_quad(6.0);
        let base = mesh.positions.len() as u32;
        mesh.positions.extend(upper.positions);
        mesh.indices.extend(upper.indices.iter().map(|i| i + base));
        let c = Collider::place(&mesh, &IDENTITY);

        assert_eq!(c.floor(5.0, 5.0, 0.0 + STEP_UP), Some(0.0));
        assert_eq!(c.floor(5.0, 5.0, 6.0 + STEP_UP), Some(6.0));
    }

    /// A wall projects to zero area in XY. A height from it would stand the
    /// character on top of the wall they are walking beside.
    #[test]
    fn a_vertical_face_is_never_a_floor() {
        let c = Collider::place(&wall(), &IDENTITY);
        assert_eq!(c.floor(5.0, 5.0, 100.0), None);
    }

    /// The grid must not lose a triangle that spans many cells. A cathedral
    /// floor can be one such triangle, and losing it from some cells leaves a
    /// hole in the floor only in those cells, which is easy to miss in play.
    #[test]
    fn a_triangle_spanning_many_cells_is_found_from_all_of_them() {
        let mesh = CollisionMesh {
            positions: vec![[0.0, 0.0, 3.0], [400.0, 0.0, 3.0], [400.0, 400.0, 3.0]],
            indices: vec![0, 1, 2],
            declared_triangles: 1,
            ..CollisionMesh::default()
        };
        let c = Collider::place(&mesh, &IDENTITY);
        assert!(c.cols > 1 && c.rows > 1, "the grid should have cells");
        for step in 1..20 {
            let p = step as f32 * 19.0;
            assert_eq!(c.floor(p + 1.0, p - 1.0, 100.0), Some(3.0), "at {p}");
        }
    }

    fn world_with(mesh: &CollisionMesh) -> CollisionWorld {
        let world = CollisionWorld::new();
        world.insert(
            0,
            (32, 32),
            SolidId::placement(1),
            Solid::Building,
            Arc::new(Collider::place(mesh, &IDENTITY)),
        );
        world
    }

    /// Every query rejects on [`Tile::boxes`] before it touches a hull, so a
    /// box vector out of step with its hulls reports that nothing is there.
    ///
    /// A missed `boxes.push` does not panic or produce an obvious error: it
    /// makes a building stop being solid, which elsewhere looks like a general
    /// collision bug. The queries test the last hull inserted, because a
    /// length mismatch always drops the tail: `zip` stops at the shorter side.
    #[test]
    fn every_hull_keeps_its_box_and_stays_answerable() {
        let world = CollisionWorld::new();
        // Ten floors, each one yard higher and offset in x, all on one tile.
        for i in 0..10u32 {
            let mut mesh = flat_quad(i as f32);
            for p in &mut mesh.positions {
                p[0] += i as f32 * 20.0;
            }
            world.insert(
                0,
                (32, 32),
                SolidId::placement(i),
                Solid::Building,
                Arc::new(Collider::place(&mesh, &IDENTITY)),
            );
        }
        {
            let inner = world.read();
            let tile = &inner.tiles[&(32, 32)];
            assert_eq!(tile.boxes.len(), tile.solids.len(), "a hull with no box");
            assert_eq!(tile.kinds.len(), tile.solids.len(), "a hull with no kind");
            assert_eq!(tile.buildings, (0..tile.solids.len()).collect::<Vec<_>>(), "a building not indexed");
            for (bounds, collider) in tile.boxes.iter().zip(&tile.solids) {
                assert_eq!(*bounds, collider.bounds(), "a box that is not its hull's");
            }
        }
        // The tenth floor: reachable by the point query and by the segment
        // query.
        assert_eq!(world.floor(0, 185.0, 5.0, 100.0), Some(9.0));
        assert!(
            world.ray(0, [185.0, 5.0, 20.0], [185.0, 5.0, 0.0]).is_some(),
            "the ray missed a hull it passes straight through"
        );
    }

    /// A ray narrowed to one population hits that population and nothing
    /// else, and the three populations are told apart.
    ///
    /// Both failure directions are silent. A filter that admitted too much
    /// would stand something on the roof of a hidden building, out of sight.
    /// A filter that admitted too little would hit nothing and drop the object
    /// to the terrain under a visible roof, the fault the filter was added to
    /// fix.
    #[test]
    fn a_ray_can_be_narrowed_to_one_population() {
        let world = CollisionWorld::new();
        // A floor at z = 5 from each of the three populations, side by side.
        let building = flat_quad(5.0);
        let mut doodad = flat_quad(5.0);
        for p in &mut doodad.positions {
            p[0] += 20.0;
        }
        let mut object = flat_quad(5.0);
        for p in &mut object.positions {
            p[0] += 40.0;
        }
        world.insert(
            0,
            (32, 32),
            SolidId::placement(1),
            Solid::Building,
            Arc::new(Collider::place(&building, &IDENTITY)),
        );
        world.insert(
            0,
            (32, 32),
            SolidId::placement(2),
            Solid::Doodad,
            Arc::new(Collider::place(&doodad, &IDENTITY)),
        );
        world.place_object(
            0,
            7,
            ObjectPlacement {
                position: [0.0; 3],
                facing: 0.0,
                scale: 1.0,
            },
            Some(Arc::new(Collider::place(&object, &IDENTITY))),
        );

        // Straight down onto each of the three, with each filter.
        let down = |x: f32, want: Surfaces| world.ray_of(0, [x, 5.0, 20.0], [x, 5.0, 0.0], want);
        for (x, kind) in [
            (5.0, Solid::Building),
            (25.0, Solid::Doodad),
            (45.0, Solid::Object),
        ] {
            let only = Surfaces {
                buildings: kind == Solid::Building,
                doodads: kind == Solid::Doodad,
                objects: kind == Solid::Object,
            };
            assert!(down(x, only).is_some(), "{kind:?} did not answer for itself");
            assert!(
                down(x, Surfaces::ALL).is_some(),
                "{kind:?} did not answer for everything"
            );
            // Every filter that leaves this population out hits nothing over
            // it. This half of the check cannot be seen in a screenshot.
            let others = Surfaces {
                buildings: !only.buildings,
                doodads: !only.doodads,
                objects: !only.objects,
            };
            assert!(down(x, others).is_none(), "{kind:?} answered for the others");
            assert!(down(x, Surfaces::NONE).is_none(), "{kind:?} answered NONE");
        }
    }

    #[test]
    fn a_stride_into_a_wall_is_stopped_and_one_along_it_is_not() {
        let world = world_with(&wall());
        let from = [3.0, 5.0, 0.0];
        // Straight at it: the whole stride is cancelled.
        let end = world.step(0, from, [7.0, 5.0, 0.0]);
        assert!((end[0] - from[0]).abs() < 1.0e-3, "x moved to {}", end[0]);
        // Along it: untouched.
        let end = world.step(0, from, [3.0, 9.0, 0.0]);
        assert!((end[1] - 9.0).abs() < 1.0e-3, "y stopped at {}", end[1]);
    }

    /// The diagonal case decides whether a character slides along a wall or
    /// sticks to it.
    #[test]
    fn a_stride_at_an_angle_slides_along_the_wall() {
        let world = world_with(&wall());
        let end = world.step(0, [3.0, 5.0, 0.0], [7.0, 9.0, 0.0]);
        assert!((end[0] - 3.0).abs() < 1.0e-3, "x moved to {}", end[0]);
        assert!((end[1] - 9.0).abs() < 1.0e-3, "y stopped at {}", end[1]);
    }

    /// A post the character's shoulder would hit stops the stride, even
    /// though nothing is ahead of their centre.
    ///
    /// With a single centre ray, a stride that passes a tree trunk, a fence
    /// post or a door jamb half a yard to the side meets nothing and goes
    /// through it. A character is a yard wide, and the sweep has three lanes.
    ///
    /// The post spans `y` from 0.4 to 0.8: 0.4 yards clear of the centre line
    /// at `y = 0`, and inside the body.
    #[test]
    fn a_post_beside_the_centre_line_still_stops_a_stride() {
        let post = CollisionMesh {
            positions: vec![
                [5.0, 0.4, 0.0],
                [5.0, 0.8, 0.0],
                [5.0, 0.8, 4.0],
                [5.0, 0.4, 4.0],
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            declared_triangles: 2,
            ..CollisionMesh::default()
        };
        let world = world_with(&post);
        // Straight past it, along the centre line, with the post entirely to
        // one side of that line.
        let end = world.step(0, [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]);
        assert!(
            (end[0] - 3.0).abs() < 1.0e-3,
            "walked to {} straight through a post beside the centre line",
            end[0],
        );
        // The same stride two yards further away is not blocked. Without this
        // check, a sweep that blocks every stride would pass the assertion
        // above.
        let end = world.step(0, [3.0, -2.0, 0.0], [7.0, -2.0, 0.0]);
        assert!((end[0] - 7.0).abs() < 1.0e-3, "a clear stride stopped at {}", end[0]);
    }

    /// Motion is cancelled, never reversed. A character who ends up inside
    /// geometry (after a teleport or a server correction) must be able to walk
    /// out.
    #[test]
    fn a_stride_away_from_a_wall_is_never_blocked() {
        let world = world_with(&wall());
        let end = world.step(0, [4.9, 5.0, 0.0], [1.0, 5.0, 0.0]);
        assert!((end[0] - 1.0).abs() < 1.0e-3, "x stopped at {}", end[0]);
    }

    /// A run along a wall at a shallow angle never crosses it, at any stride
    /// length the mover produces and at a whole 250 ms mounted stride.
    ///
    /// Each stride is swept along its whole length and half a yard beyond, so
    /// the wall is met before the stride reaches it for any stride length and
    /// any angle. The strides are the session's run and
    /// mounted ticks (0.175 and 0.35 yards), the mover's longest piece (0.5),
    /// and an unbroken mounted stall (3.5).
    #[test]
    fn a_run_along_a_wall_at_a_shallow_angle_never_crosses_it() {
        let world = world_with(&wall());
        for stride in [0.175_f32, 0.35, 0.5, 3.5] {
            for degrees in [2.0_f32, 5.0, 10.0, 20.0, 45.0] {
                let angle = degrees.to_radians();
                let (dx, dy) = (angle.sin() * stride, angle.cos() * stride);
                let mut at = [3.5_f32, 0.5, 0.0];
                while at[1] < 9.0 {
                    let end = world.step(0, at, [at[0] + dx, at[1] + dy, at[2]]);
                    assert!(
                        end[0] < 5.0,
                        "{stride}-yard strides at {degrees} degrees reached x = {} through the \
                         wall at x = 5",
                        end[0],
                    );
                    if end == at {
                        break;
                    }
                    at = end;
                }
            }
        }
    }

    /// The probes start above [`STEP_UP`] because a stair riser is a wall, and
    /// a probe that stops at one stops the character at the bottom of every
    /// staircase.
    #[test]
    fn a_step_shorter_than_the_probes_does_not_block() {
        let riser = CollisionMesh {
            positions: vec![
                [5.0, 0.0, 0.0],
                [5.0, 10.0, 0.0],
                [5.0, 10.0, STEP_UP],
                [5.0, 0.0, STEP_UP],
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            declared_triangles: 2,
            ..CollisionMesh::default()
        };
        let world = world_with(&riser);
        let end = world.step(0, [4.0, 5.0, 0.0], [6.0, 5.0, 0.0]);
        assert!((end[0] - 6.0).abs() < 1.0e-3, "stopped on a kerb at {}", end[0]);
    }

    /// A far teleport lands on tile coordinates that also exist on the new
    /// map. Answering from the old map's buildings would put invisible walls
    /// on the new one.
    #[test]
    fn another_map_is_never_answered_for() {
        let world = world_with(&flat_quad(5.0));
        assert_eq!(world.floor(0, 5.0, 5.0, 100.0), Some(5.0));
        assert_eq!(world.floor(1, 5.0, 5.0, 100.0), None);

        world.insert(
            1,
            (32, 32),
            SolidId::placement(2),
            Solid::Building,
            Arc::new(Collider::place(&flat_quad(9.0), &IDENTITY)),
        );
        assert_eq!(world.floor(0, 5.0, 5.0, 100.0), None);
        assert_eq!(world.floor(1, 5.0, 5.0, 100.0), Some(9.0));
    }

    #[test]
    fn tiles_that_have_gone_out_of_range_take_their_buildings_with_them() {
        let world = world_with(&flat_quad(5.0));
        world.retain_tiles(&HashSet::from([(32, 32)]));
        assert_eq!(world.counts().0, 1);
        world.retain_tiles(&HashSet::from([(31, 31)]));
        assert_eq!(world.counts(), (0, 0));
    }

    /// The same building placed on two neighbouring tiles carries one id, and
    /// two copies of one hull double every query's work.
    #[test]
    fn the_same_placement_is_not_held_twice() {
        let world = world_with(&flat_quad(5.0));
        world.insert(
            0,
            (32, 32),
            SolidId::placement(1),
            Solid::Building,
            Arc::new(Collider::place(&flat_quad(5.0), &IDENTITY)),
        );
        assert_eq!(world.counts().0, 1);
    }

    /// The id is not a bare `u32` because every `MODD` spawn inside a building
    /// carries the building's unique id, so a thousand chairs would dedup to
    /// one chair and the character would walk through the rest of the
    /// furniture. Three spawns of one placement are three hulls.
    #[test]
    fn furniture_sharing_its_buildings_id_is_still_three_pieces_of_furniture() {
        let world = world_with(&flat_quad(5.0));
        for ordinal in 0..3 {
            world.insert(
                0,
                (32, 32),
                SolidId::spawn(1, ordinal),
                Solid::Building,
                Arc::new(Collider::place(&flat_quad(5.0), &IDENTITY)),
            );
        }
        assert_eq!(world.counts().0, 4, "the building and its three spawns");
        // A spawn ordinal never collides with the placement itself; the
        // 1-based numbering ensures that.
        assert_ne!(SolidId::spawn(1, 0), SolidId::placement(1));
    }

    /// A hull nowhere near the stride must be rejected before any of its
    /// triangles is ray-tested. [`Collider::cell_of`] clamps, so without an
    /// XY test a distant collider resolves to its nearest corner cell and
    /// tests every triangle in it. That is negligible across forty buildings
    /// and takes the whole frame across a forest.
    #[test]
    fn a_hull_far_from_the_stride_is_rejected_without_touching_a_triangle() {
        let c = Collider::place(&wall(), &IDENTITY);
        // Straight at the wall from a yard out: found.
        assert!(c.wall([3.0, 5.0, 2.0], [1.0, 0.0, 0.0], 4.0).is_some());
        // The same ray a hundred yards away in Y, which clamps to the same
        // corner cell and would otherwise test the same two triangles.
        assert!(c.wall([3.0, 105.0, 2.0], [1.0, 0.0, 0.0], 4.0).is_none());
        // A ray that stops short of the wall in X.
        assert!(c.wall([-50.0, 5.0, 2.0], [1.0, 0.0, 0.0], 4.0).is_none());
    }

    /// The hull is built from `MOPY` and the raw index buffer, not from the
    /// draw list, so a group with no batches still contributes.
    #[test]
    fn a_group_with_no_batches_still_has_collision() {
        let group = WmoGroup {
            name: String::new(),
            flags: 0,
            bounds: [[0.0; 3]; 2],
            name_offset: 0,
            positions: vec![[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: vec![0, 1, 2],
            triangle_flags: vec![wmo::mopy_flags::COLLISION],
            batches: Vec::new(),
            doodad_refs: Vec::new(),
            colours: Vec::new(),
            interior_start: 0,
            liquid_type: 0,
            group_id: 0,
            liquid: None,
        };
        let mesh = CollisionMesh::build(std::slice::from_ref(&group));
        assert_eq!(mesh.triangle_count(), 1);

        // The two group types vmangos drops are dropped.
        let mut antiportal = group.clone();
        antiportal.flags = wmo::group_flags::ANTIPORTAL;
        assert_eq!(CollisionMesh::build(&[antiportal]).triangle_count(), 0);
        let mut unreachable = group.clone();
        unreachable.flags = wmo::group_flags::UNREACHABLE;
        assert_eq!(CollisionMesh::build(&[unreachable]).triangle_count(), 0);
    }

    /// An error in the placement matrix would not show in any triangle count,
    /// so it is checked against a rotation with a known result: a 90° yaw
    /// about Z.
    #[test]
    fn a_hull_is_placed_through_the_same_matrix_the_geometry_is() {
        let matrix = crate::world::adt::placement_matrix([100.0, 200.0, 5.0], [0.0, 90.0, 0.0], 1.0);
        let mesh = CollisionMesh {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            indices: vec![0, 1, 2],
            declared_triangles: 1,
            ..CollisionMesh::default()
        };
        let placed = Collider::place(&mesh, &matrix);
        let origin = placed.positions[0];
        assert!((origin[0] - 100.0).abs() < 1.0e-4, "{origin:?}");
        assert!((origin[1] - 200.0).abs() < 1.0e-4, "{origin:?}");
        assert!((origin[2] - 5.0).abs() < 1.0e-4, "{origin:?}");
        // A rotation preserves distances from the placement's origin,
        // whichever axes it maps to which.
        for corner in &placed.positions[1..] {
            let d = sub(*corner, origin);
            assert!((dot(d, d).sqrt() - 1.0).abs() < 1.0e-4, "{corner:?}");
        }
    }
    /// A placement is settled when its hull arrives, and a placement with no
    /// hull must be settled explicitly.
    ///
    /// The reader is `Standing::floor`, which does not stand a character on
    /// the ground under an unsettled `MODF` box. With both paths working, a
    /// character waits a moment for Stormwind's hull; with the second missing,
    /// a character inside a hull-less building hovers for ever.
    #[test]
    fn a_placement_is_settled_by_its_hull_or_by_being_told_it_has_none() {
        let world = CollisionWorld::new();
        assert!(!world.settled(0, 41), "nothing is settled before it arrives");

        world.insert(
            0,
            (32, 48),
            SolidId::placement(41),
            Solid::Building,
            Arc::new(Collider::place(&flat_quad(5.0), &IDENTITY)),
        );
        assert!(world.settled(0, 41), "a hull settles its own placement");

        // The other path: a building whose file carries no
        // `BoundingTriangles`. It never inserts, so `ids` never holds it and
        // it must be settled with `settle`.
        assert!(!world.settled(0, 42));
        world.settle(0, (32, 48), 42);
        assert!(world.settled(0, 42));

        // Checked across the resident tiles, not at one coordinate: a
        // building belongs to the tile its origin stands on and the
        // character under it is often on the next tile.
        world.settle(0, (31, 48), 43);
        assert!(world.settled(0, 43));

        // Settled marks are per map. A mark carried over from another map
        // would be applied to the wrong continent's ground.
        assert!(!world.settled(1, 41));
        world.settle(1, (32, 48), 99);
        assert!(!world.settled(1, 41), "the map change dropped the old tiles");
        assert!(world.settled(1, 99));
    }

    /// A translation by `(x, y, z)` as the column-major matrix the hulls take.
    fn shift(x: f32, y: f32, z: f32) -> [f32; 16] {
        let mut m = IDENTITY;
        m[12] = x;
        m[13] = y;
        m[14] = z;
        m
    }

    /// The same id inserted again is the same placement placed again, and
    /// the new hull replaces the old one rather than being ignored. A caller
    /// that rewrote a tile re-reads it after a move, and mints an undone
    /// placement's id again; ignoring either insert would leave the old hull
    /// in place and the new position without one.
    #[test]
    fn a_second_insert_of_the_same_id_replaces_the_hull() {
        let world = CollisionWorld::new();
        let at = (32, 48);
        world.insert(
            0,
            at,
            SolidId::placement(41),
            Solid::Building,
            Arc::new(Collider::place(&flat_quad(5.0), &IDENTITY)),
        );
        assert_eq!(world.floor(0, 1.0, 1.0, 50.0), Some(5.0));
        world.insert(
            0,
            at,
            SolidId::placement(41),
            Solid::Building,
            Arc::new(Collider::place(&flat_quad(9.0), &IDENTITY)),
        );
        assert_eq!(world.floor(0, 1.0, 1.0, 50.0), Some(9.0), "the new hull answers");
        assert_eq!(world.floor(0, 1.0, 1.0, 7.0), None, "and the old one is gone");
    }

    /// A moved placement moves every hull that carries its id: the building's
    /// own and its furniture's, and no other.
    #[test]
    fn a_moved_placement_takes_its_hulls_with_it() {
        let world = CollisionWorld::new();
        let at = (32, 48);
        let hull = || Arc::new(Collider::place(&flat_quad(5.0), &IDENTITY));
        world.insert(0, at, SolidId::placement(41), Solid::Building, hull());
        world.insert(0, at, SolidId::spawn(41, 0), Solid::Doodad, hull());
        world.insert(0, at, SolidId::placement(42), Solid::Doodad, hull());

        assert_eq!(world.move_placement(0, 41, &shift(100.0, 0.0, 3.0)), 2);
        // Where 41 stood, only 42's quad is left, at its own height.
        assert_eq!(world.floor(0, 1.0, 1.0, 50.0), Some(5.0));
        // Where 41 went, both of its hulls answer at the raised height.
        assert_eq!(world.floor(0, 101.0, 1.0, 50.0), Some(8.0));
        assert_eq!(world.move_placement(0, 7, &shift(1.0, 0.0, 0.0)), 0, "an id nothing carries");
        assert_eq!(world.move_placement(1, 41, &shift(1.0, 0.0, 0.0)), 0, "another map");
    }

    /// A re-read tile keeps only the hulls of placements it still names, and
    /// a removed placement's hulls and settled mark go with it.
    #[test]
    fn hulls_of_placements_a_tile_no_longer_carries_are_dropped() {
        let world = CollisionWorld::new();
        let at = (32, 48);
        let hull = || Arc::new(Collider::place(&flat_quad(5.0), &IDENTITY));
        world.insert(0, at, SolidId::placement(41), Solid::Building, hull());
        world.insert(0, at, SolidId::spawn(41, 0), Solid::Doodad, hull());
        world.insert(0, at, SolidId::placement(42), Solid::Doodad, hull());
        world.settle(0, at, 43);

        let keep: HashSet<u32> = [42, 43].into_iter().collect();
        assert_eq!(world.retain_placements(0, at, &keep), 2);
        assert!(!world.settled(0, 41));
        assert!(world.settled(0, 42) && world.settled(0, 43));
        assert_eq!(world.floor(0, 1.0, 1.0, 50.0), Some(5.0), "42 still answers");
        assert_eq!(world.retain_placements(0, (33, 48), &keep), 0, "a tile holding nothing");

        assert_eq!(world.remove_placement(0, 42), 1);
        assert_eq!(world.floor(0, 1.0, 1.0, 50.0), None);
        assert!(!world.settled(0, 42));
    }

    // -- game objects the server places ------------------------------------

    fn placed_at(x: f32, y: f32) -> ObjectPlacement {
        ObjectPlacement {
            position: [x, y, 0.0],
            facing: 0.0,
            scale: 1.0,
        }
    }

    /// A closed door stops a stride, and the same door standing open does not.
    ///
    /// Both halves are tested. A hull that never arrives is a door the
    /// character walks through. A hull that never leaves is a door that opens
    /// on screen and still blocks the corridor, which is harder to recognise
    /// as a bug.
    #[test]
    fn a_closed_game_object_blocks_a_stride_and_an_open_one_does_not() {
        let world = CollisionWorld::new();
        let door = 0x0F00_0000_0000_0001;
        world.place_object(
            0,
            door,
            placed_at(0.0, 0.0),
            Some(Arc::new(Collider::place(&wall(), &IDENTITY))),
        );

        // Straight at the wall in the plane x = 5, from three yards short.
        let from = [2.0, 5.0, 0.0];
        let blocked = world.step(0, from, [8.0, 5.0, 0.0]);
        assert!(blocked[0] < 5.0, "walked through the door: {blocked:?}");

        // Game object hulls also answer the floor query: a portcullis lying
        // flat can be stood on.
        world.place_object(
            0,
            0x0F00_0000_0000_0002,
            placed_at(0.0, 0.0),
            Some(Arc::new(Collider::place(&flat_quad(5.0), &IDENTITY))),
        );
        assert_eq!(world.floor(0, 5.0, 5.0, 100.0), Some(5.0));

        // The door opens: the caller stops naming it, and the stride goes
        // through. The flat one is still named, which shows that only the door
        // was removed and not every game object.
        let live: HashSet<u64> = [0x0F00_0000_0000_0002].into_iter().collect();
        world.retain_objects(&live);
        assert_eq!(world.step(0, from, [8.0, 5.0, 0.0]), [8.0, 5.0, 0.0]);
        assert_eq!(world.floor(0, 5.0, 5.0, 100.0), Some(5.0));
        assert_eq!(world.object_placement(0, door), None);
    }

    /// A game object with no `BoundingTriangles` is decided about, not
    /// pending.
    ///
    /// Without this the caller cannot tell a campfire it has not examined from
    /// one it examined and found nothing solid in, so it re-reads the M2 and
    /// re-places an empty hull once per frame, per campfire, for as long as one
    /// is in view.
    #[test]
    fn an_object_with_no_hull_is_a_finished_answer_rather_than_a_pending_one() {
        let world = CollisionWorld::new();
        let fire = 0x0F00_0000_0000_0003;
        assert_eq!(world.object_placement(0, fire), None);

        world.place_object(0, fire, placed_at(1.0, 2.0), None);
        assert_eq!(world.object_placement(0, fire), Some(placed_at(1.0, 2.0)));
        assert_eq!(world.counts().0, 0, "nothing solid was added");

        // It is still dropped by GUID when it goes out of view, which a record
        // that only tracked hulls could not do.
        world.retain_objects(&HashSet::new());
        assert_eq!(world.object_placement(0, fire), None);
    }

    /// A game object that moves. The caller compares the placement to decide
    /// whether the hull it holds is current.
    #[test]
    fn a_moved_object_replaces_its_hull_rather_than_gaining_a_second() {
        let world = CollisionWorld::new();
        let gate = 0x0F00_0000_0000_0004;
        let hull = || Arc::new(Collider::place(&wall(), &IDENTITY));

        world.place_object(0, gate, placed_at(0.0, 0.0), Some(hull()));
        assert_eq!(world.counts().0, 1);
        assert_eq!(world.object_placement(0, gate), Some(placed_at(0.0, 0.0)));

        world.place_object(0, gate, placed_at(3.0, 0.0), Some(hull()));
        assert_eq!(world.counts().0, 1, "the old hull is gone, not shadowed");
        assert_eq!(world.object_placement(0, gate), Some(placed_at(3.0, 0.0)));

        // Placing it again with no hull removes the old hull. This is what
        // happens when a game object's display id changes to a model with no
        // collision.
        world.place_object(0, gate, placed_at(3.0, 0.0), None);
        assert_eq!(world.counts().0, 0);
    }

    /// `swap_remove` must keep three vectors and an index in step, and a
    /// mistake is silent: the hull left behind is filed under the wrong GUID,
    /// so the next door to open removes another object's wall.
    #[test]
    fn removing_one_of_several_objects_leaves_the_rest_answerable() {
        let world = CollisionWorld::new();
        let guids = [10_u64, 11, 12, 13];
        for (n, guid) in guids.iter().enumerate() {
            // Four walls at x = 5, 15, 25, 35, one per object, so which hulls
            // survive is visible in the query results and not only the count.
            let matrix = object_matrix([n as f32 * 10.0, 0.0, 0.0], 0.0, 1.0);
            world.place_object(
                0,
                *guid,
                placed_at(n as f32 * 10.0, 0.0),
                Some(Arc::new(Collider::place(&wall(), &matrix))),
            );
        }
        assert_eq!(world.counts().0, 4);

        // Drop the first and the third, the two that force a swap into a hole
        // that is not the last slot.
        let live: HashSet<u64> = [11_u64, 13].into_iter().collect();
        world.retain_objects(&live);
        assert_eq!(world.counts().0, 2);

        let blocked = |x: f32| world.step(0, [x - 3.0, 5.0, 0.0], [x + 3.0, 5.0, 0.0])[0] < x;
        assert!(!blocked(5.0), "guid 10 was dropped and its wall stayed");
        assert!(blocked(15.0), "guid 11 was kept and its wall went");
        assert!(!blocked(25.0), "guid 12 was dropped and its wall stayed");
        assert!(blocked(35.0), "guid 13 was kept and its wall went");
    }

    /// A game object is keyed to a map as a placement is: a door in the
    /// Deadmines must not remain as a hull in Elwynn.
    #[test]
    fn a_map_change_takes_the_game_objects_with_it() {
        let world = CollisionWorld::new();
        let door = 0x0F00_0000_0000_0005;
        world.place_object(
            0,
            door,
            placed_at(0.0, 0.0),
            Some(Arc::new(Collider::place(&wall(), &IDENTITY))),
        );
        assert_eq!(world.counts().0, 1);
        assert_eq!(world.object_placement(1, door), None, "wrong map, no answer");

        world.place_object(1, 0x0F00_0000_0000_0006, placed_at(0.0, 0.0), None);
        assert_eq!(world.counts().0, 0, "the map change dropped the old hull");
        assert_eq!(world.object_placement(1, door), None);
    }

    /// `GO_STATE_READY` is the only solid state, and an absent field is
    /// `GO_STATE_ACTIVE`.
    ///
    /// vmangos omits a zero field from a create block, so an absent state means
    /// open. A client that treated it as unknown and assumed shut would block
    /// every door that was open when it came into view.
    #[test]
    fn only_the_ready_state_is_solid() {
        assert!(game_object_is_solid(Some(GO_STATE_READY)));
        assert!(!game_object_is_solid(Some(0)), "GO_STATE_ACTIVE is open");
        assert!(!game_object_is_solid(Some(2)), "the alternative used state");
        assert!(!game_object_is_solid(None), "an absent field is zero");
    }

    /// The server's matrix is a rotation about +Z only, with no 180° term.
    ///
    /// Both errors are easy to miss. A stray `Rz(pi)`, copied from
    /// `adt::placement_matrix` where it is correct, puts a door's hull on the
    /// other side of its hinge: the character walks through the door from one
    /// side and stops a yard early from the other. A rotation in the wrong
    /// direction does not show on a symmetric gate and is hard to spot on
    /// anything else.
    #[test]
    fn a_server_placed_hull_turns_about_up_and_nothing_else() {
        // Facing zero is the identity apart from the translation.
        let m = object_matrix([100.0, 200.0, 300.0], 0.0, 1.0);
        let moved = transform_point(&m, [1.0, 2.0, 3.0]);
        assert_eq!(moved, [101.0, 202.0, 303.0]);

        // A quarter turn counter-clockwise takes north (+X) to west (+Y), the
        // convention of both the server and the renderer.
        let m = object_matrix([0.0, 0.0, 0.0], std::f32::consts::FRAC_PI_2, 1.0);
        let north = transform_point(&m, [1.0, 0.0, 0.0]);
        assert!(north[0].abs() < 1e-6 && (north[1] - 1.0).abs() < 1e-6, "{north:?}");
        // +Z is unchanged: no pitch and no roll, whatever the facing.
        let up = transform_point(&m, [0.0, 0.0, 1.0]);
        assert!((up[2] - 1.0).abs() < 1e-6 && up[0].abs() < 1e-6 && up[1].abs() < 1e-6);

        // The scale is uniform and applies to the model, not to the position.
        let m = object_matrix([10.0, 0.0, 0.0], 0.0, 2.0);
        assert_eq!(transform_point(&m, [1.0, 1.0, 1.0]), [12.0, 2.0, 2.0]);
    }
}

#[cfg(test)]
mod ground_ray_tests {
    use super::*;

    /// A flat floor at z = 0, with no hole anywhere.
    fn flat(_x: f32, _y: f32, _z: f32) -> Option<f32> {
        Some(0.0)
    }

    /// An eye above the ground, pointing down and north, lands at the
    /// computed crossing.
    #[test]
    fn a_ray_lands_on_the_floor_it_is_pointed_at() {
        // 10 up, aimed 45 degrees down along +X (north): the crossing is at
        // x = 10 exactly.
        let hit = ground_under_ray([0.0, 0.0, 10.0], [1.0, 0.0, -1.0], 100.0, flat)
            .expect("a ray aimed at the ground hits it");
        assert!((hit[0] - 10.0).abs() < 0.05, "{hit:?}");
        assert!(hit[1].abs() < 1.0e-4, "{hit:?}");
        // The z is the floor's, not the ray's.
        assert!(hit[2].abs() < 1.0e-4, "{hit:?}");
    }

    /// A pointer aimed at the sky returns `None`, which hides the placement
    /// cursor rather than casting at the horizon.
    #[test]
    fn a_ray_that_never_reaches_the_floor_answers_nothing() {
        assert_eq!(ground_under_ray([0.0, 0.0, 10.0], [1.0, 0.0, 0.1], 100.0, flat), None);
        // So does a ray whose reach runs out first: 5 yards along it is not
        // enough to descend 10.
        assert_eq!(ground_under_ray([0.0, 0.0, 10.0], [1.0, 0.0, -1.0], 5.0, flat), None);
    }

    /// A slope is met at its true position, which a single height sample under
    /// the pointer would get wrong. The floor here rises north at 45 degrees,
    /// so a ray descending at 45 degrees meets it halfway to where it would
    /// meet a flat floor.
    #[test]
    fn the_crossing_follows_the_slope_rather_than_the_sample_grid() {
        let slope = |x: f32, _y: f32, _z: f32| Some(x);
        let hit = ground_under_ray([0.0, 0.0, 10.0], [1.0, 0.0, -1.0], 100.0, slope)
            .expect("a ray aimed at a hillside hits it");
        assert!((hit[0] - 5.0).abs() < 0.05, "{hit:?}");
        assert!((hit[2] - 5.0).abs() < 0.05, "{hit:?}");
    }

    /// A hole in the terrain is not a floor, so the walk continues past it and
    /// lands on the ground on the far side rather than the near edge.
    #[test]
    fn a_column_with_no_floor_under_it_ends_nothing() {
        let holed = |x: f32, _y: f32, _z: f32| (!(4.0..8.0).contains(&x)).then_some(0.0);
        let hit = ground_under_ray([0.0, 0.0, 10.0], [1.0, 0.0, -1.0], 100.0, holed)
            .expect("the ground resumes past the hole");
        assert!(hit[0] >= 8.0, "landed inside the hole: {hit:?}");
    }

    /// A camera inside a hillside returns its own origin, rather than marching
    /// on and pointing at whatever is beyond the hill.
    #[test]
    fn a_ray_that_starts_underground_answers_where_it_starts() {
        let hit = ground_under_ray([3.0, 4.0, -2.0], [1.0, 0.0, -1.0], 100.0, flat)
            .expect("an origin under the floor is itself the answer");
        assert!((hit[0] - 3.0).abs() < 1.0e-4, "{hit:?}");
        assert!((hit[1] - 4.0).abs() < 1.0e-4, "{hit:?}");
    }

    /// Degenerate inputs return `None` rather than looping or dividing by zero.
    #[test]
    fn a_ray_with_no_direction_or_no_reach_answers_nothing() {
        assert_eq!(ground_under_ray([0.0, 0.0, 10.0], [0.0, 0.0, 0.0], 100.0, flat), None);
        assert_eq!(ground_under_ray([0.0, 0.0, 10.0], [0.0, 0.0, -1.0], 0.0, flat), None);
        assert_eq!(
            ground_under_ray([0.0, 0.0, 10.0], [0.0, 0.0, -1.0], f32::NAN, flat),
            None
        );
    }
}

#[cfg(test)]
mod group_tests {
    use super::*;

    /// The floor a point stands on returns its own group's flags, not those of
    /// a roof over it.
    ///
    /// Two groups, stacked: a street at z = 0 flagged outdoor and a room's
    /// floor at z = 6 inside the same building. A probe at z = 1 is on the
    /// street and a probe at z = 7 is in the room, and neither may return the
    /// other's flags.
    #[test]
    fn a_building_floor_carries_the_flags_of_the_group_it_came_from() {
        let quad = |z: f32| {
            vec![[0.0, 0.0, z], [10.0, 0.0, z], [10.0, 10.0, z], [0.0, 10.0, z]]
        };
        let mut positions = quad(0.0);
        positions.extend(quad(6.0));
        let mesh = CollisionMesh {
            positions,
            indices: vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
            declared_triangles: 4,
            triangle_groups: vec![0, 0, 1, 1],
            group_flags: vec![0x8008, 0x2000],
        };
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let collider = Collider::place(&mesh, &identity);
        assert_eq!(collider.building_floor(5.0, 5.0, 1.0), Some((0.0, 0x8008)));
        assert_eq!(collider.building_floor(5.0, 5.0, 7.0), Some((6.0, 0x2000)));
        assert_eq!(collider.building_floor(20.0, 5.0, 7.0), None, "off the hull");

        // A hull with no groups, as every M2's is, returns `None`, so a crate
        // on a street cannot turn the street into a room.
        let bare = CollisionMesh {
            triangle_groups: Vec::new(),
            group_flags: Vec::new(),
            ..mesh
        };
        assert_eq!(Collider::place(&bare, &identity).building_floor(5.0, 5.0, 7.0), None);
    }
}
