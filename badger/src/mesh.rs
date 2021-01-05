use ash::vk;
use std::str::FromStr;
use std::error::Error;
use std::convert::TryFrom;
use std::collections::HashMap;

/*
 *  A mesh of vertices and their indices for indexed rendering.
 */
#[derive(Debug)]
pub struct Mesh<I : Index>
{
    pub vertices: Vec<Vertex>,
    pub indices:  Vec<I>
}

/*
 *  Vertices are each formed by a 3D position and an RGB colour. These are going to be
 *  copied directly into GPU memory, so it's important that they have a consistent layout.
 */
#[repr(C)]
#[derive(Debug)]
pub struct Vertex
{
    pub position : [f32 ; 3],
    pub normal   : [f32 ; 3],
    pub uv_coord : [f32 ; 2]
}

impl Vertex
{
    pub fn binding_descriptions() -> Vec<vk::VertexInputBindingDescription>
    {
        vec![vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()]
    }

    pub fn attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription>
    {
        vec![vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(memoffset::offset_of!(Vertex, position) as u32)
            .build(),

             vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(memoffset::offset_of!(Vertex, normal) as u32)
            .build(),

             vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(memoffset::offset_of!(Vertex, uv_coord) as u32)
            .build()]
    }
}

/*
 *  Without enabling any extensions, indices can be either 16-bit or 32-bit unsigned integers.
 *  Uses the private::Sealed supertrait to prevent clients from providing invalid implementations.
 */
pub trait Index : private::Sealed + std::convert::TryFrom<usize>
{
    const INDEX_TYPE : vk::IndexType;
    const INDEX_SIZE : u64;

    fn from_usize(index : usize) -> Result<Self, <Self as TryFrom<usize>>::Error>;
}

impl Index for u16
{
    const INDEX_TYPE : vk::IndexType = vk::IndexType::UINT16;
    const INDEX_SIZE : u64           = 2;

    fn from_usize(index : usize) -> Result<u16, <u16 as TryFrom<usize>>::Error>
    {
        u16::try_from(index)
    }
}

impl Index for u32
{
    const INDEX_TYPE : vk::IndexType = vk::IndexType::UINT32;
    const INDEX_SIZE : u64           = 4;

    fn from_usize(index : usize) -> Result<u32, <u32 as TryFrom<usize>>::Error>
    {
        u32::try_from(index)
    }
}

mod private
{
    pub trait Sealed {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
}

/*
 *  Attempt to parse a mesh from a Wavefront .obj file. Currently supports
 *  vertices, vertex texture coordinates, vertex normals, and triangular faces.
 */
impl<I : Index + FromStr> Mesh<I>
{
    #![allow(clippy::many_single_char_names)]
    pub fn parse_obj(obj : &str) -> Result<Mesh<I>, Box<dyn Error>>
    where <I as TryFrom<usize>>::Error : Error + 'static
    {
        let mut positions = Vec::new();
        let mut uv_coords = Vec::new();
        let mut normals   = Vec::new();
        let mut index_map = HashMap::new();

        let mut vertices = Vec::new();
        let mut indices  = Vec::new();

        for line in obj.lines()
        {
            match line.split_whitespace().collect::<Vec<_>>()[..]
            {
                ["v", x, y, z] =>
                {
                    positions.push([x.parse()?, y.parse()?, z.parse()?])
                },
                ["vt", u, v] =>
                {
                    uv_coords.push([u.parse()?, v.parse()?])
                },
                ["vn", x, y, z] =>
                {
                    normals.push([x.parse()?, y.parse()?, z.parse()?])
                },
                ["f", a, b, c] =>
                {
                    for is in [a, b, c].iter()
                    {
                        let (p, t, n) = match is.split('/').collect::<Vec<_>>()[..]
                        {
                            [p, t, n] => (p.parse::<usize>()? - 1, t.parse::<usize>()? - 1, n.parse::<usize>()? - 1),
                            _         => panic!("unsupported index format")
                        };

                        match index_map.get(&(p, t, n))
                        {
                            Some(&v) => indices.push(v),
                            None     =>
                            {
                                let index = vertices.len();
                                indices.push(index);
                                index_map.insert((p, t, n), index);

                                let vertex = Vertex
                                {
                                    position: positions[p],
                                    uv_coord: uv_coords[t],
                                    normal:   normals[n]
                                };
                                vertices.push(vertex);
                            }
                        }
                    }
                },
                _ => ()
            }
        }

        let indices = indices.into_iter().map(I::from_usize).collect::<Result<Vec<_>, _>>()?;
        Ok(Mesh { vertices, indices })
    }
}
