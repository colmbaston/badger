use shaderc::{ Compiler, ShaderKind };
use std::{ error::Error, fs::File, io::Write };

const VERT_PATH : &str = "shaders/shader.vert";
const FRAG_PATH : &str = "shaders/shader.frag";

fn main() -> Result<(), Box<dyn Error>>
{
    let out_dir      = std::env::var("OUT_DIR")?;
    let mut compiler = Compiler::new().expect("failed to initialise shader compiler");

    let vert_src = std::fs::read_to_string(VERT_PATH)?;
    let vert_bin = compiler.compile_into_spirv(&vert_src, ShaderKind::Vertex, VERT_PATH, "main", None)?;
    File::create(format!("{}/vert.spv", out_dir))?.write_all(vert_bin.as_binary_u8())?;

    let frag_src = std::fs::read_to_string(FRAG_PATH)?;
    let frag_bin = compiler.compile_into_spirv(&frag_src, ShaderKind::Fragment, FRAG_PATH, "main", None)?;
    File::create(format!("{}/frag.spv", out_dir))?.write_all(frag_bin.as_binary_u8())?;

    println!("cargo:rerun-if-changed={}", VERT_PATH);
    println!("cargo:rerun-if-changed={}", FRAG_PATH);

    Ok(())
}
