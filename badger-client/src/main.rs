use badger::{ Mesh, Engine, ClientMessage, Uniforms };
use std::{ error::Error, time::Instant, collections::HashSet };

use winit::
{
    dpi::{ PhysicalSize, PhysicalPosition },
    event_loop::{ EventLoopProxy, ControlFlow },
    event::{ Event, WindowEvent, DeviceEvent, MouseScrollDelta, VirtualKeyCode, MouseButton, ElementState }
};

mod camera;
use crate::camera::Camera;

#[repr(C)]
struct UniformData
{
    projection_view : glm::Mat4,
    light_direction : glm::Vec4,
    camera_position : glm::Vec4
}

const TAU  : f32               = std::f32::consts::PI * 2.0;
const SIZE : PhysicalSize<u32> = PhysicalSize { width: 800, height: 600 };

const VERT : &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vert.spv"));
const FRAG : &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/frag.spv"));

fn main() -> Result<(), Box<dyn Error>>
{
    let cube = Mesh::<u16>::parse_obj(include_str!("../assets/cube.obj"))?;

    unsafe
    {
        let (mut engine, proxy) = Engine::new(vec![cube], VERT, FRAG, SIZE)?;
        engine.run(event_handler(proxy));
    }

    Ok(())
}

/*
 *  The client is structured around passing this event handler to the engine.
 *  It returns a closure that can capture and mutate arbitrary client state.
 */
fn event_handler<'a>(proxy : EventLoopProxy<ClientMessage<UniformData>>)
-> impl 'a + FnMut(&Event<ClientMessage<UniformData>>, &mut ControlFlow)
{
    /*
     *  Specify the client state that will be used and mutated by events.
     */
    let mut window_focused   = false;
    let mut camera           = Camera::new();
    let mut keys_pressed     = HashSet::new();
    let mut prev_frame       = Instant::now();
    let mut paused           = false;
    let mut elapsed          = 0.0;
    let mut projection       = camera.projection(SIZE);
    let mut orbit_radius     = 2.0;

    move |event, control_flow|
    {
        match event
        {
            Event::MainEventsCleared =>
            {
                /*
                 *  Rendering is prompted here by generating a Render UserEvent
                 *  containing the UniformData uniform to be passed to the shaders.
                 */
                let now           = Instant::now();
                let frame_elapsed = (now - prev_frame).as_secs_f32();
                camera.update(frame_elapsed);
                if !paused { elapsed += frame_elapsed }
                prev_frame = now;

                let projection_view = projection * camera.view(orbit_radius);
                let light_rotation  = glm::rotate_x(&glm::identity(), TAU * elapsed / 20.0);
                let light_direction = glm::normalize(&(glm::mat4_to_mat3(&light_rotation) * glm::vec3(0.0, 0.0, 1.0)));
                let cube_model      = glm::rotate_z(&glm::translate(&glm::identity(), &glm::vec3(0.0, 0.0, 1.0)), TAU * (0.5 + elapsed / 10.0));

                let _ = proxy.send_event(ClientMessage::Render(Uniforms
                {
                    uniform: UniformData
                    {
                        projection_view,
                        light_direction: glm::vec3_to_vec4(&light_direction),
                        camera_position: glm::vec3_to_vec4(&camera.viewing_position(orbit_radius))
                    },
                    push_constants: vec![cube_model.iter().flat_map(|k| k.to_le_bytes().to_vec()).collect()]
                }));
            },
            Event::WindowEvent { event, .. } =>
            {
                match event
                {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(size)  =>  projection   = camera.projection(*size),
                    WindowEvent::KeyboardInput { input, .. } =>
                    {
                        /*
                         *  Keyboard WindowEvents are generated repeatedly while holding
                         *  down a key, so track which keys are currently pressed to only
                         *  generate a single event per keypress, untracking when released.
                         */
                        if window_focused
                        {
                            if input.state == ElementState::Pressed && !keys_pressed.insert(input.virtual_keycode) { return }

                            camera.keyboard_input(input);
                            if let ElementState::Pressed = input.state
                            {
                                match input.virtual_keycode
                                {
                                    Some(VirtualKeyCode::P) =>
                                    {
                                        paused = !paused;
                                    },
                                    Some(VirtualKeyCode::F) =>
                                    {
                                        let _ = proxy.send_event(ClientMessage::ToggleFullscreen);
                                    },
                                    Some(VirtualKeyCode::Escape) =>
                                    {
                                        window_focused = false;
                                        let _ = proxy.send_event(ClientMessage::WindowFocus(false));
                                    },
                                    _ => ()
                                }
                            }
                            else
                            {
                                keys_pressed.remove(&input.virtual_keycode);
                            }
                        }
                    },
                    WindowEvent::MouseWheel { delta, .. } =>
                    {
                        let y = match delta
                        {
                            MouseScrollDelta::LineDelta(_, y)                        => *y,
                            MouseScrollDelta::PixelDelta(PhysicalPosition { y, .. }) => *y as f32
                        };

                        orbit_radius = (orbit_radius - camera.scroll_sensitivity * y).min(20.0).max(2.0);
                    },
                    WindowEvent::Focused(false) =>
                    {
                        /*
                         *  Synthetic keyboard WindowEvents are generated when the window
                         *  gains or loses focus, so no need for keypress tracking here.
                         */
                        window_focused = false;
                        let _ = proxy.send_event(ClientMessage::WindowFocus(false));
                    },
                    WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } =>
                    {
                        window_focused = true;
                        let _ = proxy.send_event(ClientMessage::WindowFocus(true));
                    },
                    _ => ()
                }
            },
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta: (x, y), .. }, .. } =>
            {
                if window_focused { camera.mouse_delta(*x as f32, *y as f32) }
            },
            _ => ()
        }
    }
}
