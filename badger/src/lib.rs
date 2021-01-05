use std::{ error::Error, ffi::CString, io::Cursor };

use ash::
{
    vk,
    Entry,
    Instance,
    Device,
    extensions::khr::{ Surface, Swapchain },
    version::{ EntryV1_0, InstanceV1_0, DeviceV1_0 }
};

use winit::
{
    dpi::PhysicalSize,
    window::{ Window, WindowBuilder, Fullscreen },
    event_loop::{ EventLoop, EventLoopProxy, ControlFlow },
    event::{ Event, WindowEvent },
    platform::run_return::EventLoopExtRunReturn
};

mod mesh;
pub use crate::mesh::{ Mesh, Vertex, Index };

/*
 *  The Engine type holds all of the relevant window system and Vulkan resources.
 *  Clients have access to Engine::new() to construct an Engine, passing any
 *  vertices and SPIR-V shader code, and can then Engine::run() it with an
 *  event handler. Clients don't have direct access to any fields.
 */
pub struct Engine<U : 'static>
{
    current_frame : usize,
    models        : Vec<Model>,
    window        : WindowState<U>,
    vulkan        : VulkanState
}

struct WindowState<U : 'static>
{
    /*
     *  The EventLoop is contained within an Option<_> so that
     *  EventLoop::run_return() does not take ownership of the Engine.
     */
    handle       : Window,
    event_loop   : Option<EventLoop<ClientMessage<U>>>,
    resized_flag : bool
}

pub enum ClientMessage<U>
{
    ToggleFullscreen,
    WindowFocus(bool),
    Render(Uniforms<U>)
}

pub struct Uniforms<U>
{
    pub uniform:        U,
    pub push_constants: Vec<Vec<u8>>
}

struct VulkanState
{
    /*
     *  The Vulkan entry point isn't explicitly used after
     *  being packaged into the VulkanState, but it needs
     *  to be prevented from it from being dropped.
     *
     *  Swapchain recreation is flagged by destroying the
     *  old swapchain and setting the Option<_> to None.
     */
    #[allow(dead_code)]
    entry     : Entry,
    instance  : Instance,
    device    : DeviceState,
    surface   : SurfaceState,
    swapchain : Option<SwapchainState>
}

struct DeviceState
{
    loader                : Device,
    physical              : vk::PhysicalDevice,
    graphics_queue        : vk::Queue,
    command_pool          : vk::CommandPool,
    command_buffers       : Vec<vk::CommandBuffer>,
    descriptor_pool       : vk::DescriptorPool,
    descriptor_set_layout : vk::DescriptorSetLayout,
    shader_modules        : Vec<vk::ShaderModule>,
    uniform_buffers       : Vec<BufferState>,
    descriptor_sets       : Vec<vk::DescriptorSet>
}

struct Model
{
    vertex_buffer: BufferState,
    index_buffer:  BufferState,
    index_type:    vk::IndexType,
    index_size:    u64
}

struct BufferState
{
    handle : vk::Buffer,
    memory : vk::DeviceMemory,
    size   : vk::DeviceSize
}

struct SurfaceState
{
    loader : Surface,
    handle : vk::SurfaceKHR
}

/*
 *  Responsible for not only the swapchain, but any
 *  resources which are created along with the swapchain.
 */
struct SwapchainState
{
    loader           : Swapchain,
    handle           : vk::SwapchainKHR,
    image_extent     : vk::Extent2D,
    image_views      : Vec<vk::ImageView>,
    depth_image      : vk::Image,
    depth_image_view : vk::ImageView,
    depth_buffer     : vk::DeviceMemory,
    pipeline         : PipelineState,
    sync             : SyncState
}

struct PipelineState
{
    handle          : vk::Pipeline,
    layout          : vk::PipelineLayout,
    render_pass     : vk::RenderPass,
    framebuffers    : Vec<vk::Framebuffer>
}

struct SyncState
{
    images_available : Vec<vk::Semaphore>,
    renders_finished : Vec<vk::Semaphore>,
    in_flight_fences : Vec<vk::Fence>,
    images_in_flight : Vec<vk::Fence>
}

/*
 *  The maximum number of command
 *  buffers running concurrently.
 */
const FRAMES_IN_FLIGHT : usize = 2;

impl<U> Engine<U>
{
    /*
     *  Takes an event handler from the client, which may be a closure capturing mutable
     *  client state, and wraps it in an event handler responsible for the render loop.
     */
    pub unsafe fn run<F>(&mut self, mut event_handler : F)
    where F : FnMut(&Event<ClientMessage<U>>, &mut ControlFlow)
    {
        self.window.handle.set_visible(true);
        if let Some(mut event_loop) = self.window.event_loop.take()
        {
            event_loop.run_return(|event, _, control_flow|
            {
                event_handler(&event, control_flow);

                match event
                {
                    Event::UserEvent(user_event) =>
                    {
                        match user_event
                        {
                            ClientMessage::ToggleFullscreen =>
                            {
                                self.window.handle.set_fullscreen(match self.window.handle.fullscreen()
                                {
                                    None    => Some(Fullscreen::Borderless(self.window.handle.primary_monitor())),
                                    Some(_) => None
                                })
                            },
                            ClientMessage::WindowFocus(f) =>
                            {
                                let _ = self.window.handle.set_cursor_grab(f);
                                self.window.handle.set_cursor_visible(!f);
                            },
                            ClientMessage::Render(uniforms) =>
                            {
                                if let Err(err) = self.create_swapchain().and_then(|_| Ok(self.render(uniforms)?))
                                {
                                    eprintln!("{:?}", err);
                                    *control_flow = ControlFlow::Exit;
                                }
                            }
                        }
                    }
                    Event::WindowEvent { event, .. } =>
                    {
                        if let WindowEvent::Resized(_) = event
                        {
                            self.window.resized_flag = true;
                        }
                    },
                    _ => ()
                }
            })
        }
        self.window.handle.set_visible(false);
    }

    /*
     *  Render a single frame, acquiring an image from the swapchain
     *  and passing it onto the present queue. Will prompt a destruction
     *  of the swapchain if the window surface becomes unusable.
     */
    unsafe fn render(&mut self, Uniforms { uniform, push_constants } : Uniforms<U>) -> Result<(), Box<dyn Error>>
    {
        if let Some(swapchain) = &mut self.vulkan.swapchain
        {
            let current_frame = self.current_frame % FRAMES_IN_FLIGHT;
            self.vulkan.device.loader.wait_for_fences(&[swapchain.sync.in_flight_fences[current_frame]], true, std::u64::MAX)?;

            let result = swapchain.loader.acquire_next_image(swapchain.handle, std::u64::MAX, swapchain.sync.images_available[current_frame], vk::Fence::null());
            if let Err(vk::Result::ERROR_OUT_OF_DATE_KHR) = result { return self.destroy_swapchain() }
            let image_index = result?.0;

            /*
             *  Update the uniform buffer with the uniform provided
             *  by the user event that prompted this render call.
             */
            let current_uniform_buffer = &self.vulkan.device.uniform_buffers[current_frame];
            let mem_ptr = self.vulkan.device.loader.map_memory(current_uniform_buffer.memory, 0, current_uniform_buffer.size, vk::MemoryMapFlags::empty())?;
            {
                mem_ptr.copy_from((&uniform as *const U).cast(), current_uniform_buffer.size as usize);
            }
            self.vulkan.device.loader.unmap_memory(current_uniform_buffer.memory);

            let image_fence = &mut swapchain.sync.images_in_flight[image_index as usize];
            if *image_fence != vk::Fence::null()
            {
                self.vulkan.device.loader.wait_for_fences(&[*image_fence], true, std::u64::MAX)?;
            }
            *image_fence = swapchain.sync.in_flight_fences[current_frame];
            self.vulkan.device.loader.reset_fences(&[swapchain.sync.in_flight_fences[current_frame]])?;

            /*
             * Record a command buffer to execute the render pass.
             */
            let clear_values = [vk::ClearValue { color:         vk::ClearColorValue        { float32: [0.125, 0.125, 0.125, 1.0] }},
                                vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 }}];
            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                                        .render_pass(swapchain.pipeline.render_pass)
                                        .framebuffer(swapchain.pipeline.framebuffers[image_index as usize])
                                        .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: swapchain.image_extent })
                                        .clear_values(&clear_values)
                                        .build();

            let buffer                    = self.vulkan.device.command_buffers[current_frame];
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT).build();
            self.vulkan.device.loader.begin_command_buffer(buffer, &command_buffer_begin_info)?;
            {
                self.vulkan.device.loader.cmd_begin_render_pass(buffer, &render_pass_begin_info, vk::SubpassContents::INLINE);
                {
                    self.vulkan.device.loader.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::GRAPHICS, swapchain.pipeline.handle);
                    self.vulkan.device.loader.cmd_bind_descriptor_sets(buffer, vk::PipelineBindPoint::GRAPHICS, swapchain.pipeline.layout, 0, &[self.vulkan.device.descriptor_sets[current_frame]], &[]);

                    for (Model { vertex_buffer, index_buffer, index_type, index_size }, push_constant)
                    in self.models.iter().zip(push_constants.iter())
                    {
                        if !push_constant.is_empty()
                        {
                            self.vulkan.device.loader.cmd_push_constants(buffer, swapchain.pipeline.layout, vk::ShaderStageFlags::VERTEX, 0, &push_constant);
                        }
                        self.vulkan.device.loader.cmd_bind_vertex_buffers(buffer, 0, &[vertex_buffer.handle], &[0]);
                        self.vulkan.device.loader.cmd_bind_index_buffer(buffer, index_buffer.handle, 0, *index_type);
                        self.vulkan.device.loader.cmd_draw_indexed(buffer, (index_buffer.size / index_size) as u32, 1, 0, 0, 0);
                    }
                }
                self.vulkan.device.loader.cmd_end_render_pass(buffer);
            }
            self.vulkan.device.loader.end_command_buffer(buffer)?;

            /*
             *  Submit the command buffer to the graphics queue.
             */
            let command_buffers   = [buffer];
            let wait_semaphores   = [swapchain.sync.images_available[current_frame]];
            let signal_semaphores = [swapchain.sync.renders_finished[current_frame]];
            let submit_info       = vk::SubmitInfo::builder()
                                   .command_buffers(&command_buffers)
                                   .wait_dst_stage_mask(&[vk::PipelineStageFlags::TOP_OF_PIPE])
                                   .wait_semaphores(&wait_semaphores)
                                   .signal_semaphores(&signal_semaphores)
                                   .build();

            let image_indices = [image_index];
            let swapchains    = [swapchain.handle];
            let present_info  = vk::PresentInfoKHR::builder()
                               .image_indices(&image_indices)
                               .swapchains(&swapchains)
                               .wait_semaphores(&signal_semaphores)
                               .build();


            self.vulkan.device.loader.queue_submit(self.vulkan.device.graphics_queue, &[submit_info], swapchain.sync.in_flight_fences[current_frame])?;
            let result = swapchain.loader.queue_present(self.vulkan.device.graphics_queue, &present_info);
            if result == Err(vk::Result::ERROR_OUT_OF_DATE_KHR) || result? || self.window.resized_flag
            {
                self.window.resized_flag = false;
                self.destroy_swapchain()?;
            }

            self.current_frame += 1;
        }

        Ok(())
    }

    /*
     *  Initialise all of the required window system and Vulkan resources.
     */
    pub unsafe fn new<I : Index>(meshes : Vec<Mesh<I>>, vert_shader : &[u8], frag_shader : &[u8], window_size : PhysicalSize<u32>)
    -> Result<(Engine<U>, EventLoopProxy<ClientMessage<U>>), Box<dyn Error>>
    {
        /*
         *  A winit event loop and window.
         */
        let event_loop       = EventLoop::with_user_event();
        let event_loop_proxy = event_loop.create_proxy();
        let window_handle    = WindowBuilder::new()
                              .with_title("Badger")
                              .with_inner_size(window_size)
                              .with_visible(false)
                              .build(&event_loop)?;

        /*
         *  The Vulkan library entry point, instance, and a surface connected to
         *  the winit window. Validation layers are requested in debug builds.
         */
        let entry            = Entry::new()?;
        let engine_name      = CString::new("Badger")?;
        let application_name = CString::new("Badger")?;
        let application_info = vk::ApplicationInfo::builder()
                              .engine_name(&engine_name)
                              .engine_version(vk::make_version(0, 1, 0))
                              .application_name(&application_name)
                              .application_version(vk::make_version(0, 1, 0))
                              .api_version(vk::make_version(1, 0, 0))
                              .build();

        let instance_extensions = ash_window::enumerate_required_extensions(&window_handle)?
                                 .into_iter().map(|x| x.as_ptr()).collect::<Vec<_>>();

        let instance_info = vk::InstanceCreateInfo::builder()
                           .application_info(&application_info)
                           .enabled_extension_names(&instance_extensions);

        #[cfg(debug_assertions)]
        let layers = [CString::new("VK_LAYER_KHRONOS_validation")?];
        #[cfg(debug_assertions)]
        let layers = layers.iter().map(|x| x.as_ptr()).collect::<Vec<_>>();
        #[cfg(debug_assertions)]
        let instance_info = instance_info.enabled_layer_names(&layers);

        let instance       = entry.create_instance(&instance_info, None)?;
        let surface        = Surface::new(&entry, &instance);
        let surface_handle = ash_window::create_surface(&entry, &instance, &window_handle, None)?;

        /*
         *  Select a physical device with a queue family capable of presenting graphics to the
         *  window surface. Create the logical device, get the queue handle, and create a command pool.
         */
        let (physical, queue_family) = instance.enumerate_physical_devices()?.into_iter().find_map(|physical|
        {
            instance.get_physical_device_queue_family_properties(physical).into_iter().enumerate().find_map(|(index, properties)|
            {
                let index = index as u32;
                if !properties.queue_flags.contains(vk::QueueFlags::GRAPHICS) { return None }
                match surface.get_physical_device_surface_support(physical, index, surface_handle)
                {
                    Ok(false) => None,
                    Ok(true)  => Some(Ok((physical, index))),
                    Err(err)  => Some(Err(err))
                }
            })
        })
        .expect("no suitable physical devices found")?;

        let queue_info        = [vk::DeviceQueueCreateInfo::builder()
                                .queue_family_index(queue_family)
                                .queue_priorities(&[1.0])
                                .build()];
        let device_extensions = [Swapchain::name().as_ptr(), /* vk::KhrMaintenance1Fn::name().as_ptr() */];
        let device_info       =  vk::DeviceCreateInfo::builder()
                                .queue_create_infos(&queue_info)
                                .enabled_extension_names(&device_extensions);

        #[cfg(debug_assertions)]
        let device_info    = device_info.enabled_layer_names(&layers);
        let device         = instance.create_device(physical, &device_info, None)?;
        let graphics_queue = device.get_device_queue(queue_family, 0);

        let command_pool_info = vk::CommandPoolCreateInfo::builder()
                               .queue_family_index(queue_family)
                               .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                               .build();
        let command_pool      = device.create_command_pool(&command_pool_info, None)?;

        /*
         *  Allocate command buffers for drawing commands.
         */
        let command_buffer_info = vk::CommandBufferAllocateInfo::builder()
                                 .level(vk::CommandBufferLevel::PRIMARY)
                                 .command_pool(command_pool)
                                 .command_buffer_count(FRAMES_IN_FLIGHT as u32)
                                 .build();

        let command_buffers = device.allocate_command_buffers(&command_buffer_info)?;

        /*
         *  Closures to find and allocate suitable memory on the
         *  device for a buffer and to copy data between buffers.
         */
        let create_buffer = |size : vk::DeviceSize, usage : vk::BufferUsageFlags, properties : vk::MemoryPropertyFlags| -> Result<BufferState, Box<dyn Error>>
        {
            let buffer_info = vk::BufferCreateInfo::builder()
                             .size(size)
                             .usage(usage)
                             .sharing_mode(vk::SharingMode::EXCLUSIVE)
                             .build();

            let buffer_handle = device.create_buffer(&buffer_info, None)?;
            let requirements  = device.get_buffer_memory_requirements(buffer_handle);
            let memory_type   = find_memory_type(&instance, physical, requirements, properties);

            let alloc_info = vk::MemoryAllocateInfo::builder()
                            .allocation_size(requirements.size)
                            .memory_type_index(memory_type)
                            .build();

            let buffer_memory = device.allocate_memory(&alloc_info, None)?;
            device.bind_buffer_memory(buffer_handle, buffer_memory, 0)?;

            Ok(BufferState { handle: buffer_handle, memory: buffer_memory, size })
        };

        let copy_buffer = |src_buffer : &BufferState, dst_buffer : &BufferState| -> Result<(), Box<dyn Error>>
        {
            let copy_command_buffer_info = vk::CommandBufferAllocateInfo::builder()
                                          .level(vk::CommandBufferLevel::PRIMARY)
                                          .command_pool(command_pool)
                                          .command_buffer_count(1)
                                          .build();

            let command_buffer = device.allocate_command_buffers(&copy_command_buffer_info)?[0];
            let begin_info     = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT).build();
            device.begin_command_buffer(command_buffer, &begin_info)?;
            {
                let buffer_copy = vk::BufferCopy::builder().size(src_buffer.size).build();
                device.cmd_copy_buffer(command_buffer, src_buffer.handle, dst_buffer.handle, &[buffer_copy]);
            }
            device.end_command_buffer(command_buffer)?;

            let submit_info = vk::SubmitInfo::builder().command_buffers(&[command_buffer]).build();
            device.queue_submit(graphics_queue, &[submit_info], vk::Fence::null())?;
            device.queue_wait_idle(graphics_queue)?;
            device.free_command_buffers(command_pool, &[command_buffer]);

            Ok(())
        };

        /*
         *  Use the closures to allocate a vertex buffer and index buffer for each model, and uniform
         *  buffers. For the vertex and index buffers, the data is copied from a staging buffer first.
         */
        let models = meshes.into_iter().map::<Result<_, Box<dyn Error>>, _>(|mesh : Mesh<I>|
        {
            let vertices_bytes = (std::mem::size_of::<Vertex>() * mesh.vertices.len()) as u64;
            let staging_buffer = create_buffer(vertices_bytes,
                                               vk::BufferUsageFlags::TRANSFER_SRC,
                                               vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
            let vertex_buffer  = create_buffer(vertices_bytes,
                                               vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                                               vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

            let mem_ptr = device.map_memory(staging_buffer.memory, 0, staging_buffer.size, vk::MemoryMapFlags::empty())?;
            {
                mem_ptr.copy_from(mesh.vertices.as_ptr().cast(), staging_buffer.size as usize);
            }
            device.unmap_memory(staging_buffer.memory);

            copy_buffer(&staging_buffer, &vertex_buffer)?;
            device.destroy_buffer(staging_buffer.handle, None);
            device.free_memory(staging_buffer.memory, None);

            let indices_bytes  = (std::mem::size_of::<I>() * mesh.indices.len()) as u64;
            let staging_buffer = create_buffer(indices_bytes,
                                               vk::BufferUsageFlags::TRANSFER_SRC,
                                               vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
            let index_buffer   = create_buffer(indices_bytes,
                                               vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
                                               vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

            let mem_ptr = device.map_memory(staging_buffer.memory, 0, staging_buffer.size, vk::MemoryMapFlags::empty())?;
            {
                mem_ptr.copy_from(mesh.indices.as_ptr().cast(), staging_buffer.size as usize);
            }
            device.unmap_memory(staging_buffer.memory);

            copy_buffer(&staging_buffer, &index_buffer)?;
            device.destroy_buffer(staging_buffer.handle, None);
            device.free_memory(staging_buffer.memory, None);

            Ok(Model { vertex_buffer, index_buffer, index_type: I::INDEX_TYPE, index_size: I::INDEX_SIZE })
        })
        .collect::<Result<Vec<_>, _>>()?;

        let uniform_buffer_bytes = std::mem::size_of::<U>() as u64;
        let uniform_buffers = (0 .. FRAMES_IN_FLIGHT).map(|_|
        {
            create_buffer(uniform_buffer_bytes,
                          vk::BufferUsageFlags::UNIFORM_BUFFER,
                          vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
        })
        .collect::<Result<Vec<_>, _>>()?;

        /*
         *  Descriptor set layout, descriptor pool, and descriptor sets.
         */
        let descriptor_set_layout_binding = vk::DescriptorSetLayoutBinding::builder()
                                           .binding(0)
                                           .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                           .descriptor_count(1)
                                           .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                                           .build();

        let descriptor_set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
                                        .bindings(&[descriptor_set_layout_binding])
                                        .build();

        let descriptor_set_layout = device.create_descriptor_set_layout(&descriptor_set_layout_info, None)?;

        let descriptor_pool_size = vk::DescriptorPoolSize::builder()
                                  .ty(vk::DescriptorType::UNIFORM_BUFFER)
                                  .descriptor_count(uniform_buffers.len() as u32)
                                  .build();

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
                                  .pool_sizes(&[descriptor_pool_size])
                                  .max_sets(uniform_buffers.len() as u32)
                                  .build();

        let descriptor_pool = device.create_descriptor_pool(&descriptor_pool_info, None)?;

        let descriptor_set_layouts    = uniform_buffers.iter().map(|_| descriptor_set_layout).collect::<Vec<_>>();
        let descriptor_set_alloc_info = vk::DescriptorSetAllocateInfo::builder()
                                       .descriptor_pool(descriptor_pool)
                                       .set_layouts(&descriptor_set_layouts)
                                       .build();

        let descriptor_sets = device.allocate_descriptor_sets(&descriptor_set_alloc_info)?;

        for (buffer, set) in uniform_buffers.iter().zip(descriptor_sets.iter())
        {
            let buffer_info = vk::DescriptorBufferInfo::builder()
                             .buffer(buffer.handle)
                             .offset(0)
                             .range(buffer.size)
                             .build();

            let write_set = vk::WriteDescriptorSet::builder()
                           .dst_set(*set)
                           .dst_binding(0)
                           .dst_array_element(0)
                           .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                           .buffer_info(&[buffer_info])
                           .build();

            device.update_descriptor_sets(&[write_set], &[]);
        }

        /*
         *  Shader modules.
         */
        let mut vert_spv    = Cursor::new(vert_shader);
        let     vert_spv    = ash::util::read_spv(&mut vert_spv)?;
        let     vert_info   = vk::ShaderModuleCreateInfo::builder().code(&vert_spv).build();
        let     vert_module = device.create_shader_module(&vert_info, None)?;

        let mut frag_spv    = Cursor::new(frag_shader);
        let     frag_spv    = ash::util::read_spv(&mut frag_spv)?;
        let     frag_info   = vk::ShaderModuleCreateInfo::builder().code(&frag_spv).build();
        let     frag_module = device.create_shader_module(&frag_info, None)?;

        let engine = Engine
        {
            current_frame: 0,
            models,
            window: WindowState
            {
                handle:       window_handle,
                event_loop:   Some(event_loop),
                resized_flag: false
            },
            vulkan: VulkanState
            {
                entry,
                instance,
                device: DeviceState
                {
                    loader: device,
                    physical,
                    graphics_queue,
                    command_pool,
                    command_buffers,
                    descriptor_pool,
                    shader_modules: vec![vert_module, frag_module],
                    descriptor_set_layout,
                    uniform_buffers,
                    descriptor_sets
                },
                surface: SurfaceState
                {
                    loader: surface,
                    handle: surface_handle

                },
                swapchain: None
            }
        };

        Ok((engine, event_loop_proxy))
    }

    /*
     *  If no swapchain is present and the area of
     *  the window is non-zero, create a new swapchain.
     */
    unsafe fn create_swapchain(&mut self) -> Result<(), Box<dyn Error>>
    {
        let PhysicalSize { width, height } = self.window.handle.inner_size();
        if self.vulkan.swapchain.is_some() || width == 0 || height == 0
        {
            return Ok(())
        }

        /*
         *  Swapchain configuration.
         */
        let surface_capabilities = self.vulkan.surface.loader.get_physical_device_surface_capabilities(self.vulkan.device.physical, self.vulkan.surface.handle)?;
        let image_extent         = match surface_capabilities.current_extent.width
        {
            std::u32::MAX => vk::Extent2D { width, height },
            _             => surface_capabilities.current_extent
        };

        let image_count =
        {
            let desired = surface_capabilities.min_image_count + 1;
            match surface_capabilities.max_image_count
            {
                0   => desired,
                max => desired.min(max)
            }
        };

        let image_format = self.vulkan.surface.loader.get_physical_device_surface_formats(self.vulkan.device.physical, self.vulkan.surface.handle)?
                          .into_iter().next().map(|mut surface_format|
                           {
                               if let vk::Format::UNDEFINED = surface_format.format
                               {
                                   surface_format.format = vk::Format::B8G8R8_SRGB
                               }
                               surface_format
                           })
                          .expect("no suitable surface format found");

        let present_mode = self.vulkan.surface.loader.get_physical_device_surface_present_modes(self.vulkan.device.physical, self.vulkan.surface.handle)?
                          .into_iter().find(|&mode| mode == vk::PresentModeKHR::MAILBOX).unwrap_or(vk::PresentModeKHR::FIFO);

        let swapchain_info = vk::SwapchainCreateInfoKHR::builder()
                            .surface(self.vulkan.surface.handle)
                            .min_image_count(image_count)
                            .image_format(image_format.format)
                            .image_color_space(image_format.color_space)
                            .image_extent(image_extent)
                            .image_array_layers(1)
                            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                            .pre_transform(surface_capabilities.current_transform)
                            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                            .present_mode(present_mode)
                            .clipped(true)
                            .build();

        /*
         *  The swapchain itself, its images, and their image views.
         */
        let swapchain_loader = Swapchain::new(&self.vulkan.instance, &self.vulkan.device.loader);
        let swapchain_handle = swapchain_loader.create_swapchain(&swapchain_info, None)?;
        let images           = swapchain_loader.get_swapchain_images(swapchain_handle)?;
        let image_views      = images.iter().map(|&image|
        {
            let image_view_info = vk::ImageViewCreateInfo::builder()
                                 .image(image)
                                 .view_type(vk::ImageViewType::TYPE_2D)
                                 .format(image_format.format)
                                 .components(vk::ComponentMapping
                                  {
                                      r: vk::ComponentSwizzle::R,
                                      g: vk::ComponentSwizzle::G,
                                      b: vk::ComponentSwizzle::B,
                                      a: vk::ComponentSwizzle::A
                                  })
                                 .subresource_range(vk::ImageSubresourceRange
                                  {
                                      aspect_mask:      vk::ImageAspectFlags::COLOR,
                                      base_mip_level:   0,
                                      level_count:      1,
                                      base_array_layer: 0,
                                      layer_count:      1
                                  })
                                 .build();

            self.vulkan.device.loader.create_image_view(&image_view_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

        /*
         *  Resources related to depth buffering.
         */
        let depth_format = vec![vk::Format::D32_SFLOAT].into_iter().find(|&format|
        {
            let properties = self.vulkan.instance.get_physical_device_format_properties(self.vulkan.device.physical, format);
            properties.optimal_tiling_features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        })
        .expect("no suitable depth buffer format");

        let depth_image_info = vk::ImageCreateInfo::builder()
                              .image_type(vk::ImageType::TYPE_2D)
                              .extent(vk::Extent3D { width:  image_extent.width, height: image_extent.height, depth:  1 })
                              .mip_levels(1)
                              .array_layers(1)
                              .format(depth_format)
                              .tiling(vk::ImageTiling::OPTIMAL)
                              .initial_layout(vk::ImageLayout::UNDEFINED)
                              .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                              .samples(vk::SampleCountFlags::TYPE_1)
                              .build();

        let depth_image = self.vulkan.device.loader.create_image(&depth_image_info, None)?;

        let depth_mem_requirements  = self.vulkan.device.loader.get_image_memory_requirements(depth_image);
        let depth_buffer_alloc_info = vk::MemoryAllocateInfo::builder()
                                     .allocation_size(depth_mem_requirements.size)
                                     .memory_type_index(find_memory_type(&self.vulkan.instance,
                                                                         self.vulkan.device.physical,
                                                                         depth_mem_requirements,
                                                                         vk::MemoryPropertyFlags::DEVICE_LOCAL))
                                     .build();

        let depth_buffer = self.vulkan.device.loader.allocate_memory(&depth_buffer_alloc_info, None)?;
        self.vulkan.device.loader.bind_image_memory(depth_image, depth_buffer, 0)?;

        let depth_image_view_info = vk::ImageViewCreateInfo::builder()
                                   .image(depth_image)
                                   .view_type(vk::ImageViewType::TYPE_2D)
                                   .format(depth_format)
                                   .subresource_range(vk::ImageSubresourceRange
                                    {
                                        aspect_mask:      vk::ImageAspectFlags::DEPTH,
                                        base_mip_level:   0,
                                        level_count:      1,
                                        base_array_layer: 0,
                                        layer_count:      1
                                    })
                                   .build();

        let depth_image_view = self.vulkan.device.loader.create_image_view(&depth_image_view_info, None)?;

        /*
         *  Pipeline layout and render pass configuration.
         */
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
                                  .set_layouts(&[self.vulkan.device.descriptor_set_layout])
                                  .push_constant_ranges(&[vk::PushConstantRange
                                   {
                                       stage_flags: vk::ShaderStageFlags::VERTEX,
                                       offset:      0,
                                       size:        128
                                   }])
                                  .build();

        let pipeline_layout = self.vulkan.device.loader.create_pipeline_layout(&pipeline_layout_info, None)?;

        let render_pass_attachments = [vk::AttachmentDescription::builder()
                                      .format(image_format.format)
                                      .samples(vk::SampleCountFlags::TYPE_1)
                                      .load_op(vk::AttachmentLoadOp::CLEAR)
                                      .store_op(vk::AttachmentStoreOp::STORE)
                                      .initial_layout(vk::ImageLayout::UNDEFINED)
                                      .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                                      .build(),

                                       vk::AttachmentDescription::builder()
                                      .format(depth_format)
                                      .samples(vk::SampleCountFlags::TYPE_1)
                                      .load_op(vk::AttachmentLoadOp::CLEAR)
                                      .store_op(vk::AttachmentStoreOp::DONT_CARE)
                                      .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                                      .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                                      .initial_layout(vk::ImageLayout::UNDEFINED)
                                      .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                      .build()];

        let color_attachment_refs = [vk::AttachmentReference::builder()
                                    .attachment(0)
                                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                    .build()];

        let depth_attachment_ref = vk::AttachmentReference::builder()
                                  .attachment(1)
                                  .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                  .build();

        let subpasses = [vk::SubpassDescription::builder()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .color_attachments(&color_attachment_refs)
                        .depth_stencil_attachment(&depth_attachment_ref)
                        .build()];

        let render_pass_info = vk::RenderPassCreateInfo::builder()
                              .attachments(&render_pass_attachments)
                              .subpasses(&subpasses)
                              .build();

        let render_pass = self.vulkan.device.loader.create_render_pass(&render_pass_info, None)?;

        let binding_descriptions       = Vertex::binding_descriptions();
        let attribute_descriptions     = Vertex::attribute_descriptions();
        let vertex_input_state_info    = vk::PipelineVertexInputStateCreateInfo::builder()
                                        .vertex_binding_descriptions(&binding_descriptions)
                                        .vertex_attribute_descriptions(&attribute_descriptions)
                                        .build();
        let vertex_input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
                                        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                                        .primitive_restart_enable(false)
                                        .build();

        /*
         *  Rasterisation, multisampling, and colour blending configuration.
         */
        let viewports = [vk::Viewport::builder()
                        .x(0.0)
                        .y(0.0)
                        .width(image_extent.width as f32)
                        .height(image_extent.height as f32)
                        .min_depth(0.0)
                        .max_depth(1.0)
                        .build()];

        let scissors = [vk::Rect2D::builder()
                       .offset(vk::Offset2D { x: 0, y: 0 })
                       .extent(image_extent)
                       .build()];

        let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
                                 .viewports(&viewports)
                                 .scissors(&scissors)
                                 .build();

        let raster_info = vk::PipelineRasterizationStateCreateInfo::builder()
                         .depth_clamp_enable(false)
                         .depth_bias_enable(false)
                         .rasterizer_discard_enable(false)
                         .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                         .polygon_mode(vk::PolygonMode::FILL)
                         .line_width(1.0)
                         .cull_mode(vk::CullModeFlags::BACK)
                         .build();

        let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
                              .sample_shading_enable(false)
                              .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                              .build();

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState::builder()
                                            .color_write_mask(vk::ColorComponentFlags::all())
                                            .blend_enable(false)
                                            .build()];

        let color_blend_info = vk::PipelineColorBlendStateCreateInfo::builder()
                              .logic_op_enable(false)
                              .attachments(&color_blend_attachment_states)
                              .build();

        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
                                .depth_test_enable(true)
                                .depth_write_enable(true)
                                .depth_compare_op(vk::CompareOp::LESS)
                                .depth_bounds_test_enable(false)
                                .build();

        /*
         *  Shader configuration.
         */
        let shader_entry    = &CString::new("main")?;
        let vert_stage_info = vk::PipelineShaderStageCreateInfo::builder()
                             .stage(vk::ShaderStageFlags::VERTEX)
                             .module(self.vulkan.device.shader_modules[0])
                             .name(shader_entry)
                             .build();
        let frag_stage_info = vk::PipelineShaderStageCreateInfo::builder()
                             .stage(vk::ShaderStageFlags::FRAGMENT)
                             .module(self.vulkan.device.shader_modules[1])
                             .name(shader_entry)
                             .build();

        let shader_stages = [vert_stage_info, frag_stage_info];

        /*
         *  The graphics pipeline and framebuffer configuration.
         */
        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
                           .stages(&shader_stages)
                           .vertex_input_state(&vertex_input_state_info)
                           .input_assembly_state(&vertex_input_assembly_info)
                           .viewport_state(&viewport_state_info)
                           .rasterization_state(&raster_info)
                           .multisample_state(&multisample_info)
                           .color_blend_state(&color_blend_info)
                           .depth_stencil_state(&depth_stencil_info)
                           .layout(pipeline_layout)
                           .render_pass(render_pass)
                           .build();

        let pipeline_handle = match self.vulkan.device.loader.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        {
            Ok(pipelines) => pipelines[0],
            Err(err)      => return Err(Box::new(err.1)),
        };

        let framebuffers = image_views.iter().map(|&image_view|
        {
            let attachments      = [image_view, depth_image_view];
            let framebuffer_info = vk::FramebufferCreateInfo::builder()
                                  .render_pass(render_pass)
                                  .attachments(&attachments)
                                  .width(image_extent.width)
                                  .height(image_extent.height)
                                  .layers(1)
                                  .build();

            self.vulkan.device.loader.create_framebuffer(&framebuffer_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

        /*
         *  Initialise the render loop synchronisation objects.
         */
        let semaphore_info   = vk::SemaphoreCreateInfo::default();
        let fence_info       = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED).build();
        let images_available = (0 .. FRAMES_IN_FLIGHT).map(|_| self.vulkan.device.loader.create_semaphore(&semaphore_info, None)).collect::<Result<Vec<_>, _>>()?;
        let renders_finished = (0 .. FRAMES_IN_FLIGHT).map(|_| self.vulkan.device.loader.create_semaphore(&semaphore_info, None)).collect::<Result<Vec<_>, _>>()?;
        let in_flight_fences = (0 .. FRAMES_IN_FLIGHT).map(|_| self.vulkan.device.loader.create_fence(&fence_info, None)).collect::<Result<Vec<_>, _>>()?;
        let images_in_flight = (0 .. images.len()).map(|_| vk::Fence::null()).collect();

        self.vulkan.swapchain = Some(SwapchainState
        {
            loader: swapchain_loader,
            handle: swapchain_handle,
            image_extent,
            image_views,
            depth_image,
            depth_image_view,
            depth_buffer,
            pipeline: PipelineState
            {
                handle: pipeline_handle,
                layout: pipeline_layout,
                render_pass,
                framebuffers,

            },
            sync: SyncState
            {
                images_available,
                renders_finished,
                in_flight_fences,
                images_in_flight
            }
        });

        Ok(())
    }

    /*
     *  Wait until the device has finished any rendering in-progress,
     *  then destroy the swapchain and any dependent resources.
     */
    unsafe fn destroy_swapchain(&mut self) -> Result<(), Box<dyn Error>>
    {
        self.vulkan.device.loader.device_wait_idle()?;

        if let Some(swapchain) = self.vulkan.swapchain.take()
        {
            swapchain.loader.destroy_swapchain(swapchain.handle, None);
            swapchain.image_views.iter().for_each(|&image_view| self.vulkan.device.loader.destroy_image_view(image_view, None));
            swapchain.pipeline.framebuffers.iter().for_each(|&framebuffer| self.vulkan.device.loader.destroy_framebuffer(framebuffer, None));
            swapchain.sync.images_available.iter().for_each(|&semaphore| self.vulkan.device.loader.destroy_semaphore(semaphore, None));
            swapchain.sync.renders_finished.iter().for_each(|&semaphore| self.vulkan.device.loader.destroy_semaphore(semaphore, None));
            swapchain.sync.in_flight_fences.iter().for_each(|&fence| self.vulkan.device.loader.destroy_fence(fence, None));
            self.vulkan.device.loader.destroy_image(swapchain.depth_image, None);
            self.vulkan.device.loader.destroy_image_view(swapchain.depth_image_view, None);
            self.vulkan.device.loader.free_memory(swapchain.depth_buffer, None);
            self.vulkan.device.loader.destroy_render_pass(swapchain.pipeline.render_pass, None);
            self.vulkan.device.loader.destroy_pipeline_layout(swapchain.pipeline.layout, None);
            self.vulkan.device.loader.destroy_pipeline(swapchain.pipeline.handle, None);
        }

        Ok(())
    }
}

/*
 *  Free all of the resources managed by the Engine type.
 */
impl<U> Drop for Engine<U>
{
    fn drop(&mut self)
    {
        unsafe
        {
            /*
             *  Destroying the swapchain will first wait until the device
             *  is idle, so there's no need to repeat it explicitly here.
             */
            self.destroy_swapchain().unwrap();

            self.vulkan.device.loader.destroy_descriptor_set_layout(self.vulkan.device.descriptor_set_layout, None);
            self.vulkan.device.shader_modules.iter().for_each(|&module| self.vulkan.device.loader.destroy_shader_module(module, None));
            self.models.iter().for_each(|model|
            {
                self.vulkan.device.loader.destroy_buffer(model.vertex_buffer.handle, None);
                self.vulkan.device.loader.destroy_buffer(model.index_buffer.handle, None);
                self.vulkan.device.loader.free_memory(model.vertex_buffer.memory, None);
                self.vulkan.device.loader.free_memory(model.index_buffer.memory, None);
            });
            self.vulkan.device.uniform_buffers.iter().for_each(|buffer|
            {
                self.vulkan.device.loader.destroy_buffer(buffer.handle, None);
                self.vulkan.device.loader.free_memory(buffer.memory, None);
            });
            self.vulkan.device.loader.destroy_descriptor_pool(self.vulkan.device.descriptor_pool, None);
            self.vulkan.device.loader.free_command_buffers(self.vulkan.device.command_pool, &self.vulkan.device.command_buffers);
            self.vulkan.device.loader.destroy_command_pool(self.vulkan.device.command_pool, None);
            self.vulkan.device.loader.destroy_device(None);
            self.vulkan.surface.loader.destroy_surface(self.vulkan.surface.handle, None);
            self.vulkan.instance.destroy_instance(None);
        }
    }
}

unsafe fn find_memory_type(instance : &Instance, physical : vk::PhysicalDevice, requirements : vk::MemoryRequirements, properties : vk::MemoryPropertyFlags) -> u32
{
    let mem_properties = instance.get_physical_device_memory_properties(physical);
    mem_properties.memory_types.iter().enumerate().find(|(i, mem_type)|
    {
        requirements.memory_type_bits & (1 << i) != 0 && mem_type.property_flags.contains(properties)
    })
    .expect("no suitable memory for buffer found").0 as u32
}
