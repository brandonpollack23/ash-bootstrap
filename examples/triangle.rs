// Based off the Swapchain example of Ralith.

use erupt::{cstr, vk, DeviceLoader, EntryLoader, ExtendableFrom, InstanceLoader};
use erupt_bootstrap::{
    DeviceBuilder, InstanceBuilder, QueueFamilyCriteria, Swapchain, SwapchainOptions,
};
use std::{ffi::CStr, slice};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

pub const SUBRESOURCE_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

pub trait DeviceLoaderUtils {
    unsafe fn shader_module(self: &Self, bytes: &[u8]) -> vk::ShaderModule;
}

impl DeviceLoaderUtils for DeviceLoader {
    unsafe fn shader_module(self: &Self, bytes: &[u8]) -> vk::ShaderModule {
        let code = erupt::utils::decode_spv(bytes).unwrap();
        let module_info = vk::ShaderModuleCreateInfoBuilder::new().code(&code);
        self.create_shader_module(&module_info, None).unwrap()
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    let mut app = App::new(&window);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                app.resize(vk::Extent2D {
                    width: size.width,
                    height: size.height,
                });
            }
            Event::MainEventsCleared => {
                app.draw();
            }
            _ => (),
        }
    });
}

pub struct App {
    device: DeviceLoader,
    instance: InstanceLoader,
    _entry: EntryLoader,
    surface: vk::SurfaceKHR,
    _device_metadata: erupt_bootstrap::DeviceMetadata,
    swapchain: Swapchain,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    frames: Vec<Frame>,
    test_pass: TestPass,
}

struct TestPass {
    // Cleaned up by App.
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl TestPass {
    fn new(
        device: &DeviceLoader,
        instance: &InstanceLoader,
        device_metadata: &erupt_bootstrap::DeviceMetadata,
        surface: &vk::SurfaceKHR,
    ) -> Self {
        // --- Pipeline creation ---
        unsafe {
            let vs_module = device.shader_module(include_bytes!("../shaders/compiled/vert.spv"));
            let fs_module = device.shader_module(include_bytes!("../shaders/compiled/frag.spv"));

            let main_cstr = CStr::from_ptr(cstr!("main"));
            let shader_stages = [
                vk::PipelineShaderStageCreateInfoBuilder::new()
                    .stage(vk::ShaderStageFlagBits::VERTEX)
                    .module(vs_module)
                    .name(main_cstr),
                vk::PipelineShaderStageCreateInfoBuilder::new()
                    .stage(vk::ShaderStageFlagBits::FRAGMENT)
                    .module(fs_module)
                    .name(main_cstr),
            ];

            let formats = instance
                .get_physical_device_surface_formats_khr(
                    device_metadata.physical_device(),
                    *surface,
                    None,
                )
                .unwrap();

            // Not going to lie, no idea how to properly setup a pipeline with erupt_bootstrap...

            let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfoBuilder::new()
                .color_attachment_formats(&slice::from_ref(&formats[0].format));

            let pipeline_layout_info = vk::PipelineLayoutCreateInfoBuilder::new();
            let pipeline_layout = device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap();

            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

            let dynamic_pipeline_state = vk::PipelineDynamicStateCreateInfoBuilder::new()
                .dynamic_states(&[vk::DynamicState::SCISSOR, vk::DynamicState::VIEWPORT]);

            let viewport_state = vk::PipelineViewportStateCreateInfoBuilder::new()
                .scissor_count(1)
                .viewport_count(1);
            let rasterization_state =
                vk::PipelineRasterizationStateCreateInfoBuilder::new().line_width(1.0);
            let multisample_state = vk::PipelineMultisampleStateCreateInfoBuilder::new()
                .rasterization_samples(vk::SampleCountFlagBits::_1);

            let color_blend_attachments = vec![vk::PipelineColorBlendAttachmentStateBuilder::new()
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
                .blend_enable(false)];

            let color_blending_info = vk::PipelineColorBlendStateCreateInfoBuilder::new()
                .logic_op_enable(false)
                .attachments(&color_blend_attachments);
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfoBuilder::new();
            let pipeline_infos = &[vk::GraphicsPipelineCreateInfoBuilder::new()
                .vertex_input_state(&vertex_input_state)
                .color_blend_state(&color_blending_info)
                .multisample_state(&multisample_state)
                .stages(&shader_stages)
                .layout(pipeline_layout)
                .rasterization_state(&rasterization_state)
                .dynamic_state(&dynamic_pipeline_state)
                .viewport_state(&viewport_state)
                .input_assembly_state(&input_assembly_state)
                .extend_from(&mut pipeline_rendering_info)];

            let pipeline = device
                .create_graphics_pipelines(vk::PipelineCache::null(), pipeline_infos, None)
                .expect("Failed to create pipeline")[0];
            device.destroy_shader_module(fs_module, None);
            device.destroy_shader_module(vs_module, None);

            Self {
                pipeline,
                pipeline_layout,
            }
        }
    }
}

impl App {
    pub fn new(window: &Window) -> Self {
        unsafe {
            let entry = EntryLoader::new().unwrap();
            let instance_builder = InstanceBuilder::new()
                .require_api_version(1, 3)
                .require_surface_extensions(&window)
                .unwrap();
            let (instance, _debug_messenger, instance_metadata) =
                instance_builder.build(&entry).unwrap();

            let surface = erupt::utils::surface::create_surface(&instance, &window, None).unwrap();

            let graphics_present = QueueFamilyCriteria::graphics_present();

            let dynamic_rendering = &mut vk::PhysicalDeviceDynamicRenderingFeaturesBuilder::new()
                .dynamic_rendering(true);
            let synchronization2 = &mut vk::PhysicalDeviceSynchronization2FeaturesBuilder::new()
                .synchronization2(true);

            let features = vk::PhysicalDeviceFeatures2Builder::new()
                .extend_from(dynamic_rendering)
                .extend_from(synchronization2);

            let device_builder = DeviceBuilder::new()
                .require_extension(vk::KHR_SWAPCHAIN_EXTENSION_NAME)
                .queue_family(graphics_present)
                .for_surface(surface)
                .require_features(&features);

            let (device, device_metadata) =
                device_builder.build(&instance, &instance_metadata).unwrap();
            let (graphics_present, graphics_present_idx) = device_metadata
                .device_queue(&instance, &device, graphics_present, 0)
                .unwrap()
                .unwrap();

            let size = window.inner_size();
            let mut options = SwapchainOptions::default();

            let formats = instance
                .get_physical_device_surface_formats_khr(
                    device_metadata.physical_device(),
                    surface,
                    None,
                )
                .unwrap();

            options.format_preference(&formats);
            options.usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

            let swapchain = Swapchain::new(
                options,
                surface,
                device_metadata.physical_device(),
                &device,
                vk::Extent2D {
                    width: size.width,
                    height: size.height,
                },
            );

            let command_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfoBuilder::new()
                        .flags(
                            vk::CommandPoolCreateFlags::TRANSIENT
                                | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                        )
                        .queue_family_index(graphics_present_idx),
                    None,
                )
                .unwrap();
            let cmds = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfoBuilder::new()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(swapchain.frames_in_flight() as u32),
                )
                .unwrap();
            let frames = cmds
                .into_iter()
                .map(|cmd| Frame {
                    cmd,
                    complete: device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .unwrap(),
                })
                .collect();
            let test_pass = TestPass::new(&device, &instance, &device_metadata, &surface);

            Self {
                _entry: entry,
                instance,
                surface,
                device,
                swapchain,
                queue: graphics_present,

                command_pool,
                frames,
                _device_metadata: device_metadata,
                test_pass,
            }
        }
    }

    fn resize(&mut self, size: vk::Extent2D) {
        self.swapchain.update(size);
    }

    fn draw(&mut self) {
        unsafe {
            let acq = self
                .swapchain
                .acquire(&self.instance, &self.device, u64::MAX)
                .unwrap();
            let cmd = self.frames[acq.frame_index].cmd;
            let swapchain_image = self.swapchain.images()[acq.image_index];

            let extend = self.swapchain.extent();
            let rect = vk::Rect2DBuilder::new().extent(extend);

            self.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfoBuilder::new()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();

            //
            // Record commands to render to swapchain_image
            //

            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.test_pass.pipeline,
            );

            self.device.cmd_set_scissor(cmd, 0, &[rect]);
            let viewports = vk::ViewportBuilder::new()
                .height(extend.height as f32)
                .width(extend.width as f32)
                .max_depth(1.0);
            self.device.cmd_set_viewport(cmd, 0, &[viewports]);

            let format = self.swapchain.format();

            let image_view_info = vk::ImageViewCreateInfoBuilder::new()
                .image(swapchain_image)
                .view_type(vk::ImageViewType::_2D)
                .subresource_range(SUBRESOURCE_RANGE)
                .format(format.format);
            let image_view = self
                .device
                .create_image_view(&image_view_info, None)
                .unwrap();

            let color_attachment = vk::RenderingAttachmentInfoBuilder::new()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.1, 0.1, 0.1, 1.0],
                    },
                })
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

            let extend = self.swapchain.extent();
            let rect = vk::Rect2DBuilder::new().extent(extend);

            let rendering_info = vk::RenderingInfoBuilder::new()
                .color_attachments(slice::from_ref(&color_attachment))
                .layer_count(1)
                .render_area(rect.build());

            // Note: I only a vague idea how syncronization works.
            self.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfoBuilder::new().image_memory_barriers(&[
                    // Note: some of these settings can be removed for bravity (without having compaints from vaildation layers/visual errors).
                    // But I don't know if that's recommendable.
                    vk::ImageMemoryBarrier2Builder::new()
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_access_mask(
                            vk::AccessFlags2::COLOR_ATTACHMENT_READ
                                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                        )
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .image(swapchain_image)
                        .subresource_range(SUBRESOURCE_RANGE),
                ]),
            );
            self.device.cmd_begin_rendering(cmd, &rendering_info);

            self.device.cmd_draw(cmd, 3, 1, 0, 0);

            self.device.cmd_end_rendering(cmd);

            // Submit commands and queue present

            self.device.end_command_buffer(cmd).unwrap();

            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfoBuilder::new()
                        .wait_semaphores(&[acq.ready])
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                        .signal_semaphores(&[self.frames[acq.frame_index].complete])
                        .command_buffers(&[cmd])],
                    acq.complete,
                )
                .unwrap();
            self.swapchain
                .queue_present(
                    &self.device,
                    self.queue,
                    self.frames[acq.frame_index].complete,
                    acq.image_index,
                )
                .unwrap();
            self.device.destroy_image_view(image_view, None);
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for frame in &self.frames {
                self.device.destroy_semaphore(frame.complete, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.swapchain.destroy(&self.device);
            self.instance.destroy_surface_khr(self.surface, None);
            self.device.destroy_pipeline(self.test_pass.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.test_pass.pipeline_layout, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct Frame {
    cmd: vk::CommandBuffer,
    complete: vk::Semaphore,
}
