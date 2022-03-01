// Based off the Swapchain example of Ralith.

use erupt::{cstr, vk, DeviceLoader, EntryLoader, ExtendableFrom, InstanceLoader};
use erupt_bootstrap::{
    DeviceBuilder, InstanceBuilder, QueueFamilyCriteria, Swapchain, SwapchainOptions,
};
use std::{ffi::CStr, slice, time::Instant};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const FORMAT_CANDIDATES: &[vk::Format] = &[
    vk::Format::R8G8B8A8_UNORM,
    vk::Format::B8G8R8A8_UNORM,
    vk::Format::A8B8G8R8_UNORM_PACK32,
];
const SUBRESOURCE_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

unsafe fn shader_module(device: &DeviceLoader, bytes: &[u8]) -> vk::ShaderModule {
    let code = erupt::utils::decode_spv(bytes).unwrap();
    let module_info = vk::ShaderModuleCreateInfoBuilder::new().code(&code);
    device.create_shader_module(&module_info, None).unwrap()
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

struct TrianglePass {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl TrianglePass {
    unsafe fn new(device: &DeviceLoader, surface_format: vk::SurfaceFormatKHR) -> Self {
        // Pipeline creation
        let vs_module = shader_module(device, include_bytes!("../shaders/compiled/vert.spv"));
        let fs_module = shader_module(device, include_bytes!("../shaders/compiled/frag.spv"));

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

        let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfoBuilder::new()
            .color_attachment_formats(slice::from_ref(&surface_format.format));

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

        TrianglePass {
            pipeline,
            pipeline_layout,
        }
    }

    unsafe fn draw(&self, device: &DeviceLoader, cmd: vk::CommandBuffer) {
        device.cmd_draw(cmd, 3, 1, 0, 0);
    }

    unsafe fn destroy(&self, device: &DeviceLoader) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
    }
}

pub struct App {
    device: DeviceLoader,
    instance: InstanceLoader,
    _entry: EntryLoader,
    surface: vk::SurfaceKHR,
    _device_metadata: erupt_bootstrap::DeviceMetadata,
    epoch: Instant,
    swapchain: Swapchain,
    swapchain_image_views: Vec<vk::ImageView>,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    frames: Vec<Frame>,
    triangle_pass: TrianglePass,
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

            let mut vk1_3features = vk::PhysicalDeviceVulkan13FeaturesBuilder::new()
                .dynamic_rendering(true)
                .synchronization2(true);
            let features =
                vk::PhysicalDeviceFeatures2Builder::new().extend_from(&mut vk1_3features);

            let device_builder = DeviceBuilder::new()
                .require_version(1, 3)
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

            // Notice: Technically, there is no guarantee that the return value
            // of this function (vkGetPhysicalDeviceSurfaceFormatsKHR) doesn't
            // change, however, we are assuming this here. The `Swapchain`
            // helper is aimed at being as correct as possible, which is why its
            // internal selection code will reselect the surface format on every
            // recreate. This however means that our code would need to support
            // changing surface formats on the fly, which also means we'd need
            // to recreate all resources that depend on it (Pipelines that draw
            // onto the swapchain, etc). However, for this example, it's okay to
            // restrict the application to a single surface format we can rely
            // on. The application will crash if a different format is returned.
            let surface_formats = instance
                .get_physical_device_surface_formats_khr(
                    device_metadata.physical_device(),
                    surface,
                    None,
                )
                .unwrap();
            let surface_format = match *surface_formats.as_slice() {
                [single] if single.format == vk::Format::UNDEFINED => vk::SurfaceFormatKHR {
                    format: vk::Format::B8G8R8A8_UNORM,
                    color_space: single.color_space,
                },
                _ => *surface_formats
                    .iter()
                    .find(|surface_format| FORMAT_CANDIDATES.contains(&surface_format.format))
                    .unwrap_or(&surface_formats[0]),
            };

            let mut swapchain_options = SwapchainOptions::default();
            swapchain_options.format_preference(&[surface_format]);

            let size = window.inner_size();
            let swapchain = Swapchain::new(
                swapchain_options,
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
            let triangle_pass = TrianglePass::new(&device, surface_format);

            App {
                device,
                instance,
                _entry: entry,
                surface,
                _device_metadata: device_metadata,
                epoch: Instant::now(),
                swapchain,
                swapchain_image_views: Vec::new(),
                queue: graphics_present,
                command_pool,
                frames,
                triangle_pass,
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

            // Recreate swapchain image views when necessary
            if acq.invalidate_images {
                for &image_view in &self.swapchain_image_views {
                    self.device.destroy_image_view(image_view, None);
                }

                let format = self.swapchain.format();
                self.swapchain_image_views = self
                    .swapchain
                    .images()
                    .iter()
                    .map(|&swapchain_image| {
                        let image_view_info = vk::ImageViewCreateInfoBuilder::new()
                            .image(swapchain_image)
                            .view_type(vk::ImageViewType::_2D)
                            .subresource_range(SUBRESOURCE_RANGE)
                            .format(format.format);

                        self.device
                            .create_image_view(&image_view_info, None)
                            .unwrap()
                    })
                    .collect();
            }

            let in_flight = &self.frames[acq.frame_index];
            let swapchain_image = self.swapchain.images()[acq.image_index];
            let swapchain_image_view = self.swapchain_image_views[acq.image_index];

            let extend = self.swapchain.extent();
            let rect = vk::Rect2DBuilder::new().extent(extend);

            self.device
                .begin_command_buffer(
                    in_flight.cmd,
                    &vk::CommandBufferBeginInfoBuilder::new()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();

            // Record commands to render to swapchain_image

            self.device.cmd_bind_pipeline(
                in_flight.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.triangle_pass.pipeline,
            );

            self.device.cmd_set_scissor(in_flight.cmd, 0, &[rect]);
            let viewports = vk::ViewportBuilder::new()
                .height(extend.height as f32)
                .width(extend.width as f32)
                .max_depth(1.0);
            self.device.cmd_set_viewport(in_flight.cmd, 0, &[viewports]);

            let t = (self.epoch.elapsed().as_secs_f32().sin() + 1.0) * 0.5;
            let color_attachment = vk::RenderingAttachmentInfoBuilder::new()
                .image_view(swapchain_image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, t, 0.0, 1.0],
                    },
                })
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

            let rendering_info = vk::RenderingInfoBuilder::new()
                .color_attachments(slice::from_ref(&color_attachment))
                .layer_count(1)
                .render_area(vk::Rect2D {
                    offset: Default::default(),
                    extent: self.swapchain.extent(),
                });

            // Transition the swapchain image layout from UNDEFINED to
            // COLOR_ATTACHMENT_OPTIMAL before rendering. All
            // COLOR_ATTACHMENT_WRITEs in the COLOR_ATTACHMENT_OUTPUT stage must
            // wait for this transition to be complete.
            self.device.cmd_pipeline_barrier2(
                in_flight.cmd,
                &vk::DependencyInfoBuilder::new().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2Builder::new()
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(swapchain_image)
                        .subresource_range(SUBRESOURCE_RANGE),
                ]),
            );

            self.device
                .cmd_begin_rendering(in_flight.cmd, &rendering_info);

            self.triangle_pass.draw(&self.device, in_flight.cmd);

            self.device.cmd_end_rendering(in_flight.cmd);

            // Transition the swapchain image layout from
            // COLOR_ATTACHMENT_OPTIMAL to PRESENT_SRC_KHR in order to present
            // it to the screen. Any COLOR_ATTACHMENT access in the
            // COLOR_ATTACHMENT_OUTPUT_KHR stage, in which the
            // `in_flight.complete` semaphore is signalled, must wait for all
            // COLOR_ATTACHMENT_WRITEs in the past COLOR_ATTACHMENT_OUTPUT
            // operations to be completed.
            self.device.cmd_pipeline_barrier2(
                in_flight.cmd,
                &vk::DependencyInfoKHRBuilder::new().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2KHRBuilder::new()
                        .src_stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT_KHR)
                        .src_access_mask(vk::AccessFlags2KHR::COLOR_ATTACHMENT_WRITE_KHR)
                        .dst_stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT_KHR)
                        .dst_access_mask(
                            vk::AccessFlags2KHR::COLOR_ATTACHMENT_READ_KHR
                                | vk::AccessFlags2KHR::COLOR_ATTACHMENT_WRITE_KHR,
                        )
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(swapchain_image)
                        .subresource_range(SUBRESOURCE_RANGE),
                ]),
            );

            // Submit commands and queue present
            self.device.end_command_buffer(in_flight.cmd).unwrap();

            self.device
                .queue_submit2(
                    self.queue,
                    &[vk::SubmitInfo2Builder::new()
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfoBuilder::new()
                            .semaphore(acq.ready)
                            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfoBuilder::new()
                            .semaphore(in_flight.complete)
                            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)])
                        .command_buffer_infos(&[
                            vk::CommandBufferSubmitInfoBuilder::new().command_buffer(in_flight.cmd)
                        ])],
                    acq.complete,
                )
                .unwrap();
            self.swapchain
                .queue_present(
                    &self.device,
                    self.queue,
                    in_flight.complete,
                    acq.image_index,
                )
                .unwrap();
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for &image_view in &self.swapchain_image_views {
                self.device.destroy_image_view(image_view, None);
            }

            for frame in &self.frames {
                self.device.destroy_semaphore(frame.complete, None);
            }

            self.triangle_pass.destroy(&self.device);
            self.device.destroy_command_pool(self.command_pool, None);
            self.swapchain.destroy(&self.device);
            self.instance.destroy_surface_khr(self.surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct Frame {
    cmd: vk::CommandBuffer,
    complete: vk::Semaphore,
}
