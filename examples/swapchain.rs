// Courtesy of Ralith

use erupt::{vk, DeviceLoader, EntryLoader, InstanceLoader};
use erupt_bootstrap::{
    DeviceBuilder, InstanceBuilder, QueueFamilyCriteria, Swapchain, SwapchainOptions,
};
use std::time::Instant;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

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
    epoch: Instant,

    swapchain: Swapchain,
    queue: vk::Queue,

    command_pool: vk::CommandPool,
    frames: Vec<Frame>,
}

impl App {
    pub fn new(window: &Window) -> Self {
        unsafe {
            let entry = EntryLoader::new().unwrap();
            let instance_builder = InstanceBuilder::new()
                .require_surface_extensions(&window)
                .unwrap();
            let (instance, _debug_messenger, instance_metadata) =
                instance_builder.build(&entry).unwrap();

            let surface = erupt::utils::surface::create_surface(&instance, &window, None).unwrap();

            let graphics_present = QueueFamilyCriteria::graphics_present();

            let device_builder = DeviceBuilder::new()
                .require_extension(vk::KHR_SWAPCHAIN_EXTENSION_NAME)
                .queue_family(graphics_present)
                .for_surface(surface);
            let (device, device_metadata) =
                device_builder.build(&instance, &instance_metadata).unwrap();
            let (graphics_present, graphics_present_idx) = device_metadata
                .device_queue(&instance, &device, graphics_present, 0)
                .unwrap()
                .unwrap();

            let size = window.inner_size();
            let mut options = SwapchainOptions::default();
            options.usage(vk::ImageUsageFlags::TRANSFER_DST); // Typically this would be left as the default, COLOR_ATTACHMENT
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

            Self {
                _entry: entry,
                instance,
                surface,
                epoch: Instant::now(),

                device,
                swapchain,
                queue: graphics_present,

                command_pool,
                frames,
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
                .acquire(&self.instance, &self.device, !0)
                .unwrap();
            let cmd = self.frames[acq.frame_index].cmd;
            let swapchain_image = self.swapchain.images()[acq.image_index];
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

            // Typically this barrier would be implemented with a subpass dependency from EXTERNAL,
            // with both pipeline stages set to COLOR_ATTACHMENT_OUTPUT so that it doesn't block
            // work that doesn't write to the swapchain image. The source stage must overlap with
            // the wait_dst_stage_mask passed to `queue_submit` below to ensure that the image
            // transition doesn't happen until after the acquire semaphore is signaled.
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::default(),
                &[],
                &[],
                &[vk::ImageMemoryBarrierBuilder::new()
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(swapchain_image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })],
            );
            let t = (self.epoch.elapsed().as_secs_f32().sin() + 1.0) * 0.5;
            self.device.cmd_clear_color_image(
                cmd,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [0.0, t, 0.0, 1.0],
                },
                &[vk::ImageSubresourceRangeBuilder::new()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)],
            );
            // Typically this barrier would be implemented with the implicit subpass dependency to
            // EXTERNAL
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::default(),
                &[],
                &[],
                &[vk::ImageMemoryBarrierBuilder::new()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .image(swapchain_image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })],
            );

            //
            // Submit commands and queue present
            //

            self.device.end_command_buffer(cmd).unwrap();
            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfoBuilder::new()
                        .wait_semaphores(&[acq.ready])
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
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
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct Frame {
    cmd: vk::CommandBuffer,
    complete: vk::Semaphore,
}
