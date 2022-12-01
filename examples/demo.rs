use ash::extensions::khr::Surface;
use ash::{vk, Entry};
use ash_bootstrap::{
    DebugMessenger, DeviceBuilder, InstanceBuilder, QueueFamilyCriteria, ValidationLayers,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::event::Event;
use winit::{
    event::{KeyboardInput, StartCause, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("ash-bootstrap")
        .build(&event_loop)
        .unwrap();

    let entry = unsafe { Entry::load() }.unwrap();
    let instance_builder = InstanceBuilder::new()
        .validation_layers(ValidationLayers::Request)
        .request_debug_messenger(DebugMessenger::Default)
        .require_surface_extensions(&window)
        .unwrap();
    let (instance, (debug_loader, debug_messenger), instance_metadata) =
        unsafe { instance_builder.build(&entry) }.unwrap();

    let surface_loader = unsafe { Surface::new(&entry, &instance) };
    let surface = unsafe {
        ash_window::create_surface(
            &entry,
            &instance,
            window.raw_display_handle(),
            window.raw_window_handle(),
            None,
        )
        .expect("Cannot create surface")
    };

    let graphics_present = QueueFamilyCriteria::graphics_present();
    let transfer = QueueFamilyCriteria::preferably_separate_transfer();

    let device_features = vk::PhysicalDeviceFeatures2::builder()
        .features(vk::PhysicalDeviceFeatures::builder().build());

    let device_builder = DeviceBuilder::new()
        .queue_family(graphics_present)
        .queue_family(transfer)
        .require_features(&device_features)
        .for_surface(surface);
    let (device, device_metadata) =
        unsafe { device_builder.build(&instance, &surface_loader, &instance_metadata) }.unwrap();
    let graphics_present = device_metadata
        .device_queue(&surface_loader, &device, graphics_present, 0)
        .unwrap()
        .unwrap();
    let transfer = device_metadata
        .device_queue(&surface_loader, &device, transfer, 0)
        .unwrap()
        .unwrap();

    dbg!(device_metadata.device_name());
    dbg!(graphics_present);
    dbg!(transfer);

    event_loop.run(move |event, _, control_flow| match event {
        Event::NewEvents(StartCause::Init) => *control_flow = ControlFlow::Poll,
        Event::MainEventsCleared => {}
        Event::WindowEvent {
            event:
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                },
            ..
        } => {
            unsafe {
                device.destroy_device(None);

                surface_loader.destroy_surface(surface, None);

                if let Some(debug_messenger) = debug_messenger {
                    debug_loader.destroy_debug_utils_messenger(debug_messenger, None);
                }

                instance.destroy_instance(None);
            }

            *control_flow = ControlFlow::Exit;
        }
        _ => (),
    });
}
