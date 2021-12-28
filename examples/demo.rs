use erupt::{vk, EntryLoader};
use erupt_bootstrap::{
    DebugMessenger, DeviceBuilder, InstanceBuilder, QueueFamilyRequirements, ValidationLayers,
};
use winit::{
    event::{Event, KeyboardInput, StartCause, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("erupt-bootstrap")
        .build(&event_loop)
        .unwrap();

    let entry = EntryLoader::new().unwrap();
    let instance_builder = InstanceBuilder::new()
        .validation_layers(ValidationLayers::Request)
        .request_debug_messenger(DebugMessenger::Default)
        .require_surface_extensions(&window)
        .unwrap();
    let (instance, debug_messenger, instance_metadata) =
        unsafe { instance_builder.build(&entry) }.unwrap();

    let surface =
        unsafe { erupt::utils::surface::create_surface(&instance, &window, None) }.unwrap();

    let graphics_present = QueueFamilyRequirements::graphics_present();
    let device_features = vk::PhysicalDeviceFeatures2Builder::new().features(
        vk::PhysicalDeviceFeaturesBuilder::new()
            .alpha_to_one(true)
            .build(),
    );

    let device_builder = DeviceBuilder::new()
        .require_queue_family(graphics_present)
        .require_features(&device_features)
        .for_surface(surface);
    let (device, device_metadata) =
        unsafe { device_builder.build(&instance, &instance_metadata) }.unwrap();
    let graphics_present = device_metadata
        .device_queue(&instance, &device, graphics_present, 0)
        .unwrap()
        .unwrap();

    dbg!(device_metadata.device_name());
    dbg!(graphics_present);

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

                instance.destroy_surface_khr(surface, None);

                if let Some(debug_messenger) = debug_messenger {
                    instance.destroy_debug_utils_messenger_ext(debug_messenger, None);
                }

                instance.destroy_instance(None);
            }

            *control_flow = ControlFlow::Exit;
        }
        _ => (),
    });
}
