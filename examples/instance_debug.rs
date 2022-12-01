use ash::{vk, Entry};
use ash_bootstrap::{DebugMessenger, InstanceBuilder, ValidationLayers};
use std::ffi::CString;

fn main() {
    let entry = unsafe { Entry::load() }.unwrap();
    let instance_builder = InstanceBuilder::new()
        .validation_layers(ValidationLayers::Request)
        .request_debug_messenger(DebugMessenger::Default);
    let (instance, (debug_loader, debug_messenger), metadata) =
        unsafe { instance_builder.build(&entry) }.unwrap();

    unsafe {
        let message = CString::new(format!("{:#?}", metadata)).unwrap();
        debug_loader.submit_debug_utils_message(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            &vk::DebugUtilsMessengerCallbackDataEXT::builder()
                .message(message.as_c_str())
                .build(),
        );

        if let Some(debug_messenger) = debug_messenger {
            debug_loader.destroy_debug_utils_messenger(debug_messenger, None);
        }

        instance.destroy_instance(None);
    }
}
