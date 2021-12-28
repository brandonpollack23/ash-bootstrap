use erupt::{vk, EntryLoader};
use erupt_bootstrap::{DebugMessenger, InstanceBuilder, ValidationLayers};
use std::ffi::CString;

fn main() {
    let entry = EntryLoader::new().unwrap();
    let instance_builder = InstanceBuilder::new()
        .validation_layers(ValidationLayers::Request)
        .request_debug_messenger(DebugMessenger::Default);
    let (instance, debug_messenger, metadata) = unsafe { instance_builder.build(&entry) }.unwrap();

    unsafe {
        let message = CString::new(format!("{:#?}", metadata)).unwrap();
        instance.submit_debug_utils_message_ext(
            vk::DebugUtilsMessageSeverityFlagBitsEXT::ERROR_EXT,
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION_EXT,
            &vk::DebugUtilsMessengerCallbackDataEXTBuilder::new().message(message.as_c_str()),
        );

        if let Some(debug_messenger) = debug_messenger {
            instance.destroy_debug_utils_messenger_ext(debug_messenger, None);
        }

        instance.destroy_instance(None);
    }
}
