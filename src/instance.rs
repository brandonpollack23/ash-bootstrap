//! Instance creation utils.
use std::{
    ffi::{c_void, CStr, CString, NulError},
    fmt,
    os::raw::c_char,
};
use ash::vk;
use cstr::cstr;
use smallvec::SmallVec;
use thiserror::Error;

/// Require, request or disable validation layers.
#[derive(Debug, Copy, Clone)]
pub enum ValidationLayers {
    /// Instance creation will fail if there are no validation layers installed.
    Require,
    /// If there are validation layers installed, enable them.
    Request,
    /// Don't enable validation layers.
    Disable,
}

/// Enable or disable the debug messenger, optionally providing a custom callback.
#[derive(Copy, Clone)]
pub enum DebugMessenger {
    /// Enables the debug messenger with the [`default_debug_callback`]
    /// callback.
    Default,
    /// Enables the debug messenger with a custom, user-provided callback.
    Custom {
        /// The user provided callback function. Feel free to take a look at the
        /// [`default_debug_callback`] when implementing your own.
        callback: vk::PFN_vkDebugUtilsMessengerCallbackEXT,
        /// A user data pointer passed to the debug callback.
        user_data_pointer: *mut c_void,
    },
    /// Disables the debug messenger.
    Disable,
}

/// The default debug callback used in [`DebugMessenger::Default`].
pub unsafe extern "system" fn default_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let message_severity = format!("{:?}", message_severity);
    let message_severity = message_severity.strip_suffix("_EXT").unwrap();

    let message_type = format!("{:?}", message_type);
    let message_type = message_type.strip_suffix("_EXT").unwrap();

    let message = CStr::from_ptr((*p_callback_data).p_message).to_string_lossy();
    // \x1b[1m{string}\x1b[0m - bold text.
    eprintln!("\x1b[1m{message_severity}\x1b[0m | \x1b[1m{message_type}\x1b[0m\n{message}");

    vk::FALSE
}

/// Metadata for after instance creation.
#[derive(Clone)]
pub struct InstanceMetadata {
    instance_handle: vk::Instance,
    api_version: u32,
    enabled_layers: SmallVec<CString>,
    enabled_extensions: SmallVec<CString>,
}

impl InstanceMetadata {
    /// The instance this metadata belongs to.
    #[inline]
    pub fn instance_handle(&self) -> vk::Instance {
        self.instance_handle
    }

    /// Retrieve the used instance API version.
    #[inline]
    pub fn api_version_raw(&self) -> u32 {
        self.api_version
    }

    /// Retrieve the used instance API major version.
    #[inline]
    pub fn api_version_major(&self) -> u32 {
        vk::api_version_major(self.api_version)
    }

    /// Retrieve the used instance API minor version.
    #[inline]
    pub fn api_version_minor(&self) -> u32 {
        vk::api_version_minor(self.api_version)
    }

    /// List of all enabled layers in the instance.
    #[inline]
    pub fn enabled_layers(&self) -> &[CString] {
        &self.enabled_layers
    }

    /// Returns true if `layer` is enabled.
    #[inline]
    pub unsafe fn is_layer_enabled(&self, layer: *const c_char) -> bool {
        let qry = CStr::from_ptr(layer);
        self.enabled_layers.iter().any(|e| e.as_c_str() == qry)
    }

    /// List of all enabled extensions in the instance.
    #[inline]
    pub fn enabled_extensions(&self) -> &[CString] {
        &self.enabled_extensions
    }

    /// Returns true if `extension` is enabled.
    #[inline]
    pub unsafe fn is_extension_enabled(&self, extension: *const c_char) -> bool {
        let qry = CStr::from_ptr(extension);
        self.enabled_extensions.iter().any(|i| i.as_c_str() == qry)
    }
}

impl fmt::Debug for InstanceMetadata {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("InstanceMetadata")
            .field(
                "api_version",
                &format_args!("{}.{}", self.api_version_major(), self.api_version_minor()),
            )
            .field("enabled_layers", &self.enabled_layers)
            .field("enabled_extensions", &self.enabled_extensions)
            .finish()
    }
}

/// Errors that can occur during instance creation.
#[derive(Debug, Error)]
pub enum InstanceCreationError {
    /// Vulkan Error.
    #[error("vulkan error")]
    VulkanError(#[from] vk::Result),
    /// One or more layers are not present.
    #[error("layers ({0:?}) not present")]
    LayersNotPresent(SmallVec<CString>),
    /// One or more extensions are not present.
    #[error("extensions ({0:?}) not present")]
    ExtensionsNotPresent(SmallVec<CString>),
    /// The instance loader creation failed.
    #[error("loader creation error")]
    LoaderCreation(#[from] LoaderError),
}

/// Allows to easily create an [`erupt::InstanceLoader`] and friends.
pub struct InstanceBuilder<'a> {
    loader_builder: InstanceLoaderBuilder<'a>,
    app_name: Option<CString>,
    app_version: Option<u32>,
    engine_name: Option<CString>,
    engine_version: Option<u32>,
    required_api_version: u32,
    requested_api_version: Option<u32>,
    layers: SmallVec<(*const c_char, bool)>,
    extensions: SmallVec<(*const c_char, bool)>,
    debug_messenger: DebugMessenger,
    debug_message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    debug_message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    enabled_validation_features: SmallVec<vk::ValidationFeatureEnableEXT>,
    disabled_validation_features: SmallVec<vk::ValidationFeatureDisableEXT>,
    allocator: Option<vk::AllocationCallbacks>,
}

impl<'a> InstanceBuilder<'a> {
    /// Create a new instance builder with opinionated defaults.
    #[inline]
    pub fn new() -> Self {
        InstanceBuilder::with_loader_builder(InstanceLoaderBuilder::new())
    }

    /// Create a new instance builder with a custom
    /// [`erupt::InstanceLoaderBuilder`] and opinionated defaults.
    #[inline]
    pub fn with_loader_builder(loader_builder: InstanceLoaderBuilder<'a>) -> Self {
        InstanceBuilder {
            loader_builder,
            app_name: None,
            app_version: None,
            engine_name: None,
            engine_version: None,
            required_api_version: vk::API_VERSION_1_0,
            requested_api_version: None,
            layers: SmallVec::new(),
            extensions: SmallVec::new(),
            debug_messenger: DebugMessenger::Disable,
            debug_message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT,
            debug_message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL_EXT
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION_EXT
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE_EXT,
            enabled_validation_features: SmallVec::new(),
            disabled_validation_features: SmallVec::new(),
            allocator: None,
        }
    }

    /// Application name to advertise.
    #[inline]
    pub fn app_name(mut self, app_name: &str) -> Result<Self, NulError> {
        self.app_name = Some(CString::new(app_name)?);
        Ok(self)
    }

    /// Application version to advertise.
    #[inline]
    pub fn app_version(mut self, major: u32, minor: u32) -> Self {
        self.app_version = Some(vk::make_api_version(0, major, minor, 0));
        self
    }

    /// Application version to advertise.
    #[inline]
    pub fn app_version_raw(mut self, app_version: u32) -> Self {
        self.app_version = Some(app_version);
        self
    }

    /// Engine name to advertise.
    #[inline]
    pub fn engine_name(mut self, engine_name: &str) -> Result<Self, NulError> {
        self.engine_name = Some(CString::new(engine_name)?);
        Ok(self)
    }

    /// Engine version to advertise.
    #[inline]
    pub fn engine_version(mut self, major: u32, minor: u32) -> Self {
        self.engine_version = Some(vk::make_api_version(0, major, minor, 0));
        self
    }

    /// Engine version to advertise.
    #[inline]
    pub fn engine_version_raw(mut self, engine_version: u32) -> Self {
        self.engine_version = Some(engine_version);
        self
    }

    /// Instance API version to be used as minimum requirement.
    #[inline]
    pub fn require_api_version(mut self, major: u32, minor: u32) -> Self {
        self.required_api_version = vk::make_api_version(0, major, minor, 0);
        self
    }

    /// Instance API version to be used as minimum requirement.
    #[inline]
    pub fn require_api_version_raw(mut self, api_version: u32) -> Self {
        self.required_api_version = api_version;
        self
    }

    /// Instance API version to request. If it is not supported, fall back to
    /// the highest supported version.
    #[inline]
    pub fn request_api_version(mut self, major: u32, minor: u32) -> Self {
        self.requested_api_version = Some(vk::make_api_version(0, major, minor, 0));
        self
    }

    /// Instance API version to request. If it is not supported, fall back to
    /// the highest supported version.
    #[inline]
    pub fn request_api_version_raw(mut self, api_version: u32) -> Self {
        self.requested_api_version = Some(api_version);
        self
    }

    /// Try to enable this layer, ignore if it's not supported
    #[inline]
    pub fn request_layer(mut self, layer: *const c_char) -> Self {
        self.layers.push((layer, false));
        self
    }

    /// Enable this layer, fail if it's not supported.
    #[inline]
    pub fn require_layer(mut self, layer: *const c_char) -> Self {
        self.layers.push((layer, true));
        self
    }

    /// Try to enable this extension, ignore if it is not supported.
    #[inline]
    pub fn request_extension(mut self, extension: *const c_char) -> Self {
        self.extensions.push((extension, false));
        self
    }

    /// Enable this extension, fail if it's not supported.
    #[inline]
    pub fn require_extension(mut self, extension: *const c_char) -> Self {
        self.extensions.push((extension, true));
        self
    }

    #[cfg(feature = "surface")]
    /// Adds an requirement on all Vulkan extensions necessary to create a
    /// surface on `window_handle`. You can also manually add these extensions.
    /// Returns `None` if the corresponding Vulkan surface extensions couldn't
    /// be found. This is only supported on feature `surface`.
    #[inline]
    pub fn require_surface_extensions(
        mut self,
        window_handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Option<Self> {
        let required_extensions =
            erupt::utils::surface::enumerate_required_extensions(window_handle).ok()?;
        self.extensions
            .extend(required_extensions.into_iter().map(|name| (name, true)));
        Some(self)
    }

    /// Add Khronos validation layers.
    #[inline]
    pub fn validation_layers(mut self, validation_layers: ValidationLayers) -> Self {
        match validation_layers {
            ValidationLayers::Require | ValidationLayers::Request => {
                self.layers.push((
                    cstr!("VK_LAYER_KHRONOS_validation"),
                    matches!(validation_layers, ValidationLayers::Require),
                ));

                self.extensions
                    .push((vk::EXT_VALIDATION_FEATURES_EXTENSION_NAME, false));
            }
            ValidationLayers::Disable => (),
        }

        self
    }

    /// Try to create a debug messenger with the config provided by
    /// `debug_messenger`.
    #[inline]
    pub fn request_debug_messenger(mut self, debug_messenger: DebugMessenger) -> Self {
        if !matches!(debug_messenger, DebugMessenger::Disable) {
            self.extensions
                .push((vk::EXT_DEBUG_UTILS_EXTENSION_NAME, false));
        }

        self.debug_messenger = debug_messenger;
        self
    }

    /// Filter for the severity of debug messages.
    #[inline]
    pub fn debug_message_severity(
        mut self,
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ) -> Self {
        self.debug_message_severity = severity;
        self
    }

    /// Filter for the type of debug messages.
    #[inline]
    pub fn debug_message_type(mut self, ty: vk::DebugUtilsMessageTypeFlagsEXT) -> Self {
        self.debug_message_type = ty;
        self
    }

    /// Enable an additional feature in the validation layers.
    #[inline]
    pub fn enable_validation_feature(
        mut self,
        validation_feature: vk::ValidationFeatureEnableEXT,
    ) -> Self {
        self.enabled_validation_features.push(validation_feature);
        self
    }

    /// Disable an feature in the validation layers.
    #[inline]
    pub fn disable_validation_feature(
        mut self,
        validation_feature: vk::ValidationFeatureDisableEXT,
    ) -> Self {
        self.disabled_validation_features.push(validation_feature);
        self
    }

    /// Allocation callback to use for internal Vulkan calls in the builder.
    #[inline]
    pub fn allocation_callbacks(mut self, allocator: vk::AllocationCallbacks) -> Self {
        self.allocator = Some(allocator);
        self
    }

    /// Returns the [`erupt::InstanceLoader`], an debug messenger if it was
    /// requested and successfully created, and [`InstanceMetadata`] about what
    /// is actually enabled in the instance.
    pub unsafe fn build<T>(
        self,
        entry: &'a CustomEntryLoader<T>,
    ) -> Result<
        (
            InstanceLoader,
            Option<vk::DebugUtilsMessengerEXT>,
            InstanceMetadata,
        ),
        InstanceCreationError,
    > {
        let mut required_api_version = self.required_api_version;
        let instance_version = entry.instance_version();
        if let Some(requested_api_version) = self.requested_api_version {
            required_api_version =
                required_api_version.max(requested_api_version.min(vk::make_api_version(
                    0,
                    vk::api_version_major(instance_version),
                    vk::api_version_minor(instance_version),
                    0,
                )));
        }

        let mut app_info = vk::ApplicationInfoBuilder::new().api_version(required_api_version);

        let app_name;
        if let Some(val) = self.app_name {
            app_name = val;
            app_info = app_info.application_name(&app_name);
        }

        if let Some(app_version) = self.app_version {
            app_info = app_info.application_version(app_version);
        }

        let engine_name;
        if let Some(val) = self.engine_name {
            engine_name = val;
            app_info = app_info.engine_name(&engine_name);
        }

        if let Some(engine_version) = self.engine_version {
            app_info = app_info.engine_version(engine_version);
        }

        let layer_properties = entry.enumerate_instance_layer_properties(None).result()?;
        let mut enabled_layers = SmallVec::new();
        let mut layers_not_present = SmallVec::new();
        for (layer_name, required) in self.layers {
            let cstr = CStr::from_ptr(layer_name);
            let present = layer_properties
                .iter()
                .any(|supported_layer| CStr::from_ptr(supported_layer.layer_name.as_ptr()) == cstr);

            match (required, present) {
                (_, true) => enabled_layers.push(layer_name),
                (true, false) => layers_not_present.push(cstr.to_owned()),
                (false, false) => (),
            }
        }

        if !layers_not_present.is_empty() {
            return Err(InstanceCreationError::LayersNotPresent(layers_not_present));
        }

        let mut extension_properties = entry
            .enumerate_instance_extension_properties(None, None)
            .result()?;
        for &layer_name in &enabled_layers {
            extension_properties.extend({
                let layer_name = CStr::from_ptr(layer_name);
                entry
                    .enumerate_instance_extension_properties(Some(layer_name), None)
                    .result()?
                    .into_iter()
            });
        }

        let mut enabled_extensions = SmallVec::new();
        let mut extensions_not_present = SmallVec::new();
        let debug_utils_cstr = CStr::from_ptr(vk::EXT_DEBUG_UTILS_EXTENSION_NAME);
        let mut is_debug_utils_enabled = false;
        let validation_features_cstr = CStr::from_ptr(vk::EXT_VALIDATION_FEATURES_EXTENSION_NAME);
        let mut is_validation_features_enabled = false;
        for (extension_name, required) in self.extensions {
            let cstr = CStr::from_ptr(extension_name);
            let present = extension_properties.iter().any(|supported_extension| {
                CStr::from_ptr(supported_extension.extension_name.as_ptr()) == cstr
            });

            match (required, present) {
                (_, true) => {
                    is_debug_utils_enabled |= cstr == debug_utils_cstr;
                    is_validation_features_enabled |= cstr == validation_features_cstr;
                    enabled_extensions.push(extension_name);
                }
                (true, false) => extensions_not_present.push(cstr.to_owned()),
                (false, false) => (),
            }
        }

        if !extensions_not_present.is_empty() {
            return Err(InstanceCreationError::ExtensionsNotPresent(
                extensions_not_present,
            ));
        }

        let mut instance_info = vk::InstanceCreateInfoBuilder::new()
            .application_info(&app_info)
            .enabled_layer_names(&enabled_layers)
            .enabled_extension_names(&enabled_extensions);

        let should_create_debug_messenger = !matches!(
            (&self.debug_messenger, is_debug_utils_enabled),
            (DebugMessenger::Disable, _) | (_, false)
        );

        let messenger_info = should_create_debug_messenger.then(|| {
            let messenger_info = vk::DebugUtilsMessengerCreateInfoEXTBuilder::new()
                .message_severity(self.debug_message_severity)
                .message_type(self.debug_message_type);
            match self.debug_messenger {
                DebugMessenger::Default => {
                    messenger_info.pfn_user_callback(Some(default_debug_callback))
                }
                DebugMessenger::Custom {
                    callback,
                    user_data_pointer,
                } => messenger_info
                    .pfn_user_callback(Some(callback))
                    .user_data(user_data_pointer),
                DebugMessenger::Disable => unreachable!(),
            }
        });

        let mut instance_messenger_info;
        if let Some(messenger_info) = messenger_info {
            instance_messenger_info = *messenger_info;
            instance_info = instance_info.extend_from(&mut instance_messenger_info);
        }

        let mut validation_features;
        if is_validation_features_enabled {
            validation_features = vk::ValidationFeaturesEXTBuilder::new()
                .enabled_validation_features(&self.enabled_validation_features)
                .disabled_validation_features(&self.disabled_validation_features);
            instance_info = instance_info.extend_from(&mut validation_features);
        }

        let instance = self.loader_builder.build(entry, &instance_info)?;
        let debug_utils_messenger = messenger_info
            .map(|messenger_info| unsafe {
                instance
                    .create_debug_utils_messenger_ext(&messenger_info, self.allocator.as_ref())
                    .result()
            })
            .transpose()?;
        let instance_metadata = InstanceMetadata {
            instance_handle: instance.handle,
            api_version: app_info.api_version,
            enabled_layers: enabled_layers
                .into_iter()
                .map(|ptr| unsafe { CStr::from_ptr(ptr).to_owned() })
                .collect(),
            enabled_extensions: enabled_extensions
                .into_iter()
                .map(|ptr| unsafe { CStr::from_ptr(ptr).to_owned() })
                .collect(),
        };

        Ok((instance, debug_utils_messenger, instance_metadata))
    }
}

impl<'a> Default for InstanceBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use erupt::EntryLoader;

    #[test]
    fn basic() {
        let entry = EntryLoader::new().unwrap();
        let (instance, _debug_messenger, _metadata) =
            unsafe { InstanceBuilder::new().build(&entry).unwrap() };

        unsafe {
            instance.destroy_instance(None);
        }
    }

    #[test]
    fn validation_and_messenger() {
        let entry = EntryLoader::new().unwrap();
        let (instance, debug_messenger, _metadata) = unsafe {
            InstanceBuilder::new()
                .validation_layers(ValidationLayers::Request)
                .request_debug_messenger(DebugMessenger::Default)
                .build(&entry)
                .unwrap()
        };

        unsafe {
            if let Some(debug_messenger) = debug_messenger {
                instance.destroy_debug_utils_messenger_ext(debug_messenger, None);
            }

            instance.destroy_instance(None);
        }
    }
}
