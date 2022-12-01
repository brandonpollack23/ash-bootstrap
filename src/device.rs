//! Device creation utils.
use crate::{BootstrapSmallVec, InstanceMetadata};
use ash::extensions::khr::Surface;
use ash::prelude::VkResult;
use ash::{vk, Device, Instance, LoadingError};
use std::{
    borrow::Cow,
    collections::HashSet,
    ffi::{CStr, CString},
    hash::{Hash, Hasher},
    os::raw::{c_char, c_float},
};
use thiserror::Error;

/// Criteria for queue families.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub struct QueueFamilyCriteria {
    /// A queue family will only be considered if all these flags are set.
    pub must_support: vk::QueueFlags,
    /// A queue family will be preferred over other ones
    /// if all these flags are set. The more of these flags are set,
    /// the more likely it is for the queue family to be chosen.
    pub should_support: vk::QueueFlags,
    /// A queue family will only be considered
    /// if none of these flags are set.
    pub must_not_support: vk::QueueFlags,
    /// A queue family will be preferred over other ones
    /// if none of these flags are set. The less of these flags are set,
    /// the more likely it is for the queue family to be chosen.
    pub should_not_support: vk::QueueFlags,
    /// This criteria is only met if the presentation support matches with
    /// this flag. `None` corresponds to being indifferent to the support.
    /// `Some(expected)` corresponds to the criteria being met if the support
    /// matches with `expected`.
    pub presentation_support: Option<bool>,
}

impl QueueFamilyCriteria {
    /// Queue family criteria that are always met.
    #[inline]
    pub fn none() -> QueueFamilyCriteria {
        QueueFamilyCriteria::default()
    }

    /// The criteria are only met if the queue family supports graphics and
    /// presentation.
    #[inline]
    pub fn graphics_present() -> QueueFamilyCriteria {
        QueueFamilyCriteria::none()
            .must_support(vk::QueueFlags::GRAPHICS)
            .must_support_presentation()
    }

    /// Tries to match the queue family that's the closest to being a pure
    /// transfer queue.
    #[inline]
    pub fn preferably_separate_transfer() -> QueueFamilyCriteria {
        QueueFamilyCriteria::none()
            .must_support(vk::QueueFlags::TRANSFER)
            .should_not_support(!vk::QueueFlags::TRANSFER)
    }

    /// Add an requirement that these queue flags must be present in the
    /// queue family.
    #[inline]
    pub fn must_support(mut self, must_support: vk::QueueFlags) -> QueueFamilyCriteria {
        self.must_support |= must_support;
        self
    }

    /// Add an recommendation that these queue flags should be present in the
    /// queue family.
    #[inline]
    pub fn should_support(mut self, should_support: vk::QueueFlags) -> QueueFamilyCriteria {
        self.should_support |= should_support;
        self
    }

    /// Add an requirement that these queue flags must **not** be present in the
    /// queue family.
    #[inline]
    pub fn must_not_support(mut self, must_not_support: vk::QueueFlags) -> QueueFamilyCriteria {
        self.must_not_support |= must_not_support;
        self
    }

    /// Add an recommendation that these queue flags should **not** be present in the
    /// queue family.
    #[inline]
    pub fn should_not_support(mut self, should_not_support: vk::QueueFlags) -> QueueFamilyCriteria {
        self.should_not_support |= should_not_support;
        self
    }

    /// Require that the queue family must support presentation.
    #[inline]
    pub fn must_support_presentation(mut self) -> QueueFamilyCriteria {
        self.presentation_support = Some(true);
        self
    }

    /// Require that the queue family must not support presentation.
    #[inline]
    pub fn must_not_support_presentation(mut self) -> QueueFamilyCriteria {
        self.presentation_support = Some(false);
        self
    }

    /// Returns the index of the best suited queue family.
    /// Returns `Ok(None)` when no queue family meets the criteria.
    /// Returns `Err(_)` when an internal Vulkan call failed.
    pub fn choose_queue_family<'a>(
        &self,
        ash_surface: &Surface,
        physical_device: vk::PhysicalDevice,
        queue_family_properties: &'a [vk::QueueFamilyProperties],
        surface: Option<vk::SurfaceKHR>,
    ) -> Result<Option<(u32, &'a vk::QueueFamilyProperties)>, vk::Result> {
        let mut candidates = BootstrapSmallVec::new();
        for (i, queue_family_properties) in queue_family_properties.iter().enumerate() {
            let i = i as u32;

            let positive_required = queue_family_properties
                .queue_flags
                .contains(self.must_support);

            let negative_required = !queue_family_properties
                .queue_flags
                .intersects(self.must_not_support);

            let presentation = || {
                Ok(match (self.presentation_support, surface) {
                    (None, _) => true,
                    (Some(_), None) => false,
                    (Some(expected), Some(surface)) => unsafe {
                        let support = ash_surface.get_physical_device_surface_support(
                            physical_device,
                            i,
                            surface,
                        )?;

                        support == expected
                    },
                })
            };

            if positive_required && negative_required && presentation()? {
                candidates.push((i, queue_family_properties));
            }
        }

        let best_candidate = candidates
            .into_iter()
            .max_by_key(|(_, queue_family_properties)| {
                let positive_recommended = (self.should_support
                    & queue_family_properties.queue_flags)
                    .as_raw()
                    .count_ones();

                let negative_recommended = (self.should_not_support
                    & (!queue_family_properties.queue_flags))
                    .as_raw()
                    .count_ones();

                positive_recommended + negative_recommended
            });

        Ok(best_candidate)
    }
}

/// Errors that can occur during device creation.
#[derive(Debug, Error)]
pub enum DeviceCreationError {
    /// Vulkan Error.
    #[error("vulkan error")]
    VulkanError(#[from] vk::Result),
    /// There is no physical device at the index specified by [`DeviceBuilder::select_nth_unconditionally`].
    #[error("no physical device at specified index")]
    UnconditionalMissing,
    /// No physical device met the requirements.
    #[error("no physical device met the requirements")]
    RequirementsNotMet,
    /// The instance loader creation failed.
    #[error("loader creation error")]
    LoaderCreation(#[from] LoadingError),
}

/// Setup for [`vk::Queue`] creation. Used within [`CustomQueueSetupFn`].
/// The [`Hash`] and [`PartialEq`] implementations on this struct **only**
/// compare `queue_family_index`.
#[derive(Debug, Clone)]
pub struct QueueSetup {
    /// Flags used to specify usage behavior of the queue.
    pub flags: vk::DeviceQueueCreateFlags,
    /// Index of the queue family in the queue family array.
    pub queue_family_index: u32,
    /// Specifies the amount of queues and the respective priority for each.
    pub queue_priorities: Vec<c_float>,
}

impl QueueSetup {
    /// Create a new custom queue setup with simplified arguments.
    /// Queue priorities will all be 1.0 and all flags will be empty.
    #[inline]
    pub fn simple(queue_family_index: u32, queue_count: usize) -> QueueSetup {
        QueueSetup {
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index,
            queue_priorities: (0..queue_count).map(|_| 1.0).collect(),
        }
    }

    #[inline]
    fn as_vulkan(&self) -> vk::DeviceQueueCreateInfoBuilder {
        vk::DeviceQueueCreateInfo::builder()
            .flags(self.flags)
            .queue_family_index(self.queue_family_index)
            .queue_priorities(&self.queue_priorities)
    }
}

impl PartialEq for QueueSetup {
    fn eq(&self, rhs: &Self) -> bool {
        self.queue_family_index == rhs.queue_family_index
    }
}

impl Eq for QueueSetup {}

impl Hash for QueueSetup {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.queue_family_index.hash(state);
    }
}

/// Metadata for after device creation.
#[derive(Debug, Clone)]
pub struct DeviceMetadata {
    device_handle: vk::Device,
    physical_device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    queue_setups: BootstrapSmallVec<QueueSetup>,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    queue_family_properties: Vec<vk::QueueFamilyProperties>,
    surface: Option<vk::SurfaceKHR>,
    enabled_extensions: BootstrapSmallVec<CString>,
}

impl DeviceMetadata {
    /// The device this metadata belongs to.
    #[inline]
    pub fn device_handle(&self) -> vk::Device {
        self.device_handle
    }

    /// The physical device this device belongs to.
    #[inline]
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// The surface this device was created for.
    #[inline]
    pub fn surface(&self) -> Option<vk::SurfaceKHR> {
        self.surface
    }

    /// Properties of the physical device.
    #[inline]
    pub fn properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.properties
    }

    /// Name of the physical device.
    #[inline]
    pub fn device_name(&self) -> Cow<str> {
        unsafe { CStr::from_ptr(self.properties.device_name.as_ptr()).to_string_lossy() }
    }

    /// Type of the physical device.
    #[inline]
    pub fn device_type(&self) -> vk::PhysicalDeviceType {
        self.properties.device_type
    }

    /// Returns a queue and the index of the queue family it belongs to.
    /// The best suited queue family meeting the criteria will be chosen.
    /// `queue_index` is the index within the queue family.
    #[inline]
    pub fn device_queue(
        &self,
        surface_loader: &Surface,
        device: &Device,
        criteria: QueueFamilyCriteria,
        queue_index: u32,
    ) -> Result<Option<(vk::Queue, u32)>, vk::Result> {
        let queue_family = criteria.choose_queue_family(
            surface_loader,
            self.physical_device,
            &self.queue_family_properties,
            self.surface,
        )?;

        Ok(queue_family.and_then(|(idx, _properties)| unsafe {
            let handle = device.get_device_queue(idx, queue_index);
            (handle != vk::Queue::null()).then_some((handle, idx))
        }))
    }

    /// The queue setups which are in use.
    #[inline]
    pub fn queue_setups(&self) -> &[QueueSetup] {
        &self.queue_setups
    }

    /// The memory properties of the physical device.
    #[inline]
    pub fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.memory_properties
    }

    /// The queue family properties of the physical device.
    #[inline]
    pub fn queue_family_properties(&self) -> &[vk::QueueFamilyProperties] {
        &self.queue_family_properties
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

/// Function used to specify custom [`QueueSetup`]s, specified by
/// [`DeviceBuilder::custom_queue_setup`].
pub type CustomQueueSetupFn = dyn FnMut(
    vk::PhysicalDevice,
    &[QueueFamilyCriteria],
    &[vk::QueueFamilyProperties],
) -> Result<Option<HashSet<QueueSetup>>, vk::Result>;

/// Suitability of a physical device.
pub enum DeviceSuitability {
    /// If all requirements meet this criteria, the physical device gets picked
    /// and the search is concluded.
    Perfect,
    /// If any requirement meets this criteria, the physical device gets
    /// considered but the search for a potentially perfect physical device
    /// continues.
    NotPreferred,
    /// If any requirement meets this criteria, the physical device
    /// will under no circumstances be considered.
    Unsuitable,
}

impl From<bool> for DeviceSuitability {
    fn from(suitable_perfect: bool) -> Self {
        if suitable_perfect {
            DeviceSuitability::Perfect
        } else {
            DeviceSuitability::Unsuitable
        }
    }
}

/// Function used to specify a custom additional [`DeviceSuitability`]
/// to consider in the selection process.
pub type AdditionalSuitabilityFn = dyn FnMut(&Instance, vk::PhysicalDevice) -> DeviceSuitability;

/// Builder for an device loader.
pub struct DeviceLoaderBuilder<'a> {
    create_device_fn: Option<
        &'a mut dyn FnMut(
            vk::PhysicalDevice,
            &vk::DeviceCreateInfo,
            Option<&vk::AllocationCallbacks>,
        ) -> VkResult<Device>,
    >,
    symbol_fn:
        Option<&'a mut dyn FnMut(vk::Device, *const c_char) -> Option<vk::PFN_vkVoidFunction>>,
    allocation_callbacks: Option<&'a vk::AllocationCallbacks>,
}

impl<'a> DeviceLoaderBuilder<'a> {
    /// Create a new instance loader builder.
    pub fn new() -> Self {
        DeviceLoaderBuilder {
            create_device_fn: None,
            symbol_fn: None,
            allocation_callbacks: None,
        }
    }

    /// Specify a custom device creation function, to use in place of the
    /// default.
    ///
    /// This may be useful when creating the device using e.g. OpenXR.
    pub fn create_device_fn(
        mut self,
        create_device: &'a mut dyn FnMut(
            vk::PhysicalDevice,
            &vk::DeviceCreateInfo,
            Option<&vk::AllocationCallbacks>,
        ) -> VkResult<Device>,
    ) -> Self {
        self.create_device_fn = Some(create_device);
        self
    }

    /// Specify a custom symbol function, called to get device function
    /// pointers, to use in place of the default.
    pub fn symbol_fn(
        mut self,
        symbol: &'a mut impl FnMut(vk::Device, *const c_char) -> Option<vk::PFN_vkVoidFunction>,
    ) -> Self {
        self.symbol_fn = Some(symbol);
        self
    }

    /// Specify custom allocation callback functions.
    pub fn allocation_callbacks(mut self, allocator: &'a vk::AllocationCallbacks) -> Self {
        self.allocation_callbacks = Some(allocator);
        self
    }

    /// Build the device loader. Ensure `create_info` is the same as used in the
    /// creation of `device`.
    pub unsafe fn build_with_existing_device(
        self,
        instance_loader: &Instance,
        device: Device,
    ) -> VkResult<Device> {
        Ok(Device::load(instance_loader.fp_v1_0(), device.handle()))
    }

    /// Build the device. If you want to entirely create the device
    /// yourself, use [`DeviceBuilder::build_with_existing_device`].
    pub unsafe fn build(
        mut self,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        create_info: &vk::DeviceCreateInfo,
    ) -> VkResult<Device> {
        match &mut self.create_device_fn {
            Some(create_device) => {
                create_device(physical_device, create_info, self.allocation_callbacks)
            }
            None => instance.create_device(physical_device, create_info, self.allocation_callbacks),
        }
    }
}

/// Allows to easily create an [`erupt::Device`] and queues.
pub struct DeviceBuilder<'a> {
    loader_builder: DeviceLoaderBuilder<'a>,
    queue_setup_fn: Option<Box<CustomQueueSetupFn>>,
    additional_suitability_fn: Option<Box<AdditionalSuitabilityFn>>,
    surface: Option<vk::SurfaceKHR>,
    prioritised_device_types: BootstrapSmallVec<vk::PhysicalDeviceType>,
    queue_family_criteria: BootstrapSmallVec<QueueFamilyCriteria>,
    preferred_device_memory_size: Option<vk::DeviceSize>,
    required_device_memory_size: Option<vk::DeviceSize>,
    extensions: BootstrapSmallVec<(*const c_char, bool)>,
    preferred_version: Option<u32>,
    required_version: u32,
    required_features: Option<&'a vk::PhysicalDeviceFeatures2>,
    unconditional_nth: Option<usize>,
    allocator: Option<vk::AllocationCallbacks>,
}

impl<'a> DeviceBuilder<'a> {
    /// Create a new device builder.
    #[inline]
    pub fn new() -> Self {
        DeviceBuilder::with_loader_builder(DeviceLoaderBuilder::new())
    }

    /// Create a new device builder with a custom [`DeviceBuilder`].
    pub fn with_loader_builder(loader_builder: DeviceLoaderBuilder<'a>) -> Self {
        DeviceBuilder {
            loader_builder,
            queue_setup_fn: None,
            additional_suitability_fn: None,
            surface: None,
            prioritised_device_types: BootstrapSmallVec::new(),
            queue_family_criteria: BootstrapSmallVec::new(),
            preferred_device_memory_size: None,
            required_device_memory_size: None,
            extensions: BootstrapSmallVec::new(),
            preferred_version: None,
            required_version: vk::API_VERSION_1_0,
            required_features: None,
            unconditional_nth: None,
            allocator: None,
        }
    }

    /// Specify a custom queue setup.
    ///
    /// ### Default setup
    ///
    /// ```ignore
    /// |physical_device, queue_family_criteria, queue_family_properties| {
    ///     let mut queue_setup = HashSet::with_capacity(queue_family_criteria.len());
    ///     for queue_family_criteria in queue_family_criteria {
    ///         match queue_family_criteria.choose_queue_family(
    ///             instance,
    ///             physical_device,
    ///             queue_family_properties,
    ///             surface,
    ///         )? {
    ///             Some((idx, _properties)) => {
    ///                 queue_setup.insert(QueueSetup::simple(idx, 1));
    ///             }
    ///             None => return Ok(None),
    ///         }
    ///     }
    ///
    ///     Ok(Some(queue_setup))
    /// }
    /// ```
    #[inline]
    pub fn custom_queue_setup(mut self, custom_queue_setup: Box<CustomQueueSetupFn>) -> Self {
        self.queue_setup_fn = Some(custom_queue_setup);
        self
    }

    /// Allows to specify custom criteria for a physical device.
    /// This can for example be used to check for limits.
    #[inline]
    pub fn additional_suitability(
        mut self,
        additional_suitability: Box<AdditionalSuitabilityFn>,
    ) -> Self {
        self.additional_suitability_fn = Some(additional_suitability);
        self
    }

    /// Surface to use to check for presentation support in queue families.
    #[inline]
    pub fn for_surface(mut self, surface: vk::SurfaceKHR) -> Self {
        self.surface = Some(surface);
        self
    }

    /// Prioritise devices of these types when choosing a device.
    /// The further ahead, the higher the priority.
    #[inline]
    pub fn prioritise_device_types(mut self, types: &[vk::PhysicalDeviceType]) -> Self {
        self.prioritised_device_types = types.into();
        self
    }

    /// The queue family chosen by the criteria will be enabled on the device.
    #[inline]
    pub fn queue_family(mut self, criteria: QueueFamilyCriteria) -> Self {
        self.queue_family_criteria.push(criteria);
        self
    }

    /// Prefer a device which has at least one `DEVICE_LOCAL` memory heap with
    /// a minimum of `size` bytes of memory.
    #[inline]
    pub fn prefer_device_memory_size(mut self, size: vk::DeviceSize) -> Self {
        self.preferred_device_memory_size = Some(size);
        self
    }

    /// Require a device which has at least one `DEVICE_LOCAL` memory heap with
    /// a minimum of `size` bytes of memory.
    #[inline]
    pub fn require_device_memory_size(mut self, size: vk::DeviceSize) -> Self {
        self.required_device_memory_size = Some(size);
        self
    }

    /// Prefer a device which supports `extension`.
    /// The extension will only be enabled if it's supported.
    #[inline]
    pub fn prefer_extension(mut self, extension: *const c_char) -> Self {
        self.extensions.push((extension, false));
        self
    }

    /// Require a device which supports `extension`.
    /// The extension will be enabled.
    #[inline]
    pub fn require_extension(mut self, extension: *const c_char) -> Self {
        self.extensions.push((extension, true));
        self
    }

    /// Prefer a device which supports this version.
    #[inline]
    pub fn prefer_version(mut self, major: u32, minor: u32) -> Self {
        self.preferred_version = Some(vk::make_api_version(0, major, minor, 0));
        self
    }

    /// Prefer a device which supports this version.
    #[inline]
    pub fn prefer_version_raw(mut self, version: u32) -> Self {
        self.preferred_version = Some(version);
        self
    }

    /// Require the device to support this version.
    #[inline]
    pub fn require_version(mut self, major: u32, minor: u32) -> Self {
        self.required_version = vk::make_api_version(0, major, minor, 0);
        self
    }

    /// Require the device to support this version.
    #[inline]
    pub fn require_version_raw(mut self, version: u32) -> Self {
        self.required_version = version;
        self
    }

    /// Require these features to be present for the device.
    /// The elements of the pointer chain will only be considered if possible.
    /// The features will be enabled.
    #[inline]
    pub fn require_features(mut self, features: &'a vk::PhysicalDeviceFeatures2) -> Self {
        self.required_features = Some(features);
        self
    }

    /// Skip the selection logic and always select the physical device at the
    /// specified index.
    #[inline]
    pub fn select_nth_unconditionally(mut self, n: usize) -> Self {
        self.unconditional_nth = Some(n);
        self
    }

    /// Allocation callback to use for internal Vulkan calls in the builder.
    #[inline]
    pub fn allocation_callbacks(mut self, allocator: vk::AllocationCallbacks) -> Self {
        self.allocator = Some(allocator);
        self
    }

    /// Returns the [`erupt::Device`] and [`DeviceMetadata`], containing
    /// the handle of the used physical device handle and its properties, as
    /// wells as the enabled device extensions and used queue setups.
    pub unsafe fn build(
        mut self,
        instance: &Instance,
        surface_loader: &Surface,
        instance_metadata: &InstanceMetadata,
    ) -> Result<(Device, DeviceMetadata), DeviceCreationError> {
        assert_eq!(instance.handle(), instance_metadata.instance_handle());

        let mut queue_setup_fn = self.queue_setup_fn.unwrap_or_else(|| {
            // properly update documentation of the `custom_queue_setup` method
            // when changing this (formatting, used variables)
            Box::new(
                |physical_device, queue_family_criteria, queue_family_properties| {
                    let mut queue_setup = HashSet::with_capacity(queue_family_criteria.len());
                    for queue_family_criteria in queue_family_criteria {
                        match queue_family_criteria.choose_queue_family(
                            surface_loader,
                            physical_device,
                            queue_family_properties,
                            self.surface,
                        )? {
                            Some((idx, _properties)) => {
                                queue_setup.insert(QueueSetup::simple(idx, 1));
                            }
                            None => return Ok(None),
                        }
                    }

                    Ok(Some(queue_setup))
                },
            )
        });

        let physical_devices = instance.enumerate_physical_devices()?;
        let mut devices_properties = physical_devices.into_iter().map(|physical_device| {
            (physical_device, unsafe {
                instance.get_physical_device_properties(physical_device)
            })
        });

        let devices = if let Some(n) = self.unconditional_nth {
            vec![devices_properties
                .nth(n)
                .ok_or(DeviceCreationError::UnconditionalMissing)?]
        } else {
            let mut device_type_preference = self.prioritised_device_types;
            device_type_preference.extend([
                vk::PhysicalDeviceType::DISCRETE_GPU,
                vk::PhysicalDeviceType::INTEGRATED_GPU,
            ]);

            let mut devices_properties: Vec<_> = devices_properties.collect();
            devices_properties.sort_by_key(|(_physical_device, properties)| {
                device_type_preference
                    .iter()
                    .position(|&preference| properties.device_type == preference)
                    .unwrap_or(usize::MAX)
            });

            devices_properties
        };

        struct Candidate {
            physical_device: vk::PhysicalDevice,
            properties: vk::PhysicalDeviceProperties,
            queue_setups: BootstrapSmallVec<QueueSetup>,
            memory_properties: vk::PhysicalDeviceMemoryProperties,
            queue_family_properties: Vec<vk::QueueFamilyProperties>,
            enabled_extensions: BootstrapSmallVec<*const c_char>,
        }

        let mut perfect_candidates = BootstrapSmallVec::new();
        let mut inperfect_candidates = BootstrapSmallVec::new();
        for (physical_device, properties) in devices {
            let mut perfect_candidate = true;

            if self.required_version > properties.api_version {
                continue;
            }

            if let Some(preferred_version) = self.preferred_version {
                if preferred_version > properties.api_version {
                    perfect_candidate = false;
                }
            }

            let memory_properties = instance.get_physical_device_memory_properties(physical_device);
            let queue_family_properties =
                instance.get_physical_device_queue_family_properties(physical_device);

            if self.preferred_device_memory_size.is_some()
                || self.required_device_memory_size.is_some()
            {
                let highest_device_local_memory = memory_properties
                    .memory_heaps
                    .into_iter()
                    .take(memory_properties.memory_heap_count as usize)
                    .filter(|memory_heap| {
                        memory_heap
                            .flags
                            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                    })
                    .map(|memory_heap| memory_heap.size)
                    .max()
                    .expect(
                        "spec violation: At least one heap must include \
                            VK_MEMORY_HEAP_DEVICE_LOCAL_BIT in VkMemoryHeap::flags.",
                    );

                if let Some(preferred_device_memory_size) = self.preferred_device_memory_size {
                    if preferred_device_memory_size > highest_device_local_memory {
                        perfect_candidate = false;
                    }
                }

                if let Some(required_device_memory_size) = self.required_device_memory_size {
                    if required_device_memory_size > highest_device_local_memory {
                        continue;
                    }
                }
            }

            let queue_setup = if self.queue_family_criteria.is_empty() {
                BootstrapSmallVec::new()
            } else {
                match queue_setup_fn(
                    physical_device,
                    &self.queue_family_criteria,
                    &queue_family_properties,
                )? {
                    Some(queue_setup) => queue_setup.into_iter().collect(),
                    None => continue,
                }
            };

            let enabled_extensions = if self.extensions.is_empty() {
                BootstrapSmallVec::new()
            } else {
                let extension_properties =
                    instance.enumerate_device_extension_properties(physical_device)?;
                // TODO load with layer names to get all extension properties.

                let mut enabled_extensions = BootstrapSmallVec::new();
                for &(extension_name, required) in self.extensions.iter() {
                    let cstr = CStr::from_ptr(extension_name);
                    let present = extension_properties.iter().any(|supported_extension| {
                        CStr::from_ptr(supported_extension.extension_name.as_ptr()) == cstr
                    });

                    if present {
                        enabled_extensions.push(extension_name);
                    } else if required {
                        continue;
                    } else {
                        perfect_candidate = false;
                    }
                }

                enabled_extensions
            };

            let candidate = Candidate {
                physical_device,
                properties,
                queue_setups: queue_setup,
                memory_properties,
                queue_family_properties,
                enabled_extensions,
            };

            if let Some(additional_suitability) = self.additional_suitability_fn.as_mut() {
                match additional_suitability(instance, physical_device) {
                    DeviceSuitability::Perfect => (),
                    DeviceSuitability::NotPreferred => perfect_candidate = false,
                    DeviceSuitability::Unsuitable => continue,
                }
            }

            if perfect_candidate {
                perfect_candidates.push(candidate);
            } else {
                inperfect_candidates.push(candidate);
            }
        }

        let features2_supported = instance_metadata.api_version_raw() >= vk::API_VERSION_1_1
            || instance_metadata.is_extension_enabled(
                ash::extensions::khr::GetPhysicalDeviceProperties2::name().as_ptr(),
            );
        for candidate in perfect_candidates
            .into_iter()
            .chain(inperfect_candidates.into_iter())
        {
            let queue_create_infos: BootstrapSmallVec<_> = candidate
                .queue_setups
                .iter()
                .map(QueueSetup::as_vulkan)
                .map(|x| x.build())
                .collect();
            let mut device_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_create_infos)
                .enabled_extension_names(&candidate.enabled_extensions);

            let mut required_features;
            if let Some(&val) = self.required_features {
                required_features = val;

                if features2_supported {
                    device_info = device_info.push_next(&mut required_features);
                } else {
                    device_info = device_info.enabled_features(&required_features.features);
                }
            }

            let device_handle = instance.create_device(
                candidate.physical_device,
                &device_info,
                self.allocator.as_ref(),
            );
            match device_handle {
                Ok(device_handle) => {
                    let device = self
                        .loader_builder
                        .build_with_existing_device(instance, device_handle)?;

                    drop(queue_create_infos);
                    let device_metadata = DeviceMetadata {
                        device_handle: device.handle(),
                        physical_device: candidate.physical_device,
                        properties: candidate.properties,
                        queue_setups: candidate.queue_setups,
                        memory_properties: candidate.memory_properties,
                        queue_family_properties: candidate.queue_family_properties,
                        surface: self.surface,
                        enabled_extensions: candidate
                            .enabled_extensions
                            .into_iter()
                            .map(|ptr| unsafe { CStr::from_ptr(ptr).to_owned() })
                            .collect(),
                    };

                    return Ok((device, device_metadata));
                }
                Err(vk::Result::ERROR_FEATURE_NOT_PRESENT) => continue,
                Err(err) => return Err(err.into()),
            }
        }

        Err(DeviceCreationError::RequirementsNotMet)
    }
}

impl<'a> Default for DeviceLoaderBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}
