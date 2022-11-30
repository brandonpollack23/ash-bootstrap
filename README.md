<img src="logo.svg" height=200>

# TODO add documentation, rename to ash-bootstrap completely

# erupt-bootstrap

[![docs.rs](https://docs.rs/erupt-bootstrap/badge.svg)](https://docs.rs/erupt-bootstrap)
[![crates.io](https://img.shields.io/crates/v/erupt-bootstrap.svg)](https://crates.io/crates/erupt-bootstrap)

Vulkan Bootstrapping library for Rust, inspired by [`vk-bootstrap`].

- ✅ Instance creation
- ✅ Physical Device selection
- ✅ Device creation
- ✅ Getting queues
- ✅ Swapchain handling ([courtesy of Ralith](https://github.com/MaikKlein/ash/pull/506) - thanks!)

## Cargo Features

- `surface` (enabled by default): Enables the use of [`raw-window-handle`].

## Example

```rust
let entry = erupt::EntryLoader::new().unwrap();
let instance_builder = InstanceBuilder::new()
    .validation_layers(ValidationLayers::Request)
    .request_debug_messenger(DebugMessenger::Default)
    .require_surface_extensions(&window)
    .unwrap();
let (instance, debug_messenger, instance_metadata) =
    unsafe { instance_builder.build(&entry) }.unwrap();

let surface =
    unsafe { erupt::utils::surface::create_surface(&instance, &window, None) }.unwrap();

let graphics_present = QueueFamilyCriteria::graphics_present();
let transfer = QueueFamilyCriteria::preferably_separate_transfer();

let device_features = vk::PhysicalDeviceFeatures2Builder::new()
    .features(vk::PhysicalDeviceFeaturesBuilder::new().build());

let device_builder = DeviceBuilder::new()
    .queue_family(graphics_present)
    .queue_family(transfer)
    .require_features(&device_features)
    .for_surface(surface);
let (device, device_metadata) =
    unsafe { device_builder.build(&instance, &instance_metadata) }.unwrap();
let graphics_present = device_metadata
    .device_queue(&instance, &device, graphics_present, 0)
    .unwrap()
    .unwrap();
let transfer = device_metadata
    .device_queue(&instance, &device, transfer, 0)
    .unwrap()
    .unwrap();
```

For more examples, visit the [git repo](https://gitlab.com/Friz64/erupt-bootstrap/-/tree/main/examples).

## Licensing

The logo contains the Volcano Emoji of [Twemoji](https://twemoji.twitter.com/)
([License](https://creativecommons.org/licenses/by/4.0/)). The name "erupt" was
added on top of the volcano. The boot is the ["Hiking Boot" from Openclipart](
https://openclipart.org/detail/182950/hiking-boot),
released into the Public Domain.

This project is licensed under the [zlib License].

`vk-bootstrap`, the inspiration of this project, is licensed under the [MIT license].

[zlib License]: https://gitlab.com/Friz64/erupt-bootstrap/-/blob/main/LICENSE
[MIT license]: https://gitlab.com/Friz64/erupt-bootstrap/-/blob/main/LICENSE-vk-bootstrap
[`vk-bootstrap`]: https://github.com/charles-lunarg/vk-bootstrap
[`raw-window-handle`]: https://crates.io/crates/raw-window-handle
