# ash-bootstrap

[![docs.rs](https://docs.rs/erupt-bootstrap/badge.svg)](https://docs.rs/erupt-bootstrap)
[![crates.io](https://img.shields.io/crates/v/erupt-bootstrap.svg)](https://crates.io/crates/erupt-bootstrap)

Vulkan Bootstrapping library for Rust, inspired by [`vk-bootstrap`].

- ✅ Instance creation
- ✅ Physical Device selection
- ✅ Device creation
- ✅ Getting queues
- ✅ Swapchain handling was handled in erupt-bootstrap, but ash_window takes care of creation, we handle swapchain creation/recreation in this lib.

## Cargo Features

- `surface` (enabled by default): Enables the use of [`raw-window-handle`].

## Example

see the examples dir for up to date examples

## Licensing

This project is licensed under the [zlib License].

`vk-bootstrap`, the inspiration of this project, is licensed under the [MIT license].

`erupt-bootsrap` is the main initial work of this by Friz64.  He's the real hero, I just work here.

[zlib License]: https://gitlab.com/Friz64/erupt-bootstrap/-/blob/main/LICENSE
[MIT license]: https://gitlab.com/Friz64/erupt-bootstrap/-/blob/main/LICENSE-vk-bootstrap
[`vk-bootstrap`]: https://github.com/charles-lunarg/vk-bootstrap
[`raw-window-handle`]: https://crates.io/crates/raw-window-handle
