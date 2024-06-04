//! Hypervisor related functions

pub use axhal::mem::{phys_to_virt, virt_to_phys, PhysAddr};
pub use axruntime::GuestPageTable;
pub use axruntime::HyperCraftHalImpl;
pub use hypercraft::GuestPageTableTrait;

pub use hypercraft::HyperError as Error;
pub use hypercraft::HyperResult as Result;
pub use hypercraft::HyperCraftHal;
pub use hypercraft::{PerCpu, VCpu};
#[cfg(not(target_arch = "x86_64"))]
pub use hypercraft::{VmCpus, VM};
#[cfg(not(target_arch = "aarch64"))]
pub use hypercraft::{HyperCallMsg, VmExitInfo, GuestPhysAddr, GuestVirtAddr, HostPhysAddr, HostVirtAddr};
