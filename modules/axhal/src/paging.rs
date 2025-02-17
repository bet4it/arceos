//! Page table manipulation.

use axalloc::global_allocator;
use page_table::PagingIf;

use crate::mem::{phys_to_virt, virt_to_phys, MemRegionFlags, PhysAddr, VirtAddr, PAGE_SIZE_4K};

#[doc(no_inline)]
pub use page_table::{MappingFlags, PageSize, PagingError, PagingResult};

impl From<MemRegionFlags> for MappingFlags {
    fn from(f: MemRegionFlags) -> Self {
        let mut ret = Self::empty();
        if f.contains(MemRegionFlags::READ) {
            ret |= Self::READ;
        }
        if f.contains(MemRegionFlags::WRITE) {
            ret |= Self::WRITE;
        }
        if f.contains(MemRegionFlags::EXECUTE) {
            ret |= Self::EXECUTE;
        }
        if f.contains(MemRegionFlags::DEVICE) {
            ret |= Self::DEVICE;
        }
        ret
    }
}

/// Implementation of [`PagingIf`], to provide physical memory manipulation to
/// the [page_table] crate.
pub struct PagingIfImpl;

impl PagingIf for PagingIfImpl {
    fn new() -> Self {
        Self
    }

    fn alloc_frame(&self) -> Option<PhysAddr> {
        global_allocator()
            .alloc_pages(1, PAGE_SIZE_4K)
            .map(|vaddr| virt_to_phys(vaddr.into()))
            .ok()
    }

    #[cfg(target_arch = "riscv64")]
    fn alloc_frames(&self, page_nums: usize) -> Option<PhysAddr> {
        global_allocator()
            .alloc_pages(page_nums, PAGE_SIZE_4K * page_nums)
            .map(|vaddr| virt_to_phys(vaddr.into()))
            .ok()
    }

    fn dealloc_frame(&self, paddr: PhysAddr) {
        global_allocator().dealloc_pages(phys_to_virt(paddr).as_usize(), 1)
    }

    #[cfg(target_arch = "riscv64")]
    fn dealloc_frames(&self, paddr: PhysAddr, page_nums: usize) {
        global_allocator().dealloc_pages(phys_to_virt(paddr).as_usize(), page_nums)
    }

    #[inline]
    fn phys_to_virt(&self, paddr: PhysAddr) -> VirtAddr {
        phys_to_virt(paddr)
    }
}

cfg_if::cfg_if! {
    if #[cfg(target_arch = "x86_64")] {
        /// The architecture-specific page table.
        pub type PageTable = page_table::x86_64::X64PageTable<PagingIfImpl>;
    } else if #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))] {
        /// The architecture-specific page table.
        pub type PageTable = page_table::riscv::Sv39PageTable<PagingIfImpl>;
    } else if #[cfg(target_arch = "aarch64")]{
        /// The architecture-specific page table.
        pub type PageTable = page_table::aarch64::A64PageTable<PagingIfImpl>;
    }
}
