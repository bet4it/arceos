use axhal::mem::{PhysAddr, VirtAddr};

use hypercraft::{GuestPageTableTrait, GuestPhysAddr, HyperError, HyperResult, NestedPageTable};

use page_table_entry::MappingFlags;

pub type GuestPagingIfImpl = axhal::paging::PagingIfImpl;

/// Guest Page Table struct\
pub struct GuestPageTable(NestedPageTable<GuestPagingIfImpl>);

impl GuestPageTableTrait for GuestPageTable {
    fn new() -> HyperResult<Self> {
        #[cfg(target_arch = "riscv64")]
        {
            let npt = NestedPageTable::<GuestPagingIfImpl>::try_new_gpt()
                .map_err(|_| HyperError::NoMemory)?;
            Ok(GuestPageTable(npt))
        }
        #[cfg(target_arch = "aarch64")]
        {
            let agpt = NestedPageTable::<GuestPagingIfImpl>::try_new()
            .map_err(|_| HyperError::NoMemory)?;
            Ok(GuestPageTable(agpt))
        }
        #[cfg(target_arch = "x86_64")]
        {
            let npt = NestedPageTable::<GuestPagingIfImpl>::try_new()
                .map_err(|_| HyperError::NoMemory)?;
            Ok(GuestPageTable(npt))
        }
        #[cfg(not(any(target_arch = "riscv64", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            todo!()
        }
    }

    fn map(
        &mut self,
        gpa: GuestPhysAddr,
        hpa: hypercraft::HostPhysAddr,
        flags: MappingFlags,
    ) -> HyperResult<()> {
        #[cfg(any(target_arch = "riscv64", target_arch = "x86_64", target_arch = "aarch64"))]
        {
            self.0
                .map(
                    VirtAddr::from(gpa),
                    PhysAddr::from(hpa),
                    page_table::PageSize::Size4K,
                    flags,
                )
                .map_err(|paging_err| {
                    error!("paging error: {:?}", paging_err);
                    HyperError::Internal
                })?;
            Ok(())
        }
        #[cfg(not(any(target_arch = "riscv64", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            todo!()
        }
    }

    fn map_region(
        &mut self,
        gpa: GuestPhysAddr,
        hpa: hypercraft::HostPhysAddr,
        size: usize,
        flags: MappingFlags,
    ) -> HyperResult<()> {
        #[cfg(any(target_arch = "riscv64", target_arch = "x86_64"))]
        {
            self.0
                .map_region(VirtAddr::from(gpa), PhysAddr::from(hpa), size, flags, true)
                .map_err(|err| {
                    error!("paging error: {:?}", err);
                    HyperError::Internal
                })?;
            Ok(())
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.0
                .map_region(VirtAddr::from(gpa), PhysAddr::from(hpa), size, flags, true)
                .map_err(|err| {
                    error!("paging error: {:?}", err);
                    HyperError::Internal
                })?;
            Ok(())
        }
        #[cfg(not(any(target_arch = "riscv64", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            todo!()
        }
    }

    fn unmap(&mut self, gpa: GuestPhysAddr) -> HyperResult<()> {
        #[cfg(any(target_arch = "riscv64", target_arch = "x86_64", target_arch = "aarch64"))]
        {
            let (_, _) = self.0.unmap(VirtAddr::from(gpa)).map_err(|paging_err| {
                error!("paging error: {:?}", paging_err);
                return HyperError::Internal;
            })?;
            Ok(())
        }
        #[cfg(not(any(target_arch = "riscv64", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            todo!()
        }
    }

    fn translate(&self, gpa: GuestPhysAddr) -> HyperResult<hypercraft::HostPhysAddr> {
        #[cfg(any(target_arch = "riscv64", target_arch = "x86_64", target_arch = "aarch64"))]
        {
            let (addr, _, _) = self.0.query(VirtAddr::from(gpa)).map_err(|paging_err| {
                info!("paging error: {:?}", paging_err);
                HyperError::Internal
            })?;
            Ok(addr.into())
        }
        #[cfg(not(any(target_arch = "riscv64", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            todo!()
        }
    }

    fn token(&self) -> usize {
        #[cfg(any(target_arch = "riscv64", target_arch = "x86_64"))]
        {
            8usize << 60 | usize::from(self.0.root_paddr()) >> 12
        }
        #[cfg(target_arch = "aarch64")]
        {
            usize::from(self.0.root_paddr())  // need to lrs 1 bit for CnP??
        }
        #[cfg(not(any(target_arch = "riscv64", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            todo!()
        }
    }

    fn read_guest_phys_addrs(
        &self,
        paddr: GuestPhysAddr,
        buf: *mut u8,
        size: usize,
    ) -> HyperResult<usize> {
        let addr = self.translate(paddr)?;
        self.0
            .read_phys_addrs(addr.into(), buf, size)
            .map_err(|_| HyperError::PageFault)
    }

    fn write_guest_phys_addrs(
        &mut self,
        paddr: GuestPhysAddr,
        buf: *const u8,
        size: usize,
    ) -> HyperResult<()> {
        let addr = self.translate(paddr)?;
        self.0
            .write_phys_addrs(addr.into(), buf, size)
            .map_err(|_| HyperError::PageFault)
    }

    fn root_paddr(&self) -> hypercraft::HostPhysAddr {
        self.0.root_paddr().into()
    }
}
