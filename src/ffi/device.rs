use std::ffi::c_char;
use std::ffi::c_int;

use cpp::cpp;

use crate::device::DeviceId;
use crate::device::MemoryInfo;
use crate::ffi::result;

type Result<T> = std::result::Result<T, crate::error::Error>;

#[derive(Clone, Debug)]
#[repr(C)]
pub struct CudaDeviceProperties {
    pub can_map_host_memory: i32,
    pub clock_rate: i32,
    pub device_overlap: i32,
    pub kernel_exec_timeout_enabled: i32,
    pub major: i32,
    pub max_grid_size: [i32; 3],
    pub max_threads_dim: [i32; 3],
    pub max_threads_per_block: i32,
    pub mem_pitch: usize,
    pub memory_bus_width: i32,
    pub memory_clock_rate: i32,
    pub minor: i32,
    pub multi_processor_count: i32,
    pub name: [c_char; 256],
    pub regs_per_block: i32,
    pub shared_mem_per_block: usize,
    pub texture_alignment: usize,
    pub total_const_mem: usize,
    pub total_global_mem: usize,
    pub warp_size: i32,
    pub pci_bus_id: i32,
    pub pci_device_id: i32,
    pub pci_domain_id: i32,
}

impl Default for CudaDeviceProperties {
    fn default() -> Self {
        Self {
            clock_rate: Default::default(),
            device_overlap: Default::default(),
            kernel_exec_timeout_enabled: Default::default(),
            major: Default::default(),
            max_grid_size: Default::default(),
            can_map_host_memory: Default::default(),
            max_threads_dim: Default::default(),
            max_threads_per_block: Default::default(),
            mem_pitch: Default::default(),
            memory_bus_width: Default::default(),
            memory_clock_rate: Default::default(),
            minor: Default::default(),
            multi_processor_count: Default::default(),
            name: [0; 256],
            regs_per_block: Default::default(),
            shared_mem_per_block: Default::default(),
            texture_alignment: Default::default(),
            total_const_mem: Default::default(),
            total_global_mem: Default::default(),
            warp_size: Default::default(),
            pci_bus_id: 0,
            pci_device_id: 0,
            pci_domain_id: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
#[repr(C)]
pub struct CudaUUID {
    pub bytes: [u8; 16],
}

/// Synchronous implementation of [`crate::num_devices`].
///
/// Refer to [`crate::num_devices`] for documentation.
pub fn num_devices() -> Result<usize> {
    let mut num = 0_i32;
    let num_ptr = std::ptr::addr_of_mut!(num);
    let ret = cpp!(unsafe [
        num_ptr as "std::int32_t*"
    ] -> i32 as "std::int32_t" {
        return cudaGetDeviceCount(num_ptr);
    });

    result!(ret, num as usize)
}

/// Synchronous implementation of [`crate::Device`].
///
/// Refer to [`crate::Device`] for documentation.
pub struct Device;

impl Device {
    #[inline]
    pub fn get() -> Result<DeviceId> {
        let mut id: i32 = 0;
        let id_ptr = std::ptr::addr_of_mut!(id);
        let ret = cpp!(unsafe [
            id_ptr as "int*"
        ] -> i32 as "int" {
            return cudaGetDevice(id_ptr);
        });
        result!(ret, id)
    }

    #[inline(always)]
    pub fn get_or_panic() -> DeviceId {
        Device::get().unwrap_or_else(|err| panic!("failed to get device: {err}"))
    }

    #[inline]
    pub fn set(id: DeviceId) -> Result<()> {
        let ret = cpp!(unsafe [
            id as "int"
        ] -> i32 as "int" {
            return cudaSetDevice(id);
        });
        result!(ret)
    }

    #[inline]
    pub fn get_properties(id: DeviceId) -> Result<CudaDeviceProperties> {
        let mut props = CudaDeviceProperties::default();
        cpp! {{
            #include <cuda_runtime_api.h>
            #include <cstring>
        }}
        let can_map_host_memory_ptr = std::ptr::addr_of_mut!(props.can_map_host_memory);
        let clock_rate_ptr = std::ptr::addr_of_mut!(props.clock_rate);
        let device_overlap_ptr = std::ptr::addr_of_mut!(props.device_overlap);
        let kernel_exec_timeout_enabled_ptr =
            std::ptr::addr_of_mut!(props.kernel_exec_timeout_enabled);
        let major_ptr = std::ptr::addr_of_mut!(props.major);
        let max_grid_size_ptr = std::ptr::addr_of_mut!(props.max_grid_size);
        let max_threads_dim_ptr = std::ptr::addr_of_mut!(props.max_threads_dim);
        let max_threads_per_block_ptr = std::ptr::addr_of_mut!(props.max_threads_per_block);
        let mem_pitch_ptr = std::ptr::addr_of_mut!(props.mem_pitch);
        let memory_bus_width_ptr = std::ptr::addr_of_mut!(props.memory_bus_width);
        let memory_clock_rate_ptr = std::ptr::addr_of_mut!(props.memory_clock_rate);
        let minor_ptr = std::ptr::addr_of_mut!(props.minor);
        let multi_processor_count_ptr = std::ptr::addr_of_mut!(props.multi_processor_count);
        let name_ptr = std::ptr::addr_of_mut!(props.name);
        let regs_per_block_ptr = std::ptr::addr_of_mut!(props.regs_per_block);
        let shared_mem_per_block_ptr = std::ptr::addr_of_mut!(props.shared_mem_per_block);
        let texture_alignment_ptr = std::ptr::addr_of_mut!(props.texture_alignment);
        let total_const_mem_ptr = std::ptr::addr_of_mut!(props.total_const_mem);
        let total_global_mem_ptr = std::ptr::addr_of_mut!(props.total_global_mem);
        let warp_size_ptr = std::ptr::addr_of_mut!(props.warp_size);
        let pci_bus_id_ptr = std::ptr::addr_of_mut!(props.pci_bus_id);
        let pci_device_id_ptr = std::ptr::addr_of_mut!(props.pci_device_id);
        let pci_domain_id_ptr = std::ptr::addr_of_mut!(props.pci_domain_id);

        let ret = cpp!(unsafe [
            id as "int",
            can_map_host_memory_ptr as "int*",
            clock_rate_ptr as "int*",
            device_overlap_ptr as "int*",
            kernel_exec_timeout_enabled_ptr as "int*",
            major_ptr as "int*",
            max_grid_size_ptr as "int*",
            max_threads_dim_ptr as "int*",
            max_threads_per_block_ptr as "int*",
            mem_pitch_ptr as "unsigned long long*",
            memory_bus_width_ptr as "int*",
            memory_clock_rate_ptr as "int*",
            minor_ptr as "int*",
            multi_processor_count_ptr as "int*",
            name_ptr as "char*",
            regs_per_block_ptr as "int*",
            shared_mem_per_block_ptr as "unsigned long long*",
            texture_alignment_ptr as "unsigned long long*",
            total_const_mem_ptr as "unsigned long long*",
            total_global_mem_ptr as "unsigned long long*",
            warp_size_ptr as "int*",
            pci_bus_id_ptr as "int*",
            pci_domain_id_ptr as "int*",
            pci_device_id_ptr as "int*"
        ] -> i32 as "int" {
            cudaDeviceProp props;
            const auto ret = cudaGetDeviceProperties(&props, id);
            *can_map_host_memory_ptr = props.canMapHostMemory;
            *clock_rate_ptr = props.clockRate;
            *device_overlap_ptr = props.deviceOverlap;
            *kernel_exec_timeout_enabled_ptr = props.kernelExecTimeoutEnabled;
            *major_ptr = props.major;

            for(int i = 0; i < 3; ++i) {
                max_grid_size_ptr[i] = props.maxGridSize[i];
                max_threads_dim_ptr[i] = props.maxThreadsDim[i];
            }
            *max_threads_per_block_ptr = props.maxThreadsPerBlock;
            *mem_pitch_ptr = props.memPitch;
            *memory_bus_width_ptr = props.memoryBusWidth;
            *memory_clock_rate_ptr = props.memoryClockRate;
            *minor_ptr = props.minor;
            *multi_processor_count_ptr = props.multiProcessorCount;
            *regs_per_block_ptr = props.regsPerBlock;
            *shared_mem_per_block_ptr = props.sharedMemPerBlock;
            *texture_alignment_ptr = props.textureAlignment;
            *total_const_mem_ptr = props.totalConstMem;
            *total_global_mem_ptr = props.totalGlobalMem;
            *warp_size_ptr = props.warpSize;
            *pci_bus_id_ptr = props.pciBusID;
            *pci_device_id_ptr = props.pciDeviceID;
            *pci_domain_id_ptr = props.pciDomainID;
            memcpy(name_ptr, props.name, 256);
            return ret;
        });

        result!(ret, props)
    }

    #[inline(always)]
    pub fn set_or_panic(id: DeviceId) {
        Device::set(id).unwrap_or_else(|err| panic!("failed to set device {id}: {err}"));
    }

    pub fn synchronize() -> Result<()> {
        let ret = cpp!(unsafe [] -> i32 as "std::int32_t" {
            return cudaDeviceSynchronize();
        });
        result!(ret)
    }

    pub fn memory_info() -> Result<MemoryInfo> {
        let mut free: usize = 0;
        let free_ptr = std::ptr::addr_of_mut!(free);
        let mut total: usize = 0;
        let total_ptr = std::ptr::addr_of_mut!(total);

        let ret = cpp!(unsafe [
            free_ptr as "std::size_t*",
            total_ptr as "std::size_t*"
        ] -> i32 as "std::int32_t" {
            return cudaMemGetInfo(free_ptr, total_ptr);
        });
        result!(ret, MemoryInfo { free, total })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_devices() {
        assert!(matches!(num_devices(), Ok(num) if num > 0));
    }

    #[test]
    fn test_get_device() {
        assert!(matches!(Device::get(), Ok(0)));
    }

    #[test]
    fn test_set_device() {
        assert!(Device::set(0).is_ok());
        assert!(matches!(Device::get(), Ok(0)));
    }

    #[test]
    fn test_synchronize() {
        assert!(Device::synchronize().is_ok());
    }

    #[test]
    fn test_device_properties() {
        assert!(Device::get_properties(0).is_ok());
        eprintln!("{:?}", Device::get_properties(0));
    }

    #[test]
    fn test_memory_info() {
        let memory_info = Device::memory_info().unwrap();
        assert!(memory_info.free > 0);
        assert!(memory_info.total > 0);
    }
}
