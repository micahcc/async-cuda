use cpp::cpp;

use crate::device::DeviceId;
use crate::ffi::device::Device;
use crate::ffi::ptr::DevicePtr;
use crate::ffi::result;
use crate::ffi::stream::Stream;

type Result<T> = std::result::Result<T, crate::error::Error>;

/// Synchronous implementation of [`crate::DeviceTensor`].
///
/// Refer to [`crate::DeviceTensor`] for documentation.
pub struct DeviceTensor<T: Copy> {
    pub shape: Vec<usize>,
    num_elements: usize,
    n_bytes: usize, // in bytes
    internal: DevicePtr,
    device: DeviceId,
    _phantom: std::marker::PhantomData<T>,
}

/// Implements [`Send`] for [`DeviceTensor`].
///
/// # Safety
///
/// This property is inherited from the CUDA API, which is thread-safe.
unsafe impl<T: Copy> Send for DeviceTensor<T> {}

/// Implements [`Sync`] for [`DeviceTensor`].
///
/// # Safety
///
/// This property is inherited from the CUDA API, which is thread-safe.
unsafe impl<T: Copy> Sync for DeviceTensor<T> {}

impl<T: Copy> DeviceTensor<T> {
    pub fn new(shape: &[usize]) -> Self {
        let device = Device::get_or_panic();
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let ptr_ptr = std::ptr::addr_of_mut!(ptr);
        let num_elements = shape.iter().product();
        let n_bytes = num_elements * std::mem::size_of::<T>();
        let ret = cpp!(unsafe [
            ptr_ptr as "void**",
            n_bytes as "std::size_t"
        ] -> i32 as "std::int32_t" {
            return cudaMalloc(ptr_ptr, n_bytes);
        });

        match result!(ret, DevicePtr::from_addr(ptr)) {
            Ok(internal) => Self {
                shape: shape.to_vec(),
                num_elements,
                n_bytes,
                internal,
                device,
                _phantom: Default::default(),
            },
            Err(err) => {
                panic!("failed to allocate device memory: {err}");
            }
        }
    }

    /// Copy from device to host
    ///
    /// # Safety
    ///
    /// This function is marked unsafe because it does not synchronize and the operation might not
    /// have completed when it returns, meaning the array might have data populated after its freed
    /// elsewhere, to ensure safety array alive until synchronize
    pub unsafe fn copy_to_slice_async(&self, array: &mut [T], stream: &Stream) -> Result<()> {
        assert!(array.len() == self.num_elements);
        let src_ptr = self.internal.as_ptr();
        let dst_ptr = array.as_ptr();
        let n_bytes = self.n_bytes;
        let stream_ptr = stream.as_internal().as_ptr();
        let ret = cpp!(unsafe [
            dst_ptr as "void*",
            src_ptr as "void*",
            n_bytes as "std::size_t",
            stream_ptr as "const void*"
        ] -> i32 as "std::int32_t" {
            return cudaMemcpyAsync(dst_ptr, src_ptr, n_bytes, cudaMemcpyDeviceToHost, (cudaStream_t) stream_ptr);
        });
        result!(ret)
    }

    /// Copy from host to device
    ///
    /// # Safety
    ///
    /// This function is marked unsafe because it does not synchronize and the operation might not
    /// have completed when it returns, meaning the array might have data populated after its freed
    /// elsewhere, to ensure safety array alive until synchronize
    pub unsafe fn copy_from_slice_async(&self, array: &[T], stream: &Stream) -> Result<()> {
        assert!(array.len() == self.num_elements);
        let dst_ptr = self.internal.as_ptr();
        let src_ptr = array.as_ptr();
        let n_bytes = self.n_bytes;
        let stream_ptr = stream.as_internal().as_ptr();
        let ret = cpp!(unsafe [
            dst_ptr as "void*",
            src_ptr as "void*",
            n_bytes as "std::size_t",
            stream_ptr as "const void*"
        ] -> i32 as "std::int32_t" {
            return cudaMemcpyAsync(dst_ptr, src_ptr, n_bytes, cudaMemcpyHostToDevice, (cudaStream_t) stream_ptr);
        });
        result!(ret)
    }

    #[cfg(feature = "ndarray")]
    pub fn from_array<D: ndarray::Dimension>(
        array: &ndarray::ArrayView<T, D>,
        stream: &Stream,
    ) -> Result<Self> {
        let mut this = Self::new(array.shape());
        // SAFETY: Safe because the stream is synchronized after this.
        // otherwise the memory could go out of scope
        unsafe {
            this.copy_from_slice_async(array.as_slice(), stream)?;
        }
        stream.synchronize()?;
        Ok(this)
    }

    #[cfg(feature = "ndarray")]
    pub fn to_array<D: ndarray::Dimension>(
        array: &mut ndarray::ArrayView<T, D>,
        stream: &Stream,
    ) -> Result<Self> {
        let mut this = Self::new(array.shape());
        // SAFETY: Safe because the stream is synchronized after this.
        // otherwise the memory could go out of scope
        unsafe {
            this.copy_to_slice(array.as_slice_mut(), stream)?;
        }
        stream.synchronize()?;
        Ok(this)
    }

    #[inline(always)]
    pub fn num_elements(&self) -> usize {
        self.num_elements
    }

    /// Get readonly reference to internal [`DevicePtr`].
    #[inline(always)]
    pub fn as_internal(&self) -> &DevicePtr {
        &self.internal
    }

    /// Get mutable reference to internal [`DevicePtr`].
    #[inline(always)]
    pub fn as_mut_internal(&mut self) -> &mut DevicePtr {
        &mut self.internal
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Release the buffer memory.
    ///
    /// # Panics
    ///
    /// This function panics if binding to the corresponding device fails.
    ///
    /// # Safety
    ///
    /// The buffer may not be used after this function is called, except for being dropped.
    pub unsafe fn free(&mut self) {
        if self.internal.is_null() {
            return;
        }

        Device::set_or_panic(self.device);

        // SAFETY: Safe because we won't use pointer after this.
        let mut internal = unsafe { self.internal.take() };
        let ptr = internal.as_mut_ptr();
        let _ret = cpp!(unsafe [
            ptr as "void*"
        ] -> i32 as "std::int32_t" {
            return cudaFree(ptr);
        });
    }
}

impl<T: Copy> Drop for DeviceTensor<T> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: This is safe since the buffer cannot be used after this.
        unsafe {
            self.free();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let buffer = DeviceTensor::<u32>::new(&[120, 80, 3]);
        assert_eq!(buffer.shape()[0], 120);
        assert_eq!(buffer.shape()[1], 80);
        assert_eq!(buffer.shape()[2], 3);
        assert_eq!(buffer.num_elements(), 120 * 80 * 3);
    }

    #[test]
    fn test_copy() {
        let stream = Stream::new().unwrap();
        let all_ones = vec![1_u32; 150];
        let mut middle = vec![0_u32; 150];
        let mut out = vec![0_u32; 150];

        let device_buffer = DeviceTensor::<u32>::new(&[10, 5, 3]);
        unsafe {
            device_buffer
                .copy_from_slice_async(&all_ones, &stream)
                .unwrap();
        }

        unsafe {
            device_buffer
                .copy_to_slice_async(&mut middle, &stream)
                .unwrap();
        }

        let another_device_buffer = DeviceTensor::<u32>::new(&[10, 5, 3]);
        unsafe {
            another_device_buffer
                .copy_from_slice_async(&middle, &stream)
                .unwrap();
        }

        unsafe {
            another_device_buffer
                .copy_to_slice_async(&mut out, &stream)
                .unwrap();
        }

        stream.synchronize().unwrap();

        assert_eq!(out.len(), 150);
        assert!(out.into_iter().all(|v| v == 1_u32));
    }

    #[test]
    fn test_copy_2d() {
        let stream = Stream::new().unwrap();
        let image: [u8; 12] = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4];
        let device_buffer = DeviceTensor::<u8>::new(&[2, 2, 3]);
        unsafe {
            device_buffer
                .copy_from_slice_async(&image, &stream)
                .unwrap();
        }
        let mut return_host_buffer = [0_u8; 12];
        unsafe {
            device_buffer
                .copy_to_slice_async(&mut return_host_buffer, &stream)
                .unwrap();
        }
        stream.synchronize().unwrap();
        assert_eq!(&return_host_buffer, &[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]);
    }

    #[test]
    #[should_panic]
    fn test_it_panics_when_copying_invalid_size() {
        let stream = Stream::new().unwrap();
        let device_buffer = DeviceTensor::<u32>::new(&[5, 5, 3]);
        let mut host_buffer = [0_u32; 80];
        let _ = unsafe { device_buffer.copy_to_slice_async(&mut host_buffer, &stream) };
    }
}
