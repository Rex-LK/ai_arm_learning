/*************************************************************************
 *                     COPYRIGHT NOTICE
 *            Copyright 2020-2022 Horizon Robotics, Inc.
 *                   All rights reserved.
 *************************************************************************/

#ifndef __HBMEM_H__
#define __HBMEM_H__

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>

#define HBMEM_VERSION_MAJOR 0
#define HBMEM_VERSION_MINOR 1
#define HBMEM_VERSION_PATCH 1

#define hbmem_addr_t uint64_t

enum hbmem_backends {
	BACKEND_ION_CMA,
	BACKEND_ION_CARVEOUT,
	BACKEND_ION_SRAM,

	BACKEND_MAX
};

/* hbmem alloc flag type */

/* flag direct to backend */
#define BACKEND_FLAG(_flag)	((uint64_t)_flag << 32)
/* backend type which value from enum hbmem_backends */
#define BACKEND_TYPE(backend)	((uint64_t)backend << 24)

#define MEM_CACHEABLE (0x1 << 0)

int32_t hbmem_version(uint32_t *major, uint32_t *minor, uint32_t *patch);

/* Alloc physical continuous memory from hobot mem manager */
hbmem_addr_t hbmem_alloc(uint32_t size, uint64_t flag, const char* label);
void hbmem_free(hbmem_addr_t addr);

/* mmap continous memory to hobot mem manager for which not alloc it */
hbmem_addr_t hbmem_mmap(uint64_t phyaddr, uint32_t size, uint64_t flag);
void hbmem_munmap(hbmem_addr_t addr);

/* use dma to copy data between to hbmem buffer */
int32_t hbmem_dmacpy(hbmem_addr_t dst, hbmem_addr_t src, uint32_t size);

int32_t hbmem_is_cacheable(hbmem_addr_t addr);

/* invalide cache data for read from DDR */
void hbmem_cache_invalid(hbmem_addr_t addr, uint32_t size);
/* Flush cache to DDR and clean the cache */
void hbmem_cache_clean(hbmem_addr_t addr, uint32_t size);

/* Get the physical address */
uint64_t hbmem_phyaddr(hbmem_addr_t addr);

/*
 * most platform hbmem_addrt == virtual addr,
 * the api main for exception platform
 */
uint64_t hbmem_virtaddr(hbmem_addr_t addr);

/*
 * Get the information of the hbmem addr
 * <0: not found in hbmem manager
 * >0: addr type
 * start: the addr belong to memory block start addr
 * size: the addr belong to memory block size
 */
int32_t hbmem_info(hbmem_addr_t addr, hbmem_addr_t *start, uint32_t *size);

#ifdef __cplusplus
}
#endif
#endif
