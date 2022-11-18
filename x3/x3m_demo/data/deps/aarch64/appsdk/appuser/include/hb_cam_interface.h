/***************************************************************************
* COPYRIGHT NOTICE
* Copyright 2019 Horizon Robotics, Inc.
* All rights reserved.
***************************************************************************/
#ifndef __HB_CAMERA_H__
#define __HB_CAMERA_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define X2_CAM_DEBUG (1)
#define cam_err(format, ...) printf("[%s]%s[%d] E: "format"\n",__TIME__, __func__, __LINE__, ##__VA_ARGS__)
#define cam_log(format, ...) printf("[%s]%s[%d] W: "format"\n",__TIME__, __func__, __LINE__, ##__VA_ARGS__)
#if X2_CAM_DEBUG
#define cam_dbg(format, ...) printf("[%s]%s[%d] D: "format"\n",__TIME__, __func__, __LINE__, ##__VA_ARGS__)
#else
#define cam_dbg(format, ...)
#endif

#define HB_CAMERA_NAME_LENGTH  16
#define HB_CAMERA_CALIB_PATH_LEGNTH 64

enum BLOCK_ACCESS_DEVICE_TYPE{
	SENSOR_DEVICE,
	ISP_DEVICE,
	EEPROM_DEVICE,
	IMU_DEVICE,
	DEVICE_INVALID
};

typedef struct AWB_DATA {
	uint16_t WBG_R;
	uint16_t WBG_GR;
	uint16_t WBG_GB;
	uint16_t WBG_B;
}AWB_DATA_s;

typedef struct img_addr_info_s {
	uint16_t width;
	uint16_t height;
	uint16_t stride;
	uint64_t y_paddr;
	uint64_t c_paddr;
	uint64_t y_vaddr;
	uint64_t c_vaddr;
} img_addr_info_t;

typedef struct cam_img_info_s {
	int g_id;
	int slot_id;
	int cam_id;
	int frame_id;
	int64_t timestamp;
	img_addr_info_t img_addr;
} cam_img_info_t;

typedef enum hb_camera_mode_e {
    HB_NORMAL_M = 1,
    HB_DOL2_M = 2,
    HB_DOL3_M = 3,
    HB_DOL4_M = 4,
    HB_PWL_M = 5,
    HB_INVALID_MOD,
}camera_mode_e;

typedef struct {
	char name[HB_CAMERA_NAME_LENGTH];
	camera_mode_e mode;
	uint8_t bit_width;
	uint8_t cfa_pattern;
	unsigned char calib_path[HB_CAMERA_CALIB_PATH_LEGNTH];
} camera_info_table_t;

extern int hb_cam_init(uint32_t cfg_index, const char *cfg_file);
extern int hb_cam_deinit(uint32_t cfg_index);
extern int hb_cam_start(uint32_t port);
extern int hb_cam_stop(uint32_t port);
extern int hb_cam_start_all();
extern int hb_cam_stop_all();
extern int hb_cam_reset(uint32_t port);
extern int hb_cam_power_on(uint32_t port);
extern int hb_cam_power_off(uint32_t port);
extern int hb_cam_get_fps(uint32_t port, uint32_t *fps);
extern int hb_cam_i2c_read(uint32_t port, uint32_t reg_addr);
extern int hb_cam_i2c_read_byte(uint32_t port, uint32_t reg_addr);
extern int hb_cam_i2c_write(uint32_t port, uint32_t reg_addr, uint16_t value);
extern int hb_cam_i2c_write_byte(uint32_t port, uint32_t reg_addr, uint8_t value);
extern int hb_cam_i2c_block_write(uint32_t port, uint32_t subdev, uint32_t reg_addr, char *buffer, uint32_t size);
extern int hb_cam_i2c_block_read(uint32_t port, uint32_t subdev, uint32_t reg_addr, char *buffer, uint32_t size);
extern int hb_cam_dynamic_switch(uint32_t port, uint32_t fps, uint32_t resolution);
extern int hb_cam_dynamic_switch_fps(uint32_t port, uint32_t fps);
extern int hb_cam_get_img(cam_img_info_t *cam_img_info);
extern int hb_cam_free_img(cam_img_info_t *cam_img_info);
extern int hb_cam_clean_img(cam_img_info_t *cam_img_info);
extern int hb_cam_extern_isp_reset(uint32_t port);
extern int hb_cam_extern_isp_poweroff(uint32_t port);
extern int hb_cam_extern_isp_poweron(uint32_t port);
extern int hb_cam_spi_block_write(uint32_t port, uint32_t subdev, uint32_t reg_addr, char *buffer, uint32_t size);
extern int hb_cam_spi_block_read(uint32_t port, uint32_t subdev, uint32_t reg_addr, char *buffer, uint32_t size);
extern int hb_cam_set_mclk(uint32_t entry_num, uint32_t mclk);
extern int hb_cam_enable_mclk(uint32_t entry_num);
extern int hb_cam_disable_mclk(uint32_t entry_num);
extern int hb_cam_dynamic_switch_mode(uint32_t port, camera_info_table_t *sns_table);

#ifdef __cplusplus
}
#endif

#endif
