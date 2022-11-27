//
// Created by Minghao Wang on 17-1-5.
// Copyright (c) 2017 Horizon Robotics. All rights reserved.
//

#ifndef HOBOTLOG_HOBOTLOG_HPP
#define HOBOTLOG_HOBOTLOG_HPP

#include "hobotlog/checks.h"
#include "hobotlog/logging.h"

// check given condition, if failed abort will be called
// always valid no matter NDEBUG is defined or not
#define HOBOT_CHECK(condition) RTC_CHECK(condition)
#define HOBOT_CHECK_EQ(v1, v2) RTC_CHECK_EQ(v1, v2)
#define HOBOT_CHECK_NE(v1, v2) RTC_CHECK_NE(v1, v2)
#define HOBOT_CHECK_LE(v1, v2) RTC_CHECK_LE(v1, v2)
#define HOBOT_CHECK_LT(v1, v2) RTC_CHECK_LT(v1, v2)
#define HOBOT_CHECK_GE(v1, v2) RTC_CHECK_GE(v1, v2)
#define HOBOT_CHECK_GT(v1, v2) RTC_CHECK_GT(v1, v2)

// debug check, same as the above check
// but only valid when NDEBUG is not defined
#define HOBOT_DCHECK(condition) RTC_DCHECK(condition)
#define HOBOT_DCHECK_EQ(v1, v2) RTC_DCHECK_EQ(v1, v2)
#define HOBOT_DCHECK_NE(v1, v2) RTC_DCHECK_NE(v1, v2)
#define HOBOT_DCHECK_LE(v1, v2) RTC_DCHECK_LE(v1, v2)
#define HOBOT_DCHECK_LT(v1, v2) RTC_DCHECK_LT(v1, v2)
#define HOBOT_DCHECK_GE(v1, v2) RTC_DCHECK_GE(v1, v2)
#define HOBOT_DCHECK_GT(v1, v2) RTC_DCHECK_GT(v1, v2)

// when use direct enum value(HOBOT_LOG_DEBUG, HOBOT_LOG_ERROR, etc)
// to set log level, use this macro. default release log level is
// HOBOT_LOG_NULL; default debug log level is HOBOT_LOG_INFO
// eg:SetLogLevel(HOBOT_LOG_INFO);
#define SetLogLevel(sev) rtc::LogMessage::LogToDebug(rtc::sev)

// when use a variable to set log level, use this macro
// eg: auto lv = rtc::HOBOT_LOG_DEBUG; SetLogLevelVar(lv)
#define SetLogLevelVar(sev) rtc::LogMessage::LogToDebug(sev)
#define GetLogLevel() rtc::LogMessage::GetLogToDebug()

// set whether record timestamp(time millis relative to log start time)
// default is false
#define LogTimestamps(flag) rtc::LogMessage::LogTimestamps(flag)

// set whether log to stderr(on android, you may not wish to log to stderr)
// default is true
#define SetLogToStderr(flag) rtc::LogMessage::SetLogToStderr(flag)

#define LOGV LOG(HOBOT_LOG_VERBOSE)
#define LOGD LOG(HOBOT_LOG_DEBUG)
#define LOGI LOG(HOBOT_LOG_INFO)
#define LOGW LOG(HOBOT_LOG_WARN)
#define LOGE LOG(HOBOT_LOG_ERROR)
#define LOGF LOG(HOBOT_LOG_FATAL)

#define LOGV_T(tag) LOG_TAG(rtc::HOBOT_LOG_VERBOSE, tag)
#define LOGD_T(tag) LOG_TAG(rtc::HOBOT_LOG_DEBUG, tag)
#define LOGI_T(tag) LOG_TAG(rtc::HOBOT_LOG_INFO, tag)
#define LOGW_T(tag) LOG_TAG(rtc::HOBOT_LOG_WARN, tag)
#define LOGE_T(tag) LOG_TAG(rtc::HOBOT_LOG_ERROR, tag)
#define LOGF_T(tag) LOG_TAG(rtc::HOBOT_LOG_FATAL, tag)


#endif //HOBOTLOG_HOBOTLOG_HPP
