/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 * Description: Common ARMv8 parameters
 * Author: KPL
 * Create: 2020-11-05
 * Notes: NA
 */

#ifndef PARAM_H
#define PARAM_H

#if defined(PARAM_CONF_KP920)
#include "arch_param/param_tsv110.h"
#elif defined(PARAM_CONF_KP920B)
#include "arch_param/param_1630.h"
#elif defined(PARAM_CONF_KP920F_SME)
#include "arch_param/param_1636sme.h"
#elif defined(PARAM_CONF_KP920F_SVE)
#include "arch_param/param_1636sve.h"
#endif // PARAM_CONF

#endif