#pragma once

#include "utils/data.hpp"

#define MODULE_NO_RETURN_VALUE (std::vector<cart::system_data_pair_t>{})
#define MODULE_RETURN(key, value) (std::vector<cart::system_data_pair_t>{std::make_pair(key, value)})
#define MODULE_RETURN_ALL(...) (std::vector<cart::system_data_pair_t>{__VA_ARGS__})
#define MODULE_MAKE_PAIR(key, valueType, ...) std::make_pair(key, boost::make_shared<valueType>(__VA_ARGS__))
#define MODULE_RETURN_SHARED(key, valueType, ...) (std::vector<cart::system_data_pair_t>{std::make_pair(key, boost::make_shared<valueType>(__VA_ARGS__))})