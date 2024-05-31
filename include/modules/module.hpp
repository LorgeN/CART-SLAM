#pragma once

#include <log4cxx/logger.h>

#include "../logging.hpp"
#include "../utils/data.hpp"

namespace cart {

class System;  // Allow references
class SystemRunData;

struct module_dependency_t {
    std::string name;
    int8_t runOffset;
    bool optional;

    module_dependency_t(const std::string& name, const int runOffset, const bool optional) : name(name), runOffset(runOffset), optional(optional){};

    module_dependency_t(const std::string& name, const int runOffset) : module_dependency_t(name, runOffset, false){};

    module_dependency_t(const std::string& name) : module_dependency_t(name, 0, false){};

    module_dependency_t() : module_dependency_t("", 0, false){};
};

class SystemModule {
   public:
    SystemModule(const std::string& name) : name(name) {
        this->logger = getLogger(name);
    };

    virtual ~SystemModule() = default;
    virtual boost::future<system_data_t> run(System& system, SystemRunData& data) = 0;

    const std::vector<module_dependency_t> getRequiredData() const;

    const std::vector<std::string> getProvidedData() const;

    const std::string name;

   protected:
    log4cxx::LoggerPtr logger;
    std::vector<module_dependency_t> requiresData;
    std::vector<std::string> providesData;
};

class SyncWrapperSystemModule : public SystemModule {
   public:
    SyncWrapperSystemModule(const std::string& name) : SystemModule(name){};

    boost::future<system_data_t> run(System& system, SystemRunData& data) override;

    virtual system_data_t runInternal(System& system, SystemRunData& data) = 0;
};
}  // namespace cart