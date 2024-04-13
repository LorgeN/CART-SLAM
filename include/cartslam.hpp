#pragma once

#define CARTSLAM_WORKER_THREADS 16

#define MODULE_NO_RETURN_VALUE std::nullopt
#define MODULE_RETURN(key, value) std::make_optional(std::make_pair(key, value))

#ifdef CARTSLAM_IMAGE_MAKE_GRAYSCALE
#define CARTSLAM_RUN_RETENTION 10
#define CARTSLAM_CONCURRENT_RUN_LIMIT 6
#define CARTSLAM_IMAGE_CHANNELS 1
#else
#define CARTSLAM_RUN_RETENTION 6
#define CARTSLAM_CONCURRENT_RUN_LIMIT 3
#define CARTSLAM_IMAGE_CHANNELS 3
#endif

#include <log4cxx/logger.h>

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/thread.hpp>
#include <boost/thread/future.hpp>
#include <map>
#include <string>

#include "datasource.hpp"
#include "logging.hpp"
#include "opencv2/core/cuda.hpp"
#include "utils/data.hpp"

namespace cart {

class System;  // Allow references

class SystemRunData : public DataContainer {
   public:
    SystemRunData(uint32_t id, boost::weak_ptr<System> system, boost::shared_ptr<DataElement> dataElement) : id(id), system(system), dataElement(dataElement) {}
    ~SystemRunData() = default;

    void markAsComplete();
    bool isComplete();

    boost::shared_ptr<SystemRunData> getRelativeRun(const int8_t offset);

    boost::shared_ptr<DataElement> dataElement;
    const uint32_t id;

    log4cxx::LoggerPtr getLogger() override;
    boost::asio::thread_pool& getThreadPool() override;

    friend class System;

   private:
    bool complete = false;
    boost::weak_ptr<System> system;
};

class SystemModule {
   public:
    SystemModule(const std::string& name) : SystemModule(name, {}){};

    SystemModule(const std::string& name, const std::vector<std::string> requiresData) : name(name), requiresData(requiresData) {
        this->logger = getLogger(name);
    }

    virtual ~SystemModule() = default;
    virtual boost::future<system_data_t> run(System& system, SystemRunData& data) = 0;

    const std::vector<std::string> requiresData;

    // TODO: Add check for missing data dependencies
    // const std::string providesData;

    const std::string name;

   protected:
    log4cxx::LoggerPtr logger;
};

class SyncWrapperSystemModule : public SystemModule {
   public:
    SyncWrapperSystemModule(const std::string& name) : SystemModule(name){};

    SyncWrapperSystemModule(const std::string& name, const std::vector<std::string> requiresData) : SystemModule(name, requiresData){};

    boost::future<system_data_t> run(System& system, SystemRunData& data) override;
    virtual system_data_t runInternal(System& system, SystemRunData& data) = 0;
};

class System : public boost::enable_shared_from_this<System>, public DataContainer {
   public:
    System(boost::shared_ptr<DataSource> dataSource);
    ~System() = default;
    boost::future<void> run();

    template <typename T, typename... Args>
    void addModule(Args... args) {
        this->addModule(boost::make_shared<T>(args...));
    }

    void addModule(boost::shared_ptr<SystemModule> module);

    boost::shared_ptr<SystemRunData> startNewRun(cv::cuda::Stream& stream);
    boost::shared_ptr<SystemRunData> getRunById(const uint32_t index);
    uint8_t getActiveRunCount();

    log4cxx::LoggerPtr getLogger() override;
    boost::asio::thread_pool& getThreadPool() override;

    void insertGlobalData(const std::string key, boost::shared_ptr<void> data);
    void insertGlobalData(system_data_pair_t data);

   private:
    log4cxx::LoggerPtr logger;
    boost::asio::thread_pool threadPool = boost::asio::thread_pool(CARTSLAM_WORKER_THREADS);

    uint32_t runId = 0;
    boost::shared_ptr<DataSource> dataSource;
    std::vector<boost::shared_ptr<SystemModule>> modules;
    std::vector<boost::shared_ptr<SystemRunData>> runs;
    boost::mutex runMutex;
    boost::condition_variable runCondition;
};
}  // namespace cart
