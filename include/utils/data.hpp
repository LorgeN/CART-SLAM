#pragma once

#include <log4cxx/logger.h>

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/thread.hpp>
#include <boost/thread/future.hpp>

namespace cart {
typedef std::pair<std::string, boost::shared_ptr<void>> module_result_pair_t;
typedef std::optional<module_result_pair_t> module_result_t;

class DataContainer {
   public:
    DataContainer() = default;

    virtual ~DataContainer() = default;

    virtual log4cxx::LoggerPtr getLogger() = 0;

    virtual boost::asio::thread_pool& getThreadPool() = 0;

    bool hasData(const std::string key);

    template <typename T>
    boost::shared_ptr<T> getData(const std::string key) {
        boost::unique_lock<boost::mutex> lock(this->dataMutex);
        if (!this->data.count(key)) {
            throw std::invalid_argument("Could not find key");
        }

        return boost::static_pointer_cast<T>(this->data[key]);
    }

    template <typename T>
    boost::future<boost::shared_ptr<T>> getDataAsync(const std::string key) {
        boost::packaged_task<boost::shared_ptr<T>> task([this, key] {
            boost::unique_lock<boost::mutex> lock(this->dataMutex);
            while (!this->data.count(key)) {
                LOG4CXX_DEBUG(this->getLogger(), "Waiting for key " << key << " to be available");
                this->dataCondition.wait(lock);
            }

            LOG4CXX_DEBUG(this->getLogger(), "Key " << key << " is now available");
            return boost::static_pointer_cast<T>(this->data[key]);
        });

        auto future = task.get_future();
        boost::asio::post(this->getThreadPool(), boost::move(task));
        return future;
    }

    boost::future<void> waitForData(const std::vector<std::string> keys) {
        boost::packaged_task<void> task([this, keys] {
            boost::unique_lock<boost::mutex> lock(this->dataMutex);
            for (const auto& key : keys) {
                LOG4CXX_DEBUG(this->getLogger(), "Waiting for key " << key << " to be available");

                while (!this->data.count(key)) {
                    // If we ever have to wait more than 3 seconds for new data to be inserted, the run
                    // is most likely over and something has failed somewhere
                    const boost::system_time timeout = boost::get_system_time() + boost::posix_time::seconds(3);

                    if (!this->dataCondition.timed_wait(lock, timeout)) {
                        throw std::runtime_error("Timeout waiting for data key \"" + key + "\"");
                    }
                }
            }
        });

        auto future = task.get_future();
        boost::asio::post(this->getThreadPool(), boost::move(task));
        return future;
    }

   protected:
    void insertData(module_result_pair_t data);

   private:
    std::map<std::string, boost::shared_ptr<void>> data;
    boost::mutex dataMutex;
    boost::condition_variable dataCondition;
};
}  // namespace cart