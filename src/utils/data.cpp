#include "utils/data.hpp"

namespace cart {
void DataContainer::insertData(system_data_pair_t data) {
    LOG4CXX_INFO(this->getLogger(), "Inserting data with key " << std::quoted(data.first));
    boost::lock_guard<boost::mutex> lock(this->dataMutex);
    this->data.insert(data);
    this->dataCondition.notify_all();
}

bool DataContainer::hasData(const std::string key) {
    boost::unique_lock<boost::mutex> lock(this->dataMutex);
    return this->data.count(key);
}

boost::future<void> DataContainer::waitForData(const std::vector<std::string> keys) {
    boost::packaged_task<void> task([this, keys] {
        boost::unique_lock<boost::mutex> lock(this->dataMutex);
        for (const auto& key : keys) {
            LOG4CXX_DEBUG(this->getLogger(), "Waiting for key " << key << " to be available");

            while (!this->data.count(key)) {
                // If we ever have to wait more than 3 seconds for new data to be inserted, the run
                // is most likely over and something has failed somewhere
                const boost::system_time timeout = boost::get_system_time() + boost::posix_time::seconds(CARTSLAM_WAIT_FOR_DATA_TIMEOUT);

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
}  // namespace cart