#include "utils/data.hpp"

namespace cart {
void DataContainer::insertData(module_result_pair_t data) {
    LOG4CXX_INFO(this->getLogger(), "Inserting data with key " << std::quoted(data.first));
    boost::lock_guard<boost::mutex> lock(this->dataMutex);
    this->data.insert(data);
    LOG4CXX_DEBUG(this->getLogger(), "Notifying all");
    this->dataCondition.notify_all();
}

bool DataContainer::hasData(const std::string key) {
    boost::unique_lock<boost::mutex> lock(this->dataMutex);
    return this->data.count(key);
}
}  // namespace cart