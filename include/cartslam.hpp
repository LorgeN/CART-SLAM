#ifndef CARTSLAM_HPP
#define CARTSLAM_HPP

#include "datasource.hpp"

namespace cart {
class System {
   public:
    System(DataSource* dataSource);
    ~System();
    void run();

   private:
    DataSource* dataSource;
};
}  // namespace cart

#endif  // CARTSLAM_HPP