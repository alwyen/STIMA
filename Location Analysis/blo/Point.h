#include <Eigen/Dense>
#include <initializer_list>
#include <vector>

namespace blo {

    template<typename T, size_t d>
    class Point {
    private:
    public:
        Eigen::Matrix<T, d, 1> _vec;
        Point(std::vector<T> list)
        {
            for (size_t i = 0; i < list.size(); i++) {
                _vec(i, 0) = list[i];
            }
        }
        const Eigen::Matrix<T, d, 1> get() const {
            return _vec;
        }
        /*T at(size_t i) const {
            assert(i < _vec.rows);
            return _vec[i,0];
        }*/
    };

}