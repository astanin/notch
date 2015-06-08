#ifndef NOTCH_DEBUG_HPP
#define NOTCH_DEBUG_HPP

#include <assert.h>
#include <iostream>

std::ostream &operator<<(std::ostream &out, const Array &xs);
std::ostream &operator<<(std::ostream &out, const std::valarray<double> &xs);

#endif /* NOTCH_DEBUG_HPP */
