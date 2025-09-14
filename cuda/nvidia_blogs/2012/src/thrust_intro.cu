#include <iostream>
#include <string>

#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>

int main() {
  /*
    // Example

    // generate 32 M random numbers serially
    thrust::host_vector<int> h_vec(32 << 20);
    std::generate(h_vec.begin(), h_vec.end(), std::rand);

    // transfer data to the device
    thrust::device_vector<int> d_vec = h_vec;

    // sort data on the device
    thrust::sort(d_vec.begin(), d_vec.end());

    // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
  */

  // input data on the host
  const std::string data = "aaabbbbbcddeeeeeeeeeff";

  const auto N = data.size();

  // copy input data to the device
  thrust::device_vector<char> input(data.begin(), data.end());

  // allocate storage for output data and run lengths
  thrust::device_vector<char> output(N);
  thrust::device_vector<int> lengths(N);

  // print the initial data
  std::cout << "input data:" << std::endl;
  thrust::copy(input.begin(), input.end(),
               std::ostream_iterator<char>(std::cout, ""));
  std::cout << std::endl << std::endl;

  // compute run lengths
  const auto num_runs =
      thrust::reduce_by_key(input.begin(), input.end(),
                            thrust::constant_iterator<int>(1), output.begin(),
                            lengths.begin())
          .first -
      output.begin(); // compute output size

  // print the output
  std::cout << "run-length encoded output:" << std::endl;
  for (size_t i = 0; i < num_runs; i++)
    std::cout << "(" << output[i] << "," << lengths[i] << ")";
  std::cout << std::endl;

  return 0;
}
