// Released into Public Domain by Joshua L. Phillips 2016-10-27
// Don't forget to compile using the C++ 2011 standard
// Example: g++ -std=c++11 sort.cpp -o sort
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <sstream>

// Attribute sorting
std::vector<std::vector<int> > sort_attributes(std::vector<std::vector<double> > data) {
  std::vector<std::vector<int> > indices;
  std::vector<double> *ptr;
  indices.resize(data.size());
  for (int x = 0; x < indices.size(); x++) {
    indices[x].resize(data[x].size());
    std::iota(indices[x].begin(),indices[x].end(),0);
    ptr = &(data[x]);
    std::sort(indices[x].begin(),indices[x].end(),
	 [&](size_t i, size_t j){ return (*ptr)[i] < (*ptr)[j]; });
  }
  return indices;
}

int main(int argc, char* argv[]) {

  std::vector<std::vector<double> > data;
  std::vector<std::vector<int> > indices;
  std::string line;
  double value;
  
  std::getline(std::cin,line);
  std::stringstream parsed(line);
  
  // Prep vectors...
  while (!parsed.eof()) {
    parsed >> value;
    data.push_back(std::vector<double>());
  }
  
  while (!std::cin.eof()) {
    std::stringstream parsed(line);
    for (int i = 0; i < data.size(); i++) {
      parsed >> value;
      data[i].push_back(value);
    }
    std::getline(std::cin,line);
  }

  // This is the important stuff...
  indices = sort_attributes(data);

  // Apply permutation for specific column
  for (int k = 0; k < data.size(); k++) {
    std::cout << "Sorting by " << k << "th column..." << std::endl;
    for (int j = 0; j < data[0].size(); j++) {
      for (int i = 0; i < data.size(); i++) {
	std::cout << data[i][indices[k][j]] << " ";
      }
      std::cout << std::endl;
    }
  }

  return 0;
}
