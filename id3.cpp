// Josh Morris
// CSCI 4350
// Dr. Phillips
// id3.cpp
// Nov 16, 2017

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace std;

struct node{
    float splitValue;
    int attrIndex;
    int classification;
    bool terminal;
    struct node* leftChild; //less than split
    struct node* rightChild; // greater than equal split 

    node() {}

    node(float sV, int aI, int cl, bool t) {
        splitValue = sV;
        attrIndex = aI;
        classification = cl;
        terminal = t;
        leftChild = NULL;
        rightChild = NULL;
    }

    ~node() {
        delete leftChild;
        delete rightChild;
    }
};

const int DEFAULT_CLASS = 0;

void loadData(string filePath, vector<vector<double> >& data);
void printData(vector<vector<double> > data);
bool sameClass(const vector<double>& classification); 
int majClass(const vector<double>& classes);            // return majority class
bool attrEq(const vector<vector<double> > &data); // all attr =
double inf(vector<double> classes);
vector<vector<int> > sort_attributes(vector<vector<double> > data);
void chooseAttr(const vector<vector<double> >& data, const vector<vector<int> >& indices, int& atr, double& splitValue, int& splitInd);
struct node* trainDecisionTree(vector<vector<double> >& data);
int classify(const vector<double>& item, node* tree);

int main(int argc, char* argv[]) {
    int numFeat;
    node* tree;
    int correct = 0;    
    string trainPath;
    string testPath;
    vector<vector<double> > data; 
    vector<vector<int> > indices;


    // check input
    if (argc != 4) {
        cout << "usage: ./id3 numFeature trainingPath testPath" << endl;
        exit(-1);
    }

    numFeat = atoi(argv[1]);
    trainPath = argv[2];
    testPath = argv[3];

    loadData(trainPath, data);

    tree = trainDecisionTree(data);

    // load test data
    data.clear();
    loadData(testPath, data);

    for (int i = 0; i < data.back().size(); ++i) {
        vector<double> attributes;
        int hyp;
        double attVal;

        attributes.clear();
        for (int k = 0; k < data.size() - 1; ++k) {
            attributes.push_back(data[k][i]);
        }

        hyp = classify(attributes, tree);

        if (hyp == data.back()[i]) {
            correct++;
        }
    }

    cout << correct << endl;

    delete tree;

    return 0;
}

void loadData(string filePath, vector<vector<double> >& data) {
    fstream fs(filePath);
    string line;
    double value;
    
    getline(fs,line);
    stringstream parsed(line);
    
    // Prep vectors...
    while (!parsed.eof()) {
        parsed >> value;
        data.push_back(std::vector<double>());
    }
    
    while (!fs.eof()) {
        stringstream parsed(line);
        for (int i = 0; i < data.size(); i++) {
            parsed >> value;
            data[i].push_back(value);
        }
        getline(fs,line);
    }

    fs.close();
}

void printData(vector<vector<double> > data) {
    for (int j = 0; j < data[0].size(); j++) {
        for (int i = 0; i < data.size(); i++) {
            cout << setw(4) << data[i][j] << " ";
        }
        cout << endl;
    }
}

struct node* trainDecisionTree(vector<vector<double> >& data) {
    struct node* tree;
    int majorityClass;
    int atr; // index of atr to split on
    int splitInd; // index where split occurs in train data
    double splitValue; // value to split on
    vector<vector<int> > indices;
    vector<vector<double> > data_split0(data.size());
    vector<vector<double> > data_split1(data.size());

    // if all examples have the same classification 
    if (sameClass(data.back())) {
        // return class
        return new node(-1, -1, data.back()[0], true); 
    }
    // else if all attributes equal
    else if (attrEq(data)) {
        majorityClass = majClass(data.back());
        return new node(-1, -1, majorityClass, true);
    }
    else {
        // sort data
        indices = sort_attributes(data);
  
        // best-attr = choose-attribute(examples)
        chooseAttr(data, indices, atr, splitValue, splitInd);
  
        // sort data based on best attr
        for (int j = 0; j < data.size(); j++){
            for (int i = 0; i < splitInd; i++) {
                data_split0[j].push_back(data[j][indices[atr][i]]);
            }
  
            for(int i = splitInd; i < (data[0].size()); ++i) {
                data_split1[j].push_back(data[j][indices[atr][i]]);
            }
        }
  
        // tree = a new decision tree with root test best-attr
        tree = new node(splitValue, atr, -1, false);
  
        // create children
        tree->leftChild = trainDecisionTree(data_split0);
        tree->rightChild = trainDecisionTree(data_split1);
    }

    //  return (tree)
    return tree;
}


// function to check if all data has the same class 
bool sameClass(const vector<double>& classification) {
    for (auto i = classification.begin(); i != (classification.end() - 1); ++i) {
        if (*i != *(i + 1)) {
            return false;
        }
    }

    return true;
}


// function to find the majority classification
int majClass(const vector<double>& classes) {
    int majClass = classes.front();
    int majClassCount = 1;
    int curClass = classes.front();
    int curClassCount = 1;

    for (auto i = classes.begin() + 1; i != classes.end(); ++i) {
        if (curClass != *i) {
            curClass = *i;
            curClassCount = 1;
        }
        else {
            curClassCount++;
            if (curClassCount > majClassCount) {
                majClassCount = curClassCount;
                majClass = curClass;
            }
        }
    } //end for (auto i = classes...)

    return majClass;
}


//function to determine if all attr are equal
bool attrEq(const vector<vector<double> > &data) {
    for (auto i = data.begin(); i != data.end() - 1; ++i ) {
        for (auto j = (*i).begin(); j != (*i).end() - 1; ++j) {
            if (*j != *(j+1)) {
                return false;
            }
        }
    }

    return true;
}


void chooseAttr(const vector<vector<double> >& data, const vector<vector<int> >& indices, int& atr, double& splitValue, int& splitInd) {
    vector<double> split_0;
    vector<double> split_1;
    double information;
    atr = (data.back())[0];
    double tempGain;
    double maxGain;
    double encoding;

    // calculate information
    information = inf(data.back());

    // for each attr find max gain
    for (int k = 0; k < data.size() - 1; k++) {
        for (int i = 1; i < data[0].size(); i++) {
            // if we register a split
            if (data[k][indices[k][i]] != data[k][indices[k][i - 1]]){
                split_0.clear();
                split_1.clear();

                // calculate the encodng of the split
                for (int j = 0; j < i; ++j) {
                    split_0.push_back(data.back()[indices[k][j]]);
                }

                encoding = ((double)(i) / (double)(data[0].size())) * inf(split_0);

                for (int j = i; j <  data[0].size(); ++j) {
                    split_1.push_back(data.back()[indices[k][j]]);
                }

                encoding += (1 - ((double)(i) / (double)(data[0].size()))) * inf(split_1);

                // calculate gain
                tempGain = information - encoding;
                //cout << "gain = " << tempGain << endl;                

                // if gain greater than max gain so far, save info
                if (tempGain > maxGain) {
                    maxGain = tempGain;
                    atr = k;
                    splitInd = i;
                    splitValue = 0.5 * (data[k][indices[k][i]] + data[k][indices[k][i]]);
                }
            }
        }
    }
}

// function to calculate information of dataset
double inf(vector<double> classes) {
    double information = 0;
    double total = classes.size();
    sort(classes.begin(), classes.end());
    double curClass = classes.front();
    double curClassCount = 1; 
    double curProb;

    for (auto i = classes.begin() + 1; i != classes.end(); ++i) {
        if (curClass != *i) {
            curProb = (curClassCount / total);
            if (curProb != 0) {
                information -= curProb * log2(curProb);
            }

            curClass = *i;
            curClassCount = 1;
        }
        else {
            curClassCount++;
        }
    } //end for (auto i = classes...)

    curProb = (curClassCount / total);
    if(curProb != 0) {
        information -= curProb * log2(curProb);
    }

    return information;
}

// Attribute sorting
// Written by Joshua L. Phillips
vector<vector<int> > sort_attributes(vector<vector<double> > data) {
  vector<vector<int> > indices;
  vector<double> *ptr;
  indices.resize(data.size());

  for (int x = 0; x < indices.size(); x++) {
    indices[x].resize(data[x].size());
    iota(indices[x].begin(),indices[x].end(),0);
    ptr = &(data[x]);
    sort(indices[x].begin(),indices[x].end(),
     [&](size_t i, size_t j){ return (*ptr)[i] < (*ptr)[j]; });
  }

  return indices;
}

int classify(const vector<double>& item, node* tree) {

    // cout << "atr = " << tree->attrIndex << endl;
    if(tree->terminal) {
        return tree->classification;
    }
    else if (item[tree->attrIndex] < tree->splitValue){
        return classify(item, tree->leftChild);
    }
    else{
        return classify(item, tree->rightChild);
    }
}
