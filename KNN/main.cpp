#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <utility>
#include <queue>
#include <cmath>
#include <stdexcept>

// Generate one feature vector (length d)
void generatePoint(std::vector<double>& point,
                   int d,
                   std::mt19937& gen,
                   std::uniform_real_distribution<double>& dis)
{
    point.clear();
    point.reserve(d);

    for (int j = 0; j < d; ++j) {
        point.push_back(dis(gen));
    }
}

// Generate dataset. If test==false, append class label at the end (index d).
void generatePoints(std::vector<std::vector<double>>& points,
                    int n,
                    int d,
                    bool test)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::uniform_int_distribution<int> cls(0, 1);

    points.clear();
    points.reserve(n);

    for (int i = 0; i < n; ++i) {
        std::vector<double> temp;
        temp.reserve(test ? d : (d + 1));

        generatePoint(temp, d, gen, dis);

        if (!test) {
            temp.push_back(static_cast<double>(cls(gen))); // label stored at index d
        }

        points.push_back(std::move(temp));
    }
}

// Euclidean distance over the first d features (ignores label at index d)
double calcDistEuclidean(const std::vector<double>& p1,
                         const std::vector<double>& p2,
                         int d)
{
    double sum = 0.0;
    for (int i = 0; i < d; ++i) {
        const double diff = p1[i] - p2[i];
        sum += diff * diff; // faster than pow(diff,2)
    }
    return std::sqrt(sum);
}

using Neighbor = std::pair<double, int>; // (distance, index)

// Predict label for one test point using max-heap of size k
int predictOne(const std::vector<std::vector<double>>& train,
               const std::vector<double>& test,
               int k,
               int d)
{
    if (k <= 0) throw std::invalid_argument("k must be > 0");
    if ((int)train.size() < k) throw std::invalid_argument("k cannot be larger than number of training points");

    // max-heap by distance (largest distance among current k neighbors on top)
    auto cmp = [](const Neighbor& a, const Neighbor& b) {
        return a.first < b.first;
    };
    std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(cmp)> heap(cmp);

    for (int i = 0; i < (int)train.size(); ++i) {
        const double dist = calcDistEuclidean(test, train[i], d);

        if ((int)heap.size() < k) {
            heap.push({dist, i});
        } else if (dist < heap.top().first) {
            heap.pop();
            heap.push({dist, i});
        }
    }

    // Majority vote (binary labels 0/1 stored at train[idx][d])
    int vote = 0;
    while (!heap.empty()) {
        const int idx = heap.top().second;
        const int label = static_cast<int>(train[idx][d]);
        vote += (label == 0) ? -1 : 1;
        heap.pop();
    }

    return (vote >= 0) ? 1 : 0;
}

// Append predicted class to each test point
void findKNN(const std::vector<std::vector<double>>& train,
             std::vector<std::vector<double>>& test_points,
             int k,
             int d)
{
    for (auto& point : test_points) {
        const int cls = predictOne(train, point, k, d);
        point.push_back(static_cast<double>(cls));
    }
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <n_train> <d> <n_test> <k>\n";
        return 1;
    }

    const int n_train = std::stoi(argv[1]);
    const int d       = std::stoi(argv[2]);
    const int n_test  = std::stoi(argv[3]);
    const int k       = std::stoi(argv[4]);

    std::vector<std::vector<double>> train;
    generatePoints(train, n_train, d, false);

    std::vector<std::vector<double>> test;
    generatePoints(test, n_test, d, true);

    findKNN(train, test, k, d);

    // Print results (training + label, test + predicted label)
    std::cout << "TRAIN (features + label):\n";
    for (const auto& row : train) {
        for (double v : row) std::cout << v << " ";
        std::cout << "\n";
    }

    std::cout << "\nTEST (features + predicted label):\n";
    for (const auto& row : test) {
        for (double v : row) std::cout << v << " ";
        std::cout << "\n";
    }

    return 0;
}

