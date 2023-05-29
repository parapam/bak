#include <iostream>
#include <vector>
#include <string>
#include <hnswlib/hnswlib.h>

using namespace std;

int main()
{
    int dim = 1;             // Dimension of the elements
    int M = 2;               // Tightly connected with internal dimensionality of the data, strongly affects the memory consumption
    int max_elements = 4;    // Maximum number of elements, should be known beforehand
    int ef_construction = 2; // Controls index search speed/build speed tradeoff

    float data[4][1] = {
        {0.372},
        {0.396},
        {0.306},
        {0.354},
    };

    srand(time(NULL));
    std::list<float> recalls;

    for (int i = 0; i < 100000; i++) {
        int random_seed;
        random_seed = rand() % 100;

        // Initing index
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction, random_seed);
        
        // Add data to index
        for (int i = 0; i < max_elements; i++) {
            float *datapoint = data[i * dim];
            alg_hnsw->addPoint((void *)(data + i * dim), i);
        }

        // Query the elements for themselves and measure recall
        float correct = 0;
        
        for (int i = 0; i < max_elements; i++)
        {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
            hnswlib::labeltype label = result.top().second;
            if (data[label] == data[i])
                correct++;
        }

        float recall = correct / max_elements;
        recalls.push_back(recall);
    }
}