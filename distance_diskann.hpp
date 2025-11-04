#pragma once
#include <cstring>
#include "utils.hpp"

enum Metric
{
    L2 = 0,
    INNER_PRODUCT = 1,
    COSINE = 2,
    FAST_L2 = 3
};

template <typename T> class Distance
{
  public:
    Distance(Metric dist_metric) : _distance_metric(dist_metric)
    {
    }

    // distance comparison function
    virtual float compare(const T *a, const T *b, uint32_t length) const = 0;

    // Needed only for COSINE-BYTE and INNER_PRODUCT-BYTE
    virtual float compare(const T *a, const T *b, const float normA, const float normB,
                                            uint32_t length) const;

    // For MIPS, normalization adds an extra dimension to the vectors.
    // This function lets callers know if the normalization process
    // changes the dimension.
    virtual uint32_t post_normalization_dimension(uint32_t orig_dimension) const;

    virtual Metric get_metric() const;

    // This is for efficiency. If no normalization is required, the callers
    // can simply ignore the normalize_data_for_build() function.
    virtual bool preprocessing_required() const;

    // Check the preprocessing_required() function before calling this.
    // Clients can call the function like this:
    //
    //  if (metric->preprocessing_required()){
    //     T* normalized_data_batch;
    //      Split data into batches of batch_size and for each, call:
    //       metric->preprocess_base_points(data_batch, batch_size);
    //
    //  TODO: This does not take into account the case for SSD inner product
    //  where the dimensions change after normalization.
    virtual void preprocess_base_points(T *original_data, const size_t orig_dim,
                                                          const size_t num_points);

    // Invokes normalization for a single vector during search. The scratch space
    // has to be created by the caller keeping track of the fact that
    // normalization might change the dimension of the query vector.
    virtual void preprocess_query(const T *query_vec, const size_t query_dim, T *scratch_query);

    // If an algorithm has a requirement that some data be aligned to a certain
    // boundary it can use this function to indicate that requirement. Currently,
    // we are setting it to 8 because that works well for AVX2. If we have AVX512
    // implementations of distance algos, they might have to set this to 16
    // (depending on how they are implemented)
    virtual size_t get_required_alignment() const;

    // Providing a default implementation for the virtual destructor because we
    // don't expect most metric implementations to need it.
    virtual ~Distance() = default;

  protected:
    Metric _distance_metric;
    size_t _alignment_factor = 8;
};


class DistanceL2Float : public Distance<float>
{
  public:
    DistanceL2Float() : Distance<float>(Metric::L2)
    {
    }

#ifdef _WINDOWS
    virtual float compare(const float *a, const float *b, uint32_t size) const;
#else
    virtual float compare(const float *a, const float *b, uint32_t size) const __attribute__((hot));
#endif
};


template <typename T> Distance<T> *get_distance_function(Metric m);

