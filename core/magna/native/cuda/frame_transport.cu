
extern "C" __global__ void sparse_frames(
        const unsigned long long* columns,
        const unsigned long long* base,
        const unsigned long long* offsets,
        const unsigned int* indices,
        unsigned long long* output,
        unsigned long long shots, unsigned int width) {
    unsigned long long i = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long count = shots * width;
    if (i >= count) return;
    unsigned long long shot = i / width;
    unsigned int word = (unsigned int)(i % width);
    unsigned long long value = base[word];
    for (unsigned long long j = offsets[shot]; j < offsets[shot + 1]; ++j)
        value ^= columns[(unsigned long long)indices[j] * width + word];
    output[i] = value;
}
