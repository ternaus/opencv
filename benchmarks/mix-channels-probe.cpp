#include <opencv2/core.hpp>
#if defined(__aarch64__)
#include <arm_neon.h>
using ByteVector = uint8x16_t;
#else
#include <tmmintrin.h>
using ByteVector = __m128i;
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using Byte = unsigned char;

template<typename T>
void byPixel(const cv::Mat& src, cv::Mat& dst, const std::vector<int>& order)
{
    const int channels = static_cast<int>(order.size());
    const bool dense = src.isContinuous() && dst.isContinuous();
    const int rows = dense ? 1 : src.rows;
    const int cols = dense ? src.rows * src.cols : src.cols;
    for (int row = 0; row < rows; ++row)
    {
        const T* in = src.ptr<T>(row);
        T* out = dst.ptr<T>(row);
        for (int pixel = 0; pixel < cols; ++pixel, in += channels, out += channels)
            for (int channel = 0; channel < channels; ++channel)
                out[channel] = in[order[channel]];
    }
}

ByteVector loadBytes(const Byte* ptr)
{
#if defined(__aarch64__)
    return vld1q_u8(ptr);
#else
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
#endif
}

void storeBytes(Byte* ptr, ByteVector value)
{
#if defined(__aarch64__)
    vst1q_u8(ptr, value);
#else
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), value);
#endif
}

ByteVector combineBytes(ByteVector a, ByteVector b)
{
#if defined(__aarch64__)
    return vorrq_u8(a, b);
#else
    return _mm_or_si128(a, b);
#endif
}

struct Lookup
{
    int offset;
    int vectors;
    ByteVector mask;
#if !defined(__aarch64__)
    ByteVector perRegister[4];

    void prepare()
    {
        for (int index = 0; index < vectors; ++index)
        {
            const auto selection = _mm_sub_epi8(mask, _mm_set1_epi8(static_cast<char>(index * 16)));
            perRegister[index] = _mm_or_si128(selection, _mm_cmpgt_epi8(selection, _mm_set1_epi8(15)));
        }
    }
#endif
};

struct Store
{
    int first;
    int end;
};

struct Shuffle
{
    int blockBytes;
    int tableWidth = 1;
    bool singleLookup = true;
    std::vector<Lookup> lookups;
    std::vector<Store> stores;
};

Shuffle planShuffle(int itemBytes, const std::vector<int>& order)
{
    const int pixelBytes = static_cast<int>(order.size()) * itemBytes;
    Shuffle plan;
    plan.blockBytes = std::lcm(pixelBytes, 16);
    const int blockVectors = plan.blockBytes / 16;
    plan.stores.reserve(blockVectors);
    plan.lookups.reserve(blockVectors);
    for (int out = 0; out < plan.blockBytes; out += 16)
    {
        std::array<int, 16> indices;
        for (int lane = 0; lane < 16; ++lane)
        {
            const int target = out + lane;
            indices[lane] = (target / pixelBytes) * pixelBytes
                            + order[(target / itemBytes) % order.size()] * itemBytes
                            + target % itemBytes;
        }
        const int first = static_cast<int>(plan.lookups.size());
        for (int start = *std::min_element(indices.begin(), indices.end()) / 16 * 16;
             start < plan.blockBytes; )
        {
            std::array<Byte, 16> mask;
            mask.fill(255);
            int end = start;
            int next = plan.blockBytes;
            for (int lane = 0; lane < 16; ++lane)
            {
                const int position = indices[lane];
                if (position < start)
                    continue;
                if (position >= start + 64)
                    next = std::min(next, position / 16 * 16);
                else
                {
                    mask[lane] = static_cast<Byte>(position - start);
                    end = std::max(end, position);
                }
            }
            const int vectors = (end - start) / 16 + 1;
            plan.lookups.push_back({start, vectors, loadBytes(mask.data())});
            plan.tableWidth = std::max(plan.tableWidth, vectors);
            start = next;
        }
        const int end = static_cast<int>(plan.lookups.size());
        plan.stores.push_back({first, end});
        plan.singleLookup = plan.singleLookup && (end == first + 1);
    }
    return plan;
}

template<int vectors>
ByteVector lookup(const Byte* in, const Lookup& op)
{
#if defined(__aarch64__)
    if constexpr (vectors == 1)
        return vqtbl1q_u8(vld1q_u8(in), op.mask);
    if constexpr (vectors == 2)
        return vqtbl2q_u8(vld1q_u8_x2(in), op.mask);
    if constexpr (vectors == 3)
        return vqtbl3q_u8(vld1q_u8_x3(in), op.mask);
    if constexpr (vectors == 4)
        return vqtbl4q_u8(vld1q_u8_x4(in), op.mask);
#else
    auto value = _mm_shuffle_epi8(loadBytes(in), op.perRegister[0]);
    for (int index = 1; index < vectors; ++index)
        value = _mm_or_si128(value, _mm_shuffle_epi8(loadBytes(in + index * 16), op.perRegister[index]));
    return value;
#endif
}

template<int vectors>
void shuffledRow(const Byte* in, Byte* out, int bytes, const Shuffle& plan)
{
    if constexpr (vectors == 1)
    {
        if (plan.blockBytes == 16)
        {
            const Lookup& op = plan.lookups[0];
            for (int pos = 0; pos < bytes; pos += 16)
                storeBytes(out + pos, lookup<1>(in + pos, op));
            return;
        }
    }
    for (int pos = 0; pos < bytes; pos += plan.blockBytes)
    {
        for (size_t step = 0; step < plan.stores.size(); ++step)
        {
            ByteVector result;
            if constexpr (vectors > 0)
            {
                const Lookup& op = plan.lookups[step];
                result = lookup<vectors>(in + pos + op.offset, op);
            }
            else
            {
                #if defined(__aarch64__)
                result = vdupq_n_u8(0);
#else
                result = _mm_setzero_si128();
#endif
                const Store store = plan.stores[step];
                for (int k = store.first; k < store.end; ++k)
                {
                    const Lookup& op = plan.lookups[k];
                    const Byte* ptr = in + pos + op.offset;
                    switch (op.vectors)
                    {
                    case 1: result = combineBytes(result, lookup<1>(ptr, op)); break;
                    case 2: result = combineBytes(result, lookup<2>(ptr, op)); break;
                    case 3: result = combineBytes(result, lookup<3>(ptr, op)); break;
                    case 4: result = combineBytes(result, lookup<4>(ptr, op)); break;
                    }
                }
            }
            storeBytes(out + pos + step * 16, result);
        }
    }
}

void byteShuffle(const cv::Mat& src, cv::Mat& dst, const std::vector<int>& order)
{
    const int itemBytes = static_cast<int>(src.elemSize1());
    const int pixelBytes = static_cast<int>(src.elemSize());
    Shuffle plan = planShuffle(itemBytes, order);
    if (plan.singleLookup)
        for (Lookup& op : plan.lookups)
        {
            const int start = std::min(op.offset, plan.blockBytes - plan.tableWidth * 16);
            #if defined(__aarch64__)
            auto mask = vaddq_u8(op.mask, vdupq_n_u8(static_cast<Byte>(op.offset - start)));
#else
            auto mask = _mm_add_epi8(op.mask, _mm_set1_epi8(static_cast<char>(op.offset - start)));
#endif
            op.offset = start;
            op.mask = mask;
            op.vectors = plan.tableWidth;
        }
#if !defined(__aarch64__)
    for (Lookup& op : plan.lookups)
        op.prepare();
#endif
    const bool dense = src.isContinuous() && dst.isContinuous();
    const int rows = dense ? 1 : src.rows;
    const int rowBytes = (dense ? src.rows * src.cols : src.cols) * pixelBytes;
    const int bulk = rowBytes / plan.blockBytes * plan.blockBytes;
    const int width = plan.singleLookup ? plan.tableWidth : 0;
    for (int row = 0; row < rows; ++row)
    {
        const Byte* in = src.ptr<Byte>(row);
        Byte* out = dst.ptr<Byte>(row);
        switch (width)
        {
        case 1: shuffledRow<1>(in, out, bulk, plan); break;
        case 2: shuffledRow<2>(in, out, bulk, plan); break;
        case 3: shuffledRow<3>(in, out, bulk, plan); break;
        case 4: shuffledRow<4>(in, out, bulk, plan); break;
        default: shuffledRow<0>(in, out, bulk, plan); break;
        }
        for (int pos = bulk; pos < rowBytes; pos += pixelBytes)
            for (size_t ch = 0; ch < order.size(); ++ch)
                std::memcpy(out + pos + ch * itemBytes, in + pos + order[ch] * itemBytes, itemBytes);
    }
}

template<typename Function>
double timed(Function function, int iterations)
{
    const auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < iterations; ++rep)
        function();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / iterations;
}

int main(int argc, char** argv)
{
    cv::setNumThreads(1);
    const bool parityOnly = argc > 1 && std::strcmp(argv[1], "--parity") == 0;
    const bool mixOnly = argc > 1 && std::strcmp(argv[1], "--mix-only") == 0;
    const int benchmarkSize = argc > 2 ? std::atoi(argv[2]) : 512;
    cv::RNG bytes(137);
    std::mt19937 random(137);
    std::vector<int> channelCounts = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 31, 32, 48, 64, 96, 128};
    if (parityOnly)
    {
        channelCounts.resize(CV_CN_MAX);
        std::iota(channelCounts.begin(), channelCounts.end(), 1);
    }
    else
        std::cout << "size\tchannels\tdtype\tlayout\titerations\tmix_us\tby_pixel_us\tbyte_shuffle_us\n";
    for (int channels : channelCounts)
        for (int depth : {CV_8U, CV_16F, CV_32F})
            for (bool roi : {false, true})
            {
                const int size = parityOnly ? 33 : benchmarkSize;
                int type = CV_MAKETYPE(depth, channels);
                cv::Mat srcParent(size, size, type);
                cv::Mat dstParent(srcParent.size(), type), referenceParent(srcParent.size(), type);
                cv::Mat buffer(size, size * static_cast<int>(srcParent.elemSize()), CV_8U, srcParent.data, srcParent.step);
                bytes.fill(buffer, cv::RNG::UNIFORM, 0, 256);
                cv::Rect rect(roi ? 1 : 0, 0, roi ? size - 3 : size, size);
                const cv::Mat src = srcParent(rect);
                cv::Mat dst = dstParent(rect), reference = referenceParent(rect);
                std::vector<int> order(channels), fromTo;
                std::iota(order.begin(), order.end(), 0);
                std::shuffle(order.begin(), order.end(), random);
                if (std::is_sorted(order.begin(), order.end()))
                    std::reverse(order.begin(), order.end());
                for (int c = 0; c < channels; ++c)
                {
                    fromTo.push_back(order[c]);
                    fromTo.push_back(c);
                }
                const auto mix = [&]{ cv::mixChannels(&src, 1, &reference, 1, fromTo.data(), order.size()); };
                const auto scalar = [&]{
                    if (depth == CV_8U) byPixel<Byte>(src, dst, order);
                    else if (depth == CV_16F) byPixel<unsigned short>(src, dst, order);
                    else byPixel<unsigned>(src, dst, order);
                };
                const auto neon = [&]{ byteShuffle(src, dst, order); };
                const auto exact = [&]{
                    const size_t rowBytes = src.elemSize() * src.cols;
                    for (int row = 0; row < src.rows; ++row)
                        if (std::memcmp(dst.ptr(row), reference.ptr(row), rowBytes) != 0)
                            return false;
                    return true;
                };
                mix();
                scalar();
                if (!exact()) return 1;
                if (!mixOnly)
                {
                    neon();
                    if (!exact()) return 2;
                }
                if (parityOnly) continue;
                const int iterations = std::max(8, std::min(140, static_cast<int>(8000.0 / timed(mix, 3))));
                std::array<double, 5> generic, pixels, vector;
                for (int rep = 0; rep < 5; ++rep)
                    if (rep % 2 == 0)
                    {
                        generic[rep] = timed(mix, iterations);
                        pixels[rep] = mixOnly ? 0 : timed(scalar, iterations);
                        vector[rep] = mixOnly ? 0 : timed(neon, iterations);
                    }
                    else
                    {
                        vector[rep] = mixOnly ? 0 : timed(neon, iterations);
                        pixels[rep] = mixOnly ? 0 : timed(scalar, iterations);
                        generic[rep] = timed(mix, iterations);
                    }
                std::sort(generic.begin(), generic.end());
                std::sort(pixels.begin(), pixels.end());
                std::sort(vector.begin(), vector.end());
                std::cout << size << '\t' << channels << '\t'
                          << (depth == CV_8U ? "uint8" : depth == CV_16F ? "float16" : "float32")
                          << '\t' << (roi ? "roi" : "dense") << '\t' << iterations
                          << '\t' << generic[2] << '\t' << pixels[2] << '\t' << vector[2] << std::endl;
            }
    if (parityOnly)
        std::cout << "Bitwise parity across all " << CV_CN_MAX << " channel counts, 3 depths and 2 layouts\n";
}
