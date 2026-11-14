#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry/2d.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using Clock = std::chrono::steady_clock;

static cv::Matx23d matrix(int width, int height, int transform)
{
    switch (transform)
    {
    case 0: return {1., 0., 0.375, 0., 1., -0.625};
    case 1: return cv::getRotationMatrix2D_(cv::Point2f(width / 2.f, height / 2.f), 17., 1.);
    case 2: return cv::getRotationMatrix2D_(cv::Point2f(width / 2.f, height / 2.f), 47., 0.83);
    case 3: return {0.92, 0.31, -7.27, -0.13, 1.07, 9.89};
    case 4: return {1., 0., 17., 0., 1., -13.};
    case 5: return {1., 0., 0., 0., 1., 0.};
    case 6: return {1., 0., width * -2., 0., 1., height * 3.};
    case 7: return {1., 0., 17. + 1e-9, 0., 1., -13.};
    default: return {1., 0., width * 2. + 0.25, 0., 1., height * -3. - 0.75};
    }
}

static void bench()
{
    std::puts("side,channels,transform,microseconds,iterations");
    for (int side : {256, 512, 1024})
    for (int channels : {1, 3, 5})
    for (int transform : {0, 1, 3})
    {
        cv::Mat src(side, side, CV_8UC(channels)), dst(side, side, src.type());
        cv::RNG(137).fill(src, cv::RNG::UNIFORM, 0, 256);
        const auto affine = matrix(side, side, transform);
        const auto run = [&] {
            cv::warpAffine(src, dst, affine, src.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        };
        for (int i = 0; i < 12; ++i)
            run();
        constexpr int iterations = 50;
        std::vector<double> samples;
        for (int repeat = 0; repeat < 5; ++repeat)
        {
            const auto start = Clock::now();
            for (int i = 0; i < iterations; ++i)
                run();
            samples.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count() / iterations);
        }
        std::sort(samples.begin(), samples.end());
        std::printf("%d,%d,%d,%.3f,%d\n", side, channels, transform, samples[2], iterations * 5);
        std::fflush(stdout);
    }
}

static void compareOutputs()
{
    const cv::Scalar fill(-3., 57.25, 131.75, 300.);
    int count = 0;
    for (const auto size : {cv::Size(1, 1), cv::Size(1, 11), cv::Size(2, 13), cv::Size(7, 8),
                            cv::Size(8, 7), cv::Size(9, 8), cv::Size(31, 17), cv::Size(128, 67)})
    for (int channels : {1, 2, 3, 4, 5, 8, 17, CV_CN_MAX})
    for (int method : {cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_AREA, cv::INTER_CUBIC, cv::INTER_LANCZOS4})
    {
        if (channels > 4 && (method == cv::INTER_CUBIC || method == cv::INTER_LANCZOS4))
            continue;
        for (bool roi : {false, true})
        for (int transform = 0; transform < 9; ++transform)
        for (int border : {cv::BORDER_CONSTANT, cv::BORDER_REPLICATE, cv::BORDER_TRANSPARENT})
        for (bool inverse : {false, true})
        for (cv::AlgorithmHint hint : {cv::ALGO_HINT_ACCURATE, cv::ALGO_HINT_APPROX})
        {
            const int type = CV_8UC(channels);
            cv::Mat srcParent(size.height + 4, size.width + 6, type);
            cv::Mat src = roi ? srcParent(cv::Rect(2, 1, size.width, size.height)) : cv::Mat(size, type);
            cv::RNG(137).fill(src, cv::RNG::UNIFORM, 0, 256);
            const cv::Size outputSize(size.width + 3, size.height + 2);
            cv::Mat dstParent(outputSize.height + 3, outputSize.width + 4, type);
            dstParent.reshape(1).setTo(17);
            cv::Mat dst = roi ? dstParent(cv::Rect(1, 2, outputSize.width, outputSize.height))
                              : cv::Mat(outputSize, type);
            dst.reshape(1).setTo(17);
            cv::warpAffine(src, dst, matrix(size.width, size.height, transform), outputSize,
                           method | (inverse ? cv::WARP_INVERSE_MAP : 0), border, fill, hint);
            const cv::Mat& checked = roi ? dstParent : dst;
            std::fwrite(checked.data, 1, checked.total() * checked.elemSize(), stdout);
            ++count;
        }
    }
    std::fprintf(stderr, "Compared %d cases\n", count);
}

int main(int argc, char** argv)
{
    cv::setNumThreads(0);
    std::fprintf(stderr, "OpenCV %s, %s, threads=%d\n", CV_VERSION, cv::getCPUFeaturesLine().c_str(), cv::getNumThreads());
    if (argc == 2 && std::strcmp(argv[1], "--compare") == 0)
        compareOutputs();
    else
        bench();
}
