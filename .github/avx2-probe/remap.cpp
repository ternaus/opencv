#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <functional>
#include <vector>

namespace
{
std::array<double, 3> medianMicros(const std::array<std::function<void()>, 3>& cases, bool singleThread)
{
    const auto elapsed = [&](const std::function<void()>& run, int iterations)
    {
        const auto startWall = std::chrono::steady_clock::now();
        const std::clock_t startCpu = std::clock();
        for (int i = 0; i < iterations; ++i)
            run();
        if (singleThread)
            return (std::clock() - startCpu) * 1e6 / CLOCKS_PER_SEC / iterations;
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - startWall).count() / iterations;
    };
    std::array<int, 3> loops;
    std::array<std::vector<double>, 3> values;
    for (int i = 0; i < 3; ++i)
    {
        elapsed(cases[i], 5);
        loops[i] = std::max(8, std::min(160, (int)(25000. / std::max(1., elapsed(cases[i], 3)))));
    }
    for (int repeat = 0; repeat < 9; ++repeat)
    {
        for (int index = 0; index < 3; ++index)
        {
            const int i = (repeat + (repeat % 2 == 0 ? index : 2 - index)) % 3;
            values[i].push_back(elapsed(cases[i], loops[i]));
        }
    }
    std::array<double, 3> medians;
    for (int i = 0; i < 3; ++i)
    {
        std::sort(values[i].begin(), values[i].end());
        medians[i] = values[i][values[i].size() / 2];
    }
    return medians;
}
}

int main()
{
    CV_Assert(cv::checkHardwareSupport(CV_CPU_AVX2) && cv::checkHardwareSupport(CV_CPU_FP16));
    std::printf("# OpenCV=%s features=%s\n", CV_VERSION, cv::getCPUFeaturesLine().c_str());
    std::puts("threads,size,channels,interpolation,map,fp32_us,fp16_us,conversion_us,max_error,mae,max_error_vs_rounded_fp32");
    cv::RNG rng(137);
    for (bool singleThread : {true})
    {
        cv::setNumThreads(singleThread ? 0 : -1);
        const int threads = cv::getNumThreads();
        for (int side : {224, 512})
        {
            const double focal = 120. * side / 224;
            const cv::Matx33d K(focal, 0., side / 2., 0., focal, side / 2., 0., 0., 1.);
            for (int channels : {3})
            {
                cv::Mat samples(side, side, CV_MAKETYPE(CV_32F, channels)), srcHalf, srcFloat;
                rng.fill(samples, cv::RNG::UNIFORM, -1., 1.);
                samples.convertTo(srcHalf, CV_16F);
                srcHalf.convertTo(srcFloat, CV_32F);
                for (int interpolation : {cv::INTER_LINEAR, cv::INTER_CUBIC, cv::INTER_LANCZOS4})
                for (int mapType : {CV_32FC1, CV_16SC2})
                {
                    cv::Mat map1, map2, dstFloat, dstHalf, narrowed, workFloat, workOut;
                    cv::fisheye::initUndistortRectifyMap(K, cv::Vec4d::all(0.), cv::Matx33d::eye(), K,
                                                         srcFloat.size(), mapType, map1, map2);
                    const auto fp32 = [&]()
                    {
                        cv::remap(srcFloat, dstFloat, map1, map2, interpolation);
                    };
                    const auto fp16 = [&]()
                    {
                        cv::remap(srcHalf, dstHalf, map1, map2, interpolation);
                    };
                    const auto convert = [&]()
                    {
                        srcHalf.convertTo(workFloat, CV_32F);
                        cv::remap(workFloat, workOut, map1, map2, interpolation);
                        workOut.convertTo(narrowed, CV_16F);
                    };
                    fp32();
                    fp16();
                    convert();
                    if (side == 224 && channels == 3 && mapType == CV_16SC2 && interpolation == cv::INTER_LINEAR)
                    {
                        cv::Mat fisheyeHalf, fisheyeHalfAsFloat, fixedHalfAsFloat;
                        cv::fisheye::undistortImage(srcHalf, fisheyeHalf, K, cv::Vec4d::all(0.), K);
                        CV_Assert(fisheyeHalf.depth() == CV_16F);
                        fisheyeHalf.convertTo(fisheyeHalfAsFloat, CV_32F);
                        dstHalf.convertTo(fixedHalfAsFloat, CV_32F);
                        CV_Assert(cv::norm(fisheyeHalfAsFloat, fixedHalfAsFloat, cv::NORM_INF) == 0.);
                    }
                    cv::Mat halfAsFloat, roundedRef;
                    dstHalf.convertTo(halfAsFloat, CV_32F);
                    narrowed.convertTo(roundedRef, CV_32F);
                    const double maxError = cv::norm(dstFloat, halfAsFloat, cv::NORM_INF);
                    const double mae = cv::norm(dstFloat, halfAsFloat, cv::NORM_L1) / (dstFloat.total() * channels);
                    const double roundedError = cv::norm(roundedRef, halfAsFloat, cv::NORM_INF);
                    CV_Assert(dstHalf.depth() == CV_16F && roundedError <= 1. / 1024);
                    const auto times = medianMicros({fp32, fp16, convert}, singleThread);
                    std::printf("%d,%d,%d,%d,%s,%.3f,%.3f,%.3f,%.9g,%.9g,%.9g\n",
                                threads, side, channels, interpolation, mapType == CV_32FC1 ? "float" : "fixed",
                                times[0], times[1], times[2], maxError, mae, roundedError);
                    std::fflush(stdout);
                }
            }
        }
    }
    return 0;
}
