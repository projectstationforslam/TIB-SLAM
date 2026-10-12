/**
* This file is part of TIB-SLAM
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* TIB-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* TIB-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with TIB-SLAM.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>
#include<numeric>

#include<opencv2/core/core.hpp>
#include<opencv2/imgproc.hpp>

#include<System.h>
#include<TimingBreakdown.h>

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

void ApplyDeterministicDegradation(cv::Mat &rgb, cv::Mat &depth,
                                   const string &mode, int frameIndex)
{
    if(mode.empty() || mode == "none") return;
    if(mode == "blur")
        cv::GaussianBlur(rgb, rgb, cv::Size(9, 9), 2.0, 2.0);
    else if(mode == "mixed")
        cv::GaussianBlur(rgb, rgb, cv::Size(5, 5), 1.2, 1.2);
    if(mode == "lowlight" || mode == "mixed")
    {
        cv::Mat dark;
        rgb.convertTo(dark, rgb.type(), mode == "mixed" ? 0.50 : 0.30, 0.0);
        rgb = dark;
    }
    if(mode == "depthdrop" || mode == "mixed")
    {
        const int block = 16;
        for(int y = 0; y < depth.rows; y += block)
        {
            for(int x = 0; x < depth.cols; x += block)
            {
                const unsigned hash = static_cast<unsigned>(
                    x * 73856093u ^ y * 19349663u ^ frameIndex * 83492791u);
                const bool drop = mode == "mixed" ?
                    (hash % 20u < 3u) : (hash % 10u < 3u);
                if(drop)
                {
                    const cv::Rect region(x, y,
                        std::min(block, depth.cols - x),
                        std::min(block, depth.rows - y));
                    depth(region).setTo(cv::Scalar::all(0));
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    if(argc < 5 || argc > 7)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association [no_viewer] [none|blur|lowlight|depthdrop|mixed]" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    const bool useViewer = argc < 6 || string(argv[5]) != "no_viewer";
    const string degradation = argc == 7 ? string(argv[6]) : "none";
    if(degradation != "none" && degradation != "blur" &&
       degradation != "lowlight" && degradation != "depthdrop" &&
       degradation != "mixed")
    {
        cerr << "Unknown degradation mode: " << degradation << endl;
        return 1;
    }
    TIB_SLAM::System SLAM(argv[1],argv[2],TIB_SLAM::System::RGBD,useViewer);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;
    const auto sequenceLoopStart = std::chrono::steady_clock::now();
    double inputAndPreprocessSeconds = 0.0;
    double scheduledPacingSeconds = 0.0;
    for(int ni=0; ni<nImages; ni++)
    {
        const auto frameInputStart = std::chrono::steady_clock::now();
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

        ApplyDeterministicDegradation(imRGB, imD, degradation, ni);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif
        inputAndPreprocessSeconds += std::chrono::duration<double>(
            t1 - frameInputStart).count();

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB,imD,tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
        {
            scheduledPacingSeconds += T - ttrack;
            usleep((T-ttrack)*1e6);
        }
    }
    const auto sequenceLoopEnd = std::chrono::steady_clock::now();

    // Persist online latency before shutdown.  Shutdown can flush the
    // asynchronous mapper and export a map, and is intentionally excluded
    // from these per-frame end-to-end TrackRGBD measurements.
    ofstream timing("FrameTiming.csv", ios::out | ios::trunc);
    timing << "frame_id,timestamp,track_ms,within_33_333_ms\n";
    for(int ni=0; ni<nImages; ++ni)
        timing << ni << ',' << setprecision(15) << vTimestamps[ni] << ','
               << setprecision(9) << 1000.0 * vTimesTrack[ni] << ','
               << (vTimesTrack[ni] <= 1.0 / 30.0 ? 1 : 0) << '\n';
    timing.close();

    // Stop all threads
    const auto shutdownStart = std::chrono::steady_clock::now();
    SLAM.Shutdown();
    const auto shutdownEnd = std::chrono::steady_clock::now();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;
    const auto quantile = [&vTimesTrack, nImages](double probability) {
        const double position = probability * static_cast<double>(nImages - 1);
        const size_t lower = static_cast<size_t>(floor(position));
        const size_t upper = static_cast<size_t>(ceil(position));
        const double fraction = position - static_cast<double>(lower);
        return (1.0 - fraction) * vTimesTrack[lower] +
               fraction * vTimesTrack[upper];
    };
    const int overBudget = static_cast<int>(count_if(
        vTimesTrack.begin(), vTimesTrack.end(),
        [](float seconds) { return seconds > 1.0f / 30.0f; }));
    cout << "p90 tracking time: " << quantile(0.90) << endl;
    cout << "p95 tracking time: " << quantile(0.95) << endl;
    cout << "p99 tracking time: " << quantile(0.99) << endl;
    cout << "frames over 33.333 ms: " << overBudget << "/" << nImages
         << " (" << 100.0 * overBudget / max(1, nImages) << "%)" << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");  

    const double trackingComputeSeconds = std::accumulate(
        vTimesTrack.begin(), vTimesTrack.end(), 0.0);
    const double sequenceLoopSeconds = std::chrono::duration<double>(
        sequenceLoopEnd - sequenceLoopStart).count();
    const double shutdownSeconds = std::chrono::duration<double>(
        shutdownEnd - shutdownStart).count();
    ofstream runtime("RuntimeBreakdown.json", ios::out | ios::trunc);
    runtime << "{\n"
            << "  \"input_frames\": " << nImages << ",\n"
            << "  \"input_and_preprocess_s\": " << inputAndPreprocessSeconds << ",\n"
            << "  \"tracking_compute_s\": " << trackingComputeSeconds << ",\n"
            << "  \"scheduled_pacing_s\": " << scheduledPacingSeconds << ",\n"
            << "  \"sequence_loop_wall_s\": " << sequenceLoopSeconds << ",\n"
            << "  \"shutdown_flush_s\": " << shutdownSeconds << ",\n"
            << "  \"tracking_compute_fps\": "
            << nImages / std::max(1e-9, trackingComputeSeconds) << ",\n"
            << "  \"online_sequence_fps\": "
            << nImages / std::max(1e-9, sequenceLoopSeconds) << "\n"
            << "}\n";
    runtime.close();

    TIB_SLAM::DumpTimingBreakdown("StageTiming.json");

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
            
            

        }
    }
}
