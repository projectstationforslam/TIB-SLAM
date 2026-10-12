#include "InformationBudgetScheduler.h"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>

using Scheduler = TIB_SLAM::InformationBudgetScheduler;

int main()
{
    Scheduler::Vector6d spectrum;
    spectrum << 1e-8, 1e-6, 1e-3, 1.0, 10.0, 100.0;
    const Scheduler::Vector6d weights = Scheduler::SoftWeakness(spectrum, 1e-4, 0.35);
    if((weights.array() < 0.0).any() || (weights.array() > 1.0).any() ||
       weights(0) <= weights(5))
        return 1;
    const Scheduler::Vector6d hard = Scheduler::HardWeakness(spectrum, 1e-4);
    if(hard(0) != 1.0 || hard(1) != 1.0 || hard(2) != 1.0 ||
       hard(3) != 0.0 || hard(4) != 0.0 || hard(5) != 0.0 ||
       ((hard.array() != 0.0) && (hard.array() != 1.0)).any())
        return 5;

    Scheduler::Matrix6d hessian = spectrum.asDiagonal();
    std::vector<Scheduler::Candidate,
        Eigen::aligned_allocator<Scheduler::Candidate>> candidates(3);
    candidates[0].information(0,0) = 10.0;
    candidates[0].cost = 1.0;
    candidates[0].externalIndex = 10;
    candidates[1].information(1,1) = 10.0;
    candidates[1].cost = 2.0;
    candidates[1].externalIndex = 11;
    candidates[2].information(5,5) = 1000.0;
    candidates[2].cost = 1.0;
    candidates[2].externalIndex = 12;
    Scheduler::Config config;
    config.enabled = true;
    config.temporalTransport = false;
    config.factorBudget = 2.0;
    config.maximumFactors = 3;
    config.minimumMarginalGain = 0.0;
    Scheduler::State state;
    const Scheduler::Result first = Scheduler::Select(hessian,
        Eigen::Matrix4d::Identity(), 1, candidates, config, &state);
    if(!first.valid || first.totalCost > config.factorBudget + 1e-9 ||
       first.selected.empty() || first.selected.front() != 10 ||
       first.totalGain < -1e-10)
        return 2;

    Scheduler::Matrix6d identity = Scheduler::Matrix6d::Identity();
    Eigen::Matrix4d previous = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d current = Eigen::Matrix4d::Identity();
    current(0,3) = 0.2;
    const Scheduler::Matrix6d transported = Scheduler::TransportInformation(
        identity, previous, current);
    Eigen::SelfAdjointEigenSolver<Scheduler::Matrix6d> solver(transported);
    if(!transported.allFinite() ||
       (transported - transported.transpose()).norm() > 1e-10 ||
       solver.info() != Eigen::Success || solver.eigenvalues().minCoeff() <= 0.0)
        return 3;

    Scheduler::Matrix6d rejected = Scheduler::Matrix6d::Identity() -
        first.informationProjector;
    if((first.informationProjector * rejected).norm() > 1e-8)
        return 4;
    std::cout << "InformationBudgetSchedulerTest passed\n";
    return 0;
}
