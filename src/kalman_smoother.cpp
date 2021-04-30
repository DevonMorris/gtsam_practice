// This is like using a jack hammer to drive in a nail
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

class MeasurementFactor: public gtsam::NoiseModelFactor1<gtsam::Vector> {
public:
    gtsam::Vector m_measured;
    gtsam::Matrix m_C;

    MeasurementFactor(gtsam::Key j, gtsam::Vector measured, gtsam::Matrix C, const gtsam::SharedNoiseModel& model)
        : NoiseModelFactor1<gtsam::Vector>(model, j), m_measured(measured), m_C(C) {}

    gtsam::Vector evaluateError(const gtsam::Vector& x,
                   boost::optional<gtsam::Matrix&> H = boost::none) const override {
        if (H) (*H) = m_C;
        return m_C * x - m_measured;
    }
};

class MotionFactor: public gtsam::NoiseModelFactor2<gtsam::Vector, gtsam::Vector> {
public:
    gtsam::Matrix m_A;
    gtsam::Vector m_Bu;

    MotionFactor(gtsam::Key j, gtsam::Key k, gtsam::Matrix A, gtsam::Vector Bu, const gtsam::SharedNoiseModel& model)
        : NoiseModelFactor2<gtsam::Vector, gtsam::Vector>(model, j, k), m_A(A), m_Bu(Bu) {}

    gtsam::Vector evaluateError(const gtsam::Vector& xp, const gtsam::Vector& x,
                   boost::optional<gtsam::Matrix&> Hp = boost::none,
                   boost::optional<gtsam::Matrix&> H = boost::none) const override {
        if (Hp) (*Hp) = -m_A;
        if (H) (*H) = gtsam::Matrix::Identity(x.rows(), x.rows());
        return x - (m_A * xp + m_Bu);
    }
};

int main(int argc, char* argv[]) {
    auto A = (gtsam::Matrix(2,2) << 1, 0.1, 0, 1).finished();
    auto B = (gtsam::Matrix(2,1) << 0, 1).finished();
    auto C = (gtsam::Matrix(1,2) << 1, 0).finished();
    auto u = 0.1;

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    auto prior_cov =gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(2) << 0.1, 0.1).finished());
    auto factor_x0 = gtsam::PriorFactor<gtsam::Vector>(gtsam::Symbol('x',0), (gtsam::Vector(2) << 1.0, 1.0).finished(), prior_cov);
    initial.insert(gtsam::Symbol('x',0), (gtsam::Vector(2) << 0, 1).finished());
    graph.push_back(factor_x0);

    auto motion_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(2) << 0.1, 0.1).finished());
    auto factor_x1_x0 = MotionFactor(gtsam::Symbol('x',0), gtsam::Symbol('x',1), A, B*u, motion_noise);
    initial.insert(gtsam::Symbol('x',1), (gtsam::Vector(2) << 1.1, 1.1).finished());
    graph.push_back(factor_x1_x0);

    auto measurement_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(1) << 0.1).finished());
    auto z1 = (gtsam::Vector(1) << 1.5).finished();
    MeasurementFactor factor_z1_x1(gtsam::Symbol('x',1), z1, C, measurement_noise);
    graph.push_back(factor_z1_x1);

    gtsam::Values result = gtsam::LevenbergMarquardtOptimizer(graph, initial).optimize();
    result.print();
}
