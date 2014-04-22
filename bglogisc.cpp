#include <iostream>
#include <string>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <armadillo>
#include <cmath>

using namespace boost::math;

arma::mat logreg(arma::mat y, arma::mat x) {
    // arma::mat xstddev = stddev(x);
    // arma::mat xmean = mean(x);
    // x.each_row() -= xmean;
    // x.each_row() /= xstddev;
    // add a col of all ones
    int m = x.n_rows;
    arma::mat allOne(m, 1, arma::fill::ones);
    x.insert_cols(0, allOne);
    int n = x.n_cols;

    double alpha = 1/m;

    arma::mat b(n, 1, arma::fill::zeros);
    arma::mat v = exp(-x * b);
    arma::mat h = 1 / (1 + v);
    arma::mat J = -(y.t() * log(h) + (1-y).t() * log(1-h));
    arma::mat derivJ;
    derivJ = x.t() * (h-y);

    while(1) {
        arma::mat newb = b - alpha * derivJ;
        v = exp(-x * newb);
        h = 1 / (1 + v);
        arma::mat newJ = -(y.t() * log(h) + (1-y).t() * log(1-h));
        // if(newJ(0) >= J(0)) {
        //     std::cout << "newJ >= J" << std::endl;
        // } else {
        //     std::cout << "newJ < J" << std::endl;
        // }
        while(newJ(0) >= J(0)) {
            std::cout << "inner while" << std::endl;
            alpha /= 1.5;
            newb = b - alpha * derivJ;
            v = exp(-x * newb);
            h = 1 / (1 + v);
            newJ = -(y.t() * log(h) + (1-y).t() * log(1-h));
        }
        if(max(max(abs(b) - newb)) < 0.001) {
            break;
        }
        b = newb;
        J = newJ;
        derivJ = x.t() * (h-y);
    }

    arma::mat w = pow(h, 2) * v;
    arma::sp_mat wmat(diagmat(w));
    arma::mat hess = x.t() * wmat * x;
    // arma::mat seMat = sqrt()
    w.print("w.....");
    wmat.print("wmat......");
    hess.print("hess........");

    return J;
}


int main(int argc, char const* argv[])
{
    {
        arma::arma_rng::set_seed_random();
        int nr = 50;
        int ncx = 2;
        arma::mat x(nr, ncx, arma::fill::randn);
        arma::mat y = arma::randi<arma::mat>(nr, 1, arma::distr_param(0, 1));
        logreg(y, x);
        // arma::mat xcol;
        // arma::mat res(ncx, 4);
        // for(int i=0; i<ncx; i++) {
        //     xcol = x(arma::span::all, i);
        //     // res.row(i) = (glmMat(y, xcol, "binomial", "logit")).row(1);
        //     logreg(y, x);
        // }
        // res.print("res..........");
    }
    return 0;
}
