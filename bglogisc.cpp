#include <iostream>
#include <string>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <armadillo>
#include <cmath>

using namespace boost::math;
using std::cout;
using std::endl;
using arma::max;
using arma::abs;
using arma::round;

arma::mat logreg(arma::mat y, arma::mat x) {
    arma::mat xstddev = stddev(x);
    // arma::mat xmean = mean(x);
    // x.each_row() -= xmean;
    x.each_row() /= xstddev;
    // add a col of all ones
    int m = x.n_rows;
    arma::mat allOne(m, 1, arma::fill::ones);
    x.insert_cols(0, allOne);
    int n = x.n_cols;

    double alpha = 1.0/m;

    arma::mat b(n, 1, arma::fill::zeros);
    arma::mat v = exp(-x * b);
    arma::mat h = 1 / (1 + v);
    arma::mat J = -(y.t() * log(h) + (1-y).t() * log(1-h));
    arma::mat derivJ = x.t() * (h-y);

    double derivThresh = 0.0000001;
    double bThresh = 0.001;
    while(1) {
        arma::mat newb = b - alpha * derivJ;
        if(max(max(abs(b-newb))) < bThresh) break;

        v = exp(-x * newb);
        h = 1 / (1 + v);

        arma::mat newderivJ = x.t() * (h-y);
        if(max(max(newderivJ - derivJ)) < derivThresh) break;

        arma::mat newJ = -(y.t() * log(h) + (1-y).t() * log(1-h));
        if(newJ(0) >= J(0)) alpha /= 2.0;

        b = newb;
        J = newJ;
        derivJ = newderivJ;
    }

    arma::mat w = pow(h, 2) % v; //Schur product
    arma::sp_mat wmat(diagmat(w));
    arma::mat hess = x.t() * wmat * x;
    arma::mat hessInv = arma::pinv(hess);
    arma::mat seMat = arma::sqrt(hessInv.diag());
    arma::mat zscore = b/seMat;
    arma::mat res = arma::join_rows(b, seMat);

    return res;
}


int main(int argc, char const* argv[])
{
    arma::arma_rng::set_seed_random();
    int nr = 5000;
    int ncx = 2;
    arma::mat x(nr, ncx, arma::fill::randn);
    // arma::mat y = arma::randi<arma::mat>(nr, 1, arma::distr_param(0, 1));
    arma::mat b(ncx, 1, arma::fill::randn);
    arma::mat v = exp(-x * b);
    arma::mat h = 1 / (1 + v);
    arma::mat y = arma::round(h);
    y(arma::span(0, nr/2), 0) = arma::randi<arma::mat>((nr/2)+1, 1, arma::distr_param(0, 1));

    int ntests = 13;
    arma::mat res;
    for (int i=0; i<ntests; i++)
    {
        res = logreg(y, x);
    }
    res.print("res........");
    return 0;
}
