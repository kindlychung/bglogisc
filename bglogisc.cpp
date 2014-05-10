#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/regex.hpp>
#include <armadillo>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace boost::math;
using std::cout;
using std::endl;
using arma::max;
using arma::abs;
using arma::round;
using namespace arma;

template<typename T>
T readCsv(std::string csvfile)
{
	boost::filesystem::path csvfilePath(csvfile);
	if(csvfilePath.extension() != ".csv") {
		throw std::invalid_argument("This does not seem to be of csv format.");
	}
	T bigm;
	bigm.load(csvfile);
	return bigm;
}

///dec
std::string stripExt(std::string& x)
///edec
{
    std::string xNoExt = "";
    boost::regex fnRegex(R"delim((.*)\..*)delim");
    boost::smatch regexRes;
    if(boost::regex_match(x, regexRes, fnRegex)) {
        xNoExt = regexRes[1];
    }
    return xNoExt;
}

template<typename T>
T readBigm(std::string bmdesFilen)
// void readBigm(std::string bmdesFilen)
{
	boost::filesystem::path csvfilePath(bmdesFilen);
	if(csvfilePath.extension() != ".bmdes")
		throw std::invalid_argument("This does not seem to be of bigmatrix format.");

	// get the content of the descriptor file
	std::ifstream fIn(bmdesFilen);
	std::stringstream sBuffer;
	sBuffer << fIn.rdbuf();
	std::string bmdesFileStr = sBuffer.str();
	if(bmdesFileStr == "") {
		throw "descriptor file not right.";
	}

    boost::regex rcRegex(R"delim(totalRows\s*=\s*(\d+),\s*totalCols\s*=\s*(\d+),.*type = "(\w+)")delim");
    boost::smatch regexRes;
	int nr, nc;
	nr = nc = 0;
	std::string matType = "";
	if(boost::regex_search(bmdesFileStr, regexRes, rcRegex)) {
		// for(std::string i : regexRes) {
		// 	std::cout << i << std::endl;
		// }
		std::istringstream(regexRes[1]) >> nr;
		std::istringstream(regexRes[2]) >> nc;
		matType = regexRes[3];
	} else {
		throw "Something went wrong with regex search.";
	}

	// strip extension, add big.matrix bin file extension
	std::string bmbinFilen = stripExt(bmdesFilen) + ".bmbin";

	T bigm;
	bigm.load(bmbinFilen);
	bigm.reshape(nr, nc);
	return bigm;
}

///dec
arma::Mat<short> readMatrixGen(std::string genfile)
///edec
{
	try {
		return readCsv<arma::Mat<short>>(genfile);
	}
	catch(...) {
		return readBigm<arma::Mat<short>>(genfile);
	}
}


///dec
arma::mat readMatrix(std::string datafile)
///edec
{
	try {
		return readCsv<arma::mat>(datafile);
	}
	catch(...) {
		return readBigm<arma::mat>(datafile);
	}
}

arma::mat logreg(arma::mat y, arma::mat x) {
    arma::mat xstddev = stddev(x);
    arma::mat xmean = mean(x);
    x.each_row() -= xmean;
    x.each_row() /= xstddev;
    // add a col of all ones
    int m = x.n_rows;
    arma::mat allOne(m, 1, arma::fill::ones);
    x.insert_cols(0, allOne);
    int n = x.n_cols;

    double alpha = 2.0/m;

    arma::mat b(n, 1, arma::fill::zeros);
    arma::mat v = exp(-x * b);
    arma::mat h = 1 / (1 + v);
    arma::mat J = -(y.t() * log(h) + (1-y).t() * log(1-h));
    arma::mat derivJ = x.t() * (h-y);

    double derivThresh = 0.000001;
    double bThresh = 0.001;
    while(1) {
        // std::cout << "while loop......" << std::endl;
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
    arma::mat wt = w.t();
    arma::mat xt = x.t();
    xt.each_row() %= wt;
    arma::mat hess = xt * x;
    arma::mat hessInv = arma::pinv(hess);
    arma::mat seMat = arma::sqrt(hessInv.diag());
    arma::mat zscore = b/seMat;
    arma::mat res(n, 3);
    res.col(0) = b;
    res.col(1) = seMat;
    res.col(2) = zscore;

    return res;
}


int main(int argc, char const* argv[])
{
    // arma::arma_rng::set_seed_random();
    // int nr = 5000;
    // int ncx = 5;
    // arma::mat x(nr, ncx, arma::fill::randn);
    // // arma::mat y = arma::randi<arma::mat>(nr, 1, arma::distr_param(0, 1));
    // arma::mat b(ncx, 1, arma::fill::randn);
    // arma::mat v = exp(-x * b);
    // arma::mat h = 1 / (1 + v);
    // arma::mat y = arma::round(h);
    // y(arma::span(0, nr/2), 0) = arma::randi<arma::mat>((nr/2)+1, 1, arma::distr_param(0, 1));

    arma::mat x = readMatrix("x.csv");
    arma::mat y = readMatrix("y.csv");

    int ntests = 1;
    arma::mat res;
    for (int i=0; i<ntests; i++)
    {
        res = logreg(y, x);
    }
    res.print("res........");
    return 0;
}
