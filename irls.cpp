#include <boost/timer.hpp>
#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <armadillo>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;
using namespace arma;
using std::cout;
using std::endl;
using arma::max;
using arma::abs;
using arma::round;

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


arma::mat getz(arma::mat& y, arma::mat& beta, arma::mat& x)
{
    arma::mat z;
    arma::mat tmp = exp(x * beta);
    z = x*beta + y % (pow(1+tmp, 2)/tmp) - 1 - tmp;
    return z;
}


arma::mat getw(arma::mat& beta, arma::mat& x)
{
    arma::mat w;
    arma::mat tmp = exp(x * beta);
    w = tmp/pow(1+tmp, 2);
    return w;
}

void pm(arma::mat x, std::string s)
{
    cout << s << "..........." << endl;
    int n = x.n_rows;
    int k = x.n_cols;
    cout << "nrow: " << n << endl;
    cout << "ncol: " << k << endl;
    int firstrow = 0;
    int firstcol = 0;
    int lastrow = n-1;
    int lastcol = k-1;
    if(n >= 5) {
        lastrow = 5;
    }
    if(k >= 5) {
        lastcol = 5;
    }
    cout << x.submat(firstrow, firstcol, lastrow, lastcol) << endl;
}

void pv(arma::vec v, std::string s)
{
    int n = v.n_elem;
    arma::mat vmat(n, 1);
    vmat.col(0) = v;
    pm(vmat, s);
}

arma::mat logreg(arma::mat y, arma::mat x, bool fitted) {
    int n = x.n_rows;
    arma::mat allOne(n, 1, arma::fill::ones);
    x.insert_cols(0, allOne);
    int k = x.n_cols;

    arma::mat coef = solve(x, y);
    arma::mat w = getw(coef, x);
    arma::mat z = getz(y, coef, x);
    arma::mat xtw(x);
    xtw.each_col() %= w;
    xtw = xtw.t();
    arma::mat J = xtw * x;
    arma::mat coef1 = arma::solve(J, xtw*z);
    arma::mat coefdiff_mat = arma::abs(coef - coef1);
    double coefdiff = coefdiff_mat.max();
    while(coefdiff >= 0.00001) {
        coef = coef1;
        w = getw(coef, x);
        z = getz(y, coef, x);
        xtw = x;
        xtw.each_col() %= w;
        xtw = xtw.t();
        J = xtw * x;
        coef1 = arma::solve(J, xtw*z);
        coefdiff_mat = arma::abs(coef - coef1);
        coefdiff = coefdiff_mat.max();
    }


    arma::mat vcmat = pinv(J); // var-covariance matrix
    arma::vec coefSe = pow(vcmat.diag(), .5);
    arma::mat coefSeMat(coefSe);
    arma::mat res(k, 3);
    res.col(0) = coef;
    res.col(1) = coefSeMat;
    res.col(2) = coef/coefSeMat;

    if(fitted) {
        arma::mat xb = x * coef;
        arma::mat yhat = 1 / (1 + exp(-xb));
        return yhat;
    }
    return res;
}


int main(int argc, char const* argv[])
{
    arma::mat x = readMatrix("/tmp/S.csv");
    arma::mat y = readMatrix("/tmp/y.csv");

    arma::mat res1 = logreg(y, x, true);
    arma::mat res2 = logreg(y, x, false);
    pm(res1, "res1");
    pm(res2, "res2");


    // int ntests = 100;
    // boost::timer t1;
    // for (int i=0; i<ntests; i++)
    // {
    //     cout << "Iteration " << i << endl;
    //     res = logreg(y, x);
    // }
    // std::cout << "Time passed: " << t1.elapsed() << std::endl;
    return 0;
}
