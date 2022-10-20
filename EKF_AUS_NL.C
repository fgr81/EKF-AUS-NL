/*
  Copyright (C) 2017 L. Palatella, F. Grasso
  
  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/. 
  
*/



 /* 
  * @file EKF_AUS_NL.C
  * @author L.Palatella & F. Grasso
  * @date September 2017

    EKF_AUS algorithm with nonlinear corrections, for details 
    please read:

    A. Trevisan, L. Palatella, On the Kalman Filter 
    error covariance collapse into the unstable subspace, 
    Nonlin. Processes in Geophys. 18, 243-250 (2011).

    L. Palatella, A. Trevisan, Interaction of Lyapunov vectors in the
    formulation of the nonlinear extension of the Kalman filter,
    Phys. Rev. E 91, 042905 (2015)
 */

#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

using namespace std;
using namespace Eigen;

#include "GrahamSchmidt.C"

typedef MatrixXd (*FP)(MatrixXd&);

class EKF_AUS
{
private:
  long int n,m; // n degrees of freedom, m
                // number of perturbation, if m=n => EKF standard
  long int p; // number of measurements
  long int ml; // number of vectors to be nonlinearly combined
  double inflation, add_inflation;
  double lin_factor;
  static const double alphabar = 1.7320508;
  double H_lin_factor;

public:

  bool alternatore;

  EKF_AUS()
  {
    inflation=1.;
    add_inflation = 0.;
    lin_factor = H_lin_factor = 1.e-1;
    ml = 0;
  };

  EKF_AUS(long nn, long mm, long pp) : m(mm), n(nn), p(pp) { 
    ml = 0;
    lin_factor = H_lin_factor = 1.e-1;
    inflation=1.;
    add_inflation = 0.;
  }
  EKF_AUS(long nn, long mm, long pp, long mlml) : m(mm), n(nn), p(pp), ml(mlml) {
    lin_factor = H_lin_factor = 1.e-1;
    inflation=1.;
    add_inflation = 0.;
  }

  ~EKF_AUS(){};

  long P() { return p;}
  long N() { return n;}
  long linM() { return m;}
  long NonlinM() { return ml;}
  long HalfNumberNonLinPert(){ return ml * (ml +1)/2; }  
  long NumberNonLinPert(){ return 2 * ml * (ml +1)/2; }  
  long TotNumberPert(){ return m + 2 * ml * (ml +1)/2; }

  void Lfactor(double x){ lin_factor = x; }

  void Init(long nn, long mm, long pp) 
  {
    n = nn;
    m = mm; 
    p = pp;
  }

  void Init(long nn, long mm, long pp, long mmll) 
  {
    n = nn;
    m = mm; 
    p = pp;
    ml = mmll;
  }

  /**
   *
   * @brief the core routine that performs the data assimilation
   * @param measure: the measure MatrixXd with p rows and oner column 
   * @param NonLinH: the funtcion pointer to the nonlinear H function
   * @param R: the diagonal terms in the covariance measure matrix, it is a px1 MatrixXd
   * @param xf: the N0x1 MatrixXd with the state of the system, at output it becomes the analysis state
   * @param Xf: the  N0 x TotNumberPert() MatrixXd with the column vectors with the perturbations \
   at forecast time. On output the columns become the Xa vectors.
   */   
  inline void Assimilate(MatrixXd& measure, FP NonLinH, MatrixXd& R, MatrixXd& xf, MatrixXd& Xf);

  /**
   *
   *@brief the routine that set the number of model error variables 
   *@param istart: the first index, included
   *@param iend: the last index, included
   *@param error: the MatrixXd vector with the sigma of the model error variables
   *@param Xa: the  N0 x TotNumberPert() MatrixXd with the column vectors with the perturbations 
   */
  inline void SetModelErrorVariable(long istart, long iend, MatrixXd error, MatrixXd& Xa);
  
  inline MatrixXd ComputeHEf(FP NonLinH, MatrixXd Ef, double LinearScale, MatrixXd& xf);
    
  inline void Inflation(double in){ inflation = in;}
  
  /**
   *@brief the routine to set the moltiplicative inflation factor \
   ( 1 means no inflation as default), if needed  
   */
  inline double Inflation(){ return inflation;}
  inline void AddInflation(double in){ add_inflation = in;}
  
  /**
   *@brief the routine to set the additive inflation, if needed  
   */
  inline double AddInflation(){ return add_inflation;}

  inline MatrixXd PrepareForEvolution(MatrixXd&, MatrixXd&, MatrixXd&);
  inline MatrixXd PrepareForAnalysis(MatrixXd&, MatrixXd&, MatrixXd&);
}; 


MatrixXd EKF_AUS::ComputeHEf(FP NonLinH, MatrixXd Ef, double LinearScale, MatrixXd& xf)
{
  MatrixXd hef, xfp(xf.rows(),1), hefcol;

  double invLinearScale = 1. / LinearScale;

  hef.resize( p, Ef.cols() );
  hefcol.resize( p , 1 );

  for(long j=0; j<Ef.cols(); j++)
    {
      xfp.col(0) = xf.col(0) + LinearScale * Ef.col(j);
      hefcol = invLinearScale * ( NonLinH(xfp) - NonLinH(xf) ) ;
     
      for(long i=0; i<p; i++) hef(i,j) = hefcol(i,0);
    }
  return hef;
}


/**
   *
   * @brief the core routine that performs the data assimilation
   * @param measure: the measure MatrixXd with p rows and oner column 
   * @param NonLinH: the funtcion pointer to the nonlinear H function
   * @param R: the diagonal terms in the covariance measure matrix, it is a px1 MatrixXd
   * @param xf: the N0x1 MatrixXd with the state of the system, at output it becomes the analysis state
   * @param Xf: the  N0 x TotNumberPert() MatrixXd with the column vectors with the perturbations \
   at forecast time. On output the columns become the Xa vectors.
   */
void EKF_AUS::Assimilate(MatrixXd& measure, FP NonLinH, MatrixXd& R,\
			 MatrixXd& xf, MatrixXd& Xf)
{

  MatrixXd anom;

  try{
    anom = measure - NonLinH(xf); 
  }
  catch (int ercode){
    cout << "Problems in the dimension of measurement vector or in the measurement operator\n" << endl << endl;
    throw int(123);
  }

  this->p = anom.rows();
  this->n = xf.rows();

  alog << "---------------------------------------\n";
  alog << "Starting Assimilation.......\n\n";

  //  alog << "p= " << p << " m=" << m << " n=" << n << endl;  

  MatrixXd K;

  alog << "p = " << p << endl;
  alog << "total m = " << Xf.cols() << endl;
  alog << "N0 = " << n << endl;

  alog << "Orthonormalizing.....\n\n" ; cout.flush();

  MatrixXd Ef = gramsh(Xf); 

  MatrixXd ef = Xf.transpose() * Ef; 
  MatrixXd gamma1 = ef.transpose() *ef;
  MatrixXd gamma2;


  gamma2.resize(Xf.cols(),Xf.cols());

  alog << "Computing  H*Ef...\n\n";

  MatrixXd HEf = ComputeHEf(NonLinH, Ef, 0.1, xf); 	  
  MatrixXd ToBeInv, Inv, check, I, CT;

  //  ToBeInv = R + HEf * gamma1 * HEf.transpose();

  cout << HEf.rows() << " x " << HEf.cols() << endl;
  cout << gamma1.rows() << " x " << gamma1.cols() << endl;

  MatrixXd tmp1, Kanom, tmp3, tmp2;

  ToBeInv = HEf * gamma1 * HEf.transpose();
 
  for(long i=0; i<ToBeInv.rows(); i++)
    ToBeInv(i,i) += R(i,0);

  try{
    Inv = ToBeInv.inverse(); // matrix inversion
  }
  catch(int inversione_flag){
    cout << "EKF_AUS_NL::Assimilate: \nProblems in the matrix inversione, stop now, sorry.\n\n";
    throw int(123);
  }

  /*
  cout << "Check on the inverse....\n\n";

  check = Inv*ToBeInv;

  for(long i=0; i<ToBeInv.rows(); i++) check(i,i) -= 1.;
  
  cout << "norma errore sull'inversa .... " << check.norm() << endl; */

  tmp1 = Inv * anom;

  tmp2 = HEf.transpose() * tmp1;

  Kanom = Ef * gamma1 * tmp2;

  //  K = Ef * gamma1 * HEf.transpose() * Inv; 
  // Eq.(10) of Nonlin. Proc. Geophys.  18, 243-250 (2011).

  MatrixXd Ea;
  MatrixXd xa;

  xa = xf + Kanom; 

  gamma2 = gamma1 - gamma1 * HEf.transpose() * Inv * HEf * gamma1;

  alog << "Diagonalizing via singular values (more stable)....." << endl;

  JacobiSVD<Eigen::MatrixXd> svd(gamma2, ComputeThinU | ComputeThinV);

  cout << "The singular values are:" << endl << svd.singularValues() << endl;

  VectorXd l, ev = svd.singularValues(); // ev = eig.eigenvalues();

  cout << endl;

  l.resize( Xf.cols() );

  long int i,j;


  for(i=0;i<Xf.cols();i++)
    {
      l(i) = sqrt( ev(i) ) * inflation + add_inflation;
    } 

  alog << "EIGEN: ";
  for(i=0; i< Xf.cols(); i++) alog << l(i) << " ";
  alog << endl;

  MatrixXd eigensorted = svd.matrixU(), eigenasc;

  Ea = Ef * eigensorted; 

  for(i=0; i<Xf.cols() ;i++)
    for(j=0;j<n;j++)
      {
	Xf(j,i) = l(i)*Ea(j,i);
      }  

  xf = xa; 
  
  alog << "Assimilation succesfully terminated \n\n\n";
  alog << "---------------------------------------\n";
  alog.flush();
}


inline MatrixXd EKF_AUS::PrepareForEvolution(MatrixXd& xa, MatrixXd& Ea, MatrixXd& Gmunu)
{
  MatrixXd outV(N(), TotNumberPert()), EaG(N(), TotNumberPert());
  long k;



  for(long i=0; i< linM() + HalfNumberNonLinPert(); i++) 
    for(long j=0; j<N(); j++) 
      EaG(j,i) = Ea(j,i) * Gmunu(j,0);

  for(long i=0; i<linM() + HalfNumberNonLinPert(); i++) 
    {
      for(long j=0; j<xa.rows(); j++)
	outV(j,i) = xa(j,0) + lin_factor * EaG(j,i);
    }    

  k = linM() + HalfNumberNonLinPert();

  for(long j=0; j<NonlinM() ; j++)
    for(long i=0; i<= j; i++)
      {
	cout << k << " " << i << " " << j << endl; cout.flush();
	
	for(long l=0; l<xa.rows(); l++)
	      outV(l,k) = xa(l,0) +  0.5 * ( EaG(l,i) + EaG(l,j) );

	k++;
      }
  
  return outV;
}

inline MatrixXd EKF_AUS::PrepareForAnalysis(MatrixXd& xf, MatrixXd& Evoluted, MatrixXd& Gmunu)
{
  MatrixXd outXf(N(),  linM() + HalfNumberNonLinPert() );
  long k=0;
  double invfactor = 1. / lin_factor, fac = alphabar /2.; 

  cout << "invfact = " << invfactor << endl;

  for(long i=0; i<linM() + HalfNumberNonLinPert(); i++)
    for(long l=0; l<xf.rows(); l++)
      outXf(l,i) = invfactor * ( Evoluted(l,i) - xf(l,0) );


  k = linM() + HalfNumberNonLinPert();
 
  for(long j=0; j<NonlinM() ; j++)
    for(long i=0; i<= j; i++)
      {
	for(long l=0; l<xf.rows(); l++)
	  {
	      outXf(l,k - HalfNumberNonLinPert() ) +=\
		fac * ( Evoluted(l,k) - xf(l,0) - 0.5 * ( outXf(l,i) + outXf(l,j) ) ) ;
	  }

	k++;
      }  

  for(long i=0; i< linM() + HalfNumberNonLinPert(); i++) 
    for(long j=0; j<N(); j++) 
      outXf(j,i) = outXf(j,i) / Gmunu(j,0);

  return outXf;
}

inline void EKF_AUS::SetModelErrorVariable(long istart, long iend, MatrixXd error, MatrixXd& Xa)
{
  long nme = iend - istart + 1;

  cout << "There are " << nme << " model error variables\n\n";

  if(nme != error.rows())
    {
      cout << "The error matrix is not well dimensioned: rows = " << error.rows()<< endl;
      cout << "istart = " << istart << " iend = " << iend << endl;
      throw int(435);
    }

  if(nme > linM())
    {
      cout << "Too many model error variables, increase m in the EKF-AUS constructor\n\n";
      throw int(535);
    }

  /* No model error */

  for(long j=0; j<linM() - nme; j++)
    {
      for(long i = istart; i <= iend; i++)
	Xa(i,j) = 0.; 
    }

  for(long j=linM(); j< Xa.cols(); j++)
    {
      for(long i = istart; i <= iend; i++)
	Xa(i,j) = 0.; 
    }

  /* Model error */
 
  for(long j=0; j < nme; j++)
    {
      for(long i = 0; i < Xa.rows(); i++) Xa(i,j + linM() - nme) = 0.; 
      Xa(istart + j, j + linM() - nme) = error(j,0);
    }  
}


