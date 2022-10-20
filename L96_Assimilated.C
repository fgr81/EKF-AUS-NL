/*
  Copyright (C) 2017 L. Palatella, F. Grasso
  
  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/. 
  
*/


/**
 *
 * @file L96_Assimilated.C
 * @author L.Palatella & F. Grasso
 * @date September 2017
 *
 */

/**
 *
 * @class L96_Assimilated
 * @brief Handle external L96 program and it implements the interface class. 
 * 
 */

class L96_Assimilated : public IAssimilate
{

public:
  L96_Assimilated(){};
  ~L96_Assimilated(){};

private:
  double gaussiano()
  {
    double x,y,r,fac=0;
    while(fac==0)
      {
	x=2*ranf_arr_next()-1;
        y=2*ranf_arr_next()-1;

	r=x*x+y*y;
	if(r<1. && r!=0.)
	  //      if(r<1. && r>1.e-4 )
	  {
	    fac = sqrt(-2.0*log(r)/r);
	    return x*fac;
	  }
      }
  }
  
  static MatrixXd nonlinH(MatrixXd& state)
  {
    /*
      In this example we measure one every 2 state variable.
     */

    MatrixXd misura;
    long k = state.rows();

    misura.resize(k / 2,1);

    for(long i=0; i<misura.rows(); i++)
      misura(i,0) = state(i * 2,0);

    return misura;
  }
  

  int perfect(const char* name)
  {
    MatrixXd truth, analysis, R;
    MatrixXd tmp, measure, noise, perturbazioni, T_A_perturbazioni; 
    FP meas_operator;
    long Assimilationcycles = 1000;

    double sigma0 = 0.1; /* the measurement error */

    SEED = 12342453;
    ranf_start(SEED); /* Initialize the random number generator */
    N0 = dimensions(name);

    cout << "Inizializing the truth n0 = " << N0 << endl;
    truth.resize(N0,1);
    analysis.resize(N0,1);
    

    cout << "Reading the input file " << name << endl; 
    readFile(name, truth);

    cout << "read " << endl << endl;
    
    class EKF_AUS ekf_aus(N0, 14, 20, 3);

    ekf_aus.Lfactor(0.001);
    
    long nc = ekf_aus.TotNumberPert();

    MatrixXd Xa = MatrixXd::Random(N0, ekf_aus.TotNumberPert());
    Xa *= sigma0;

    cout << "num tot pert = " << nc << endl; 

  ifstream sp;
  char nomepert[100];

  alog << "Initializing Xa, \n we have " << ekf_aus.TotNumberPert()  << " perturbations\n\n"; 
  alog.flush();
  
  alog << "Initializing the analysis\n\n"; alog.flush();

  for(long j=0; j<N0; j++)
    {
      analysis(j,0) = truth(j,0) + Xa(j,0) + Xa(j,1);
    }

  MatrixXd gmunu(N0,1);     

  meas_operator = &nonlinH;
  for(long kk = 0; kk<N0; kk++) gmunu(kk,0) = 1.;

  alog << "Defining the R matrix \n\n" << endl;

  R.setZero(ekf_aus.P(),1);

  for(long i=0; i<ekf_aus.P(); i++)
    R(i,0) = sigma0 * sigma0;

  measure = nonlinH(truth);

  for(long i=0; i<ekf_aus.P(); i++) 
    measure(i,0) += sigma0 * gaussiano(); 
  
  alog << "Assimilating...." << endl; alog.flush();

  ekf_aus.Assimilate(measure, meas_operator, R, analysis, Xa);

  VectorXd errore;


  for(long kk = 0; kk<N0; kk++) gmunu(kk,0) = 1.; 
  /* Trivial metric in this case, change if necessary, 
     only diagonal form handled in the present version */


  for(long kk = 0; kk<Assimilationcycles ; kk++)
    {
      perturbazioni = ekf_aus.PrepareForEvolution(analysis,Xa, gmunu);
      
      //  PrintState(perturbazioni.col(0),"Pert0.dat");
      
      errore = analysis - truth;

      alog << "Timestep = " << kk <<  " Analysis Error = " << errore.norm() / sqrt(N0) << endl;
      cout << "Timestep = " << kk <<  " Analysis Error = " << errore.norm() / sqrt(N0) << endl;

      T_A_perturbazioni.resize(N0, perturbazioni.cols()+2); 

      T_A_perturbazioni.col(0) = truth.col(0);
      T_A_perturbazioni.col(1) = analysis.col(0);
        
      for(long i=0; i< perturbazioni.cols(); i++)       
        T_A_perturbazioni.col(i+2) = perturbazioni.col(i);

      evolve(T_A_perturbazioni, 0.125);
      
      truth.col(0) =  T_A_perturbazioni.col(0);
      analysis.col(0) =  T_A_perturbazioni.col(1);

      for(long i=0; i< perturbazioni.cols(); i++)
        perturbazioni.col(i) = T_A_perturbazioni.col(i+2);

      errore = analysis - truth;

      alog << "Timestep = " << kk << " Forecast Error = " << errore.norm() / sqrt(N0) << endl;

      Xa = ekf_aus.PrepareForAnalysis(analysis, perturbazioni, gmunu);
      
      measure = nonlinH(truth);

      for(long i=0; i<ekf_aus.P(); i++) 
          measure(i,0) += sigma0 * gaussiano(); 

      ekf_aus.Assimilate(measure, meas_operator, R, analysis, Xa);
    }
  }
  

  long dimensions(const char* forecast)
  {
    long ci = 0, n0;
    ifstream in(forecast);
    in >> n0;
    in.close();
    return n0;
  }
  
  void readFile(const char* forecast, MatrixXd& xf)
  {
    long ci = 0, n0;
    ifstream in(forecast);
    in >> n0;
    xf.resize(n0,1);
    for(long i=0; i<n0; i++) in >> xf(i,0);
    in.close();
  }

  void writeFile(const char* forecast, MatrixXd& xf)
  {
    long ci = 0;
    ofstream out(forecast);
    out << xf.rows() << endl;
    for(long i=0; i< xf.rows(); i++) out << xf(i,0) << endl;
    out.close();
  }
  

  void evolve(MatrixXd& XX, double time)
  {
    int last_run, first_run, nsteps = time / 0.0125;
    char comando[200];
    MatrixXd colonna;
    
    
    for(long k=0; k<XX.cols(); k++)
      {
	cout << "Evolving column " << k << endl;
	colonna = XX.col(k);
	writeFile("statotemporaneo.dat",colonna);
	sprintf(comando,"./external/my_l96/MyL96 statotemporaneo.dat %lf",time);
	cout << "Executing the command " << comando << endl;
	system(comando);
	readFile("statotemporaneo.dat",colonna);
	XX.col(k) = colonna;
      }
  }
  
};
