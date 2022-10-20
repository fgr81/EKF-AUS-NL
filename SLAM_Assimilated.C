/*
  Copyright (C) 2017 L. Palatella, F. Grasso
  
  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/. 
  
*/


/**
 *
 * @file SLAM_Assimilated.C
 * @author L.Palatella & F. Grasso
 * @date September 2017
 *
 */

/**
 *
 * @class SLAM_Assimilated
 * @brief Handle external SLAM program and it implements the interface class. 
 * 
 */
class SLAM_Assimilated : public IAssimilate
{
public:
  
  /**
   * Constructor
   */	
  SLAM_Assimilated(){};
  
  /**
   * Destructor
   */
  ~SLAM_Assimilated(){};
  
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

    /* Lo state variables are sorted in this way
       x, y, phi, x1, y1, x2, y2, x3, y3, ......, V, G
    */

    MatrixXd misura;
    long nm, k = state.rows() ;
    double d, angle, dx, dy, ang;
    
    nm = (k - 5);
    
    misura.resize(nm,1);
    
    for(long i=0; i<misura.rows(); i+=2)
      {
	dx = state(i + 3,0) - state(0,0);
	dy = state(i + 4,0) - state(1,0);
	
	d = sqrt(dx*dx + dy*dy);
	
	misura(i,0) = d;
	misura(i+1,0) = atan2(dy,dx) - state(2,0);
      }
    
    return misura;
  }
  
  void InitializeEKF_SLAM(MatrixXd& state, MatrixXd& Xa, MatrixXd& R, MatrixXd& firstMeasure)
  {    
    double r, theta, a, b, c, d;
    long n;
    
    state(0,0) = state(1,0) = state(2,0) = 0.;
    
    for(long i=0; i<firstMeasure.rows(); i+=2)
      {
	r = firstMeasure(i,0);
	theta = firstMeasure(i+1,0);
	
	state(i+3,0) = r * cos(theta + state(2,0));
	state(i+4,0) = r * sin(theta + state(2,0));
	
	for(long j=0; j<Xa.cols(); j++)
	  {
	    a = gaussiano();
	    b = gaussiano();
	    c = gaussiano();
	    d = gaussiano();
	    
	    Xa(i+3,j) = a * sqrt(R(i,0)) * cos(theta + state(2,0)) +	\
	      b * sqrt(R(i+1,0)) * sin(theta + state(2,0)); 
	    Xa(i+3,j) = c * sqrt(R(i,0)) * sin(theta + state(2,0)) +	\
	      d * sqrt(R(i+1,0)) * cos(theta + state(2,0)); 
	  }
      }
  }

  
  int perfect(const char* name)
  {
    MatrixXd truth, analysis, R, Xa;
    MatrixXd tmp, measure, noise, perturbazioni, T_A_perturbazioni; 
    FP meas_operator;
    long Assimilationcycles = 5000;

    SEED = 12342453;
    ranf_start(SEED); /* Initialize the random number generator */
    N0 = dimensions(name);

    cout << "Inizializing the truth n0 = " << N0 << endl;
    truth.resize(N0,1);
    analysis.resize(N0,1);
    startpert.resize(N0,1);

    cout << "Reading the input file " << name << endl; 
    readFile(name, truth);

    cout << "read " << endl << endl;
 
    double VV, GG;
    VV = truth(N0-2);
    GG = truth(N0-1);

    cout << "read V (velocity) = " << VV << " G (steering) = " << GG << endl << endl;

    class EKF_AUS ekf_aus(N0, 6, N0 - 5, 0); /* initializing the EKF_AUS_NL class with N0 degrees of freedom,
					      6 linear Lyapunov vectors, N0-2 number of measurements, 
					      no nonlinear interaction*/
    ekf_aus.AddInflation(1.e-12);
    ekf_aus.Lfactor(0.01);
    long nc = ekf_aus.TotNumberPert();

    Xa.resize(N0,nc);

    double sigmad = 0.20, sigmaA = 3 * ppi / 360.; /* 1 grado sessagesimale */
    
    R.resize(ekf_aus.P(),1);
    
    for(long i=0; i<ekf_aus.P(); i++)
      {
	if(i%2) /* odd: angle */
	  R(i,0) = sigmaA * sigmaA;
	else /* even: distance */
	  R(i,0) = sigmad * sigmad;
      }
    
    VectorXd errore;
    MatrixXd gmunu(N0,1), ModelError(2,1);
    
    ModelError(0,0) = 0.05; /* (m/s) error in the velocity  */
    ModelError(1,0) = 1.*ppi/360.; /* 3 degrees error for the steering angle */
    
    measure = nonlinH(truth);
    
    for(long i=0; i<ekf_aus.P(); i++)
      measure(i,0) += sqrt(R(i,0)) * gaussiano(); 
    
    cout << "Initializing analyis and Xa" << endl;
    
    InitializeEKF_SLAM(analysis, Xa, R, measure);

    ofstream traj("traiettorie.dat");
    char nomepert[100];
    
    meas_operator = &nonlinH;
    
    measure = nonlinH(truth);
    
    for(long kk = 0; kk<N0; kk++) gmunu(kk,0) = 1.;
    
    for(long kk = 0; kk<Assimilationcycles; kk++)
      {
	ekf_aus.SetModelErrorVariable(N0-2, N0-1, ModelError, Xa);
		
	/*	if(kk==0)
	  {
	    truth(N0-2,0) = analysis(N0-2,0) = VV;
	    truth(N0-1,0) = analysis(N0-1,0) = GG;
	  }
	else{
	  analysis(N0-2,0) = VV;
	  analysis(N0-1,0) = GG;
	  
	  truth(N0-2,0) = analysis(N0-2,0) + ModelError(0,0) * gaussiano();
	  truth(N0-1,0) = analysis(N0-1,0) + ModelError(1,0) * gaussiano();
	  }*/
	
	truth(N0-2,0) = VV;
	truth(N0-1,0) = GG;

	analysis(N0-2,0) = VV + ModelError(0,0) * gaussiano();
	analysis(N0-1,0) = GG + ModelError(1,0) * gaussiano();

	
	perturbazioni = ekf_aus.PrepareForEvolution(analysis, Xa, gmunu);	
	errore = analysis - truth;
	
	alog << "Timestep = " << kk <<  " Analysis Error = " << errore.norm() / sqrt(N0) << endl;
	cout << "Timestep = " << kk <<  " Analysis Error = " << errore.norm() / sqrt(N0) << endl;
	
	/*	for(long j=0; j<truth.rows(); j++)
	  alog << truth(j,0) << " " << analysis(j,0) << endl;
	
	  alog << endl;*/
	
	T_A_perturbazioni.resize(N0, perturbazioni.cols()+2); 
	
	T_A_perturbazioni.col(0) = truth.col(0);
	T_A_perturbazioni.col(1) = analysis.col(0);
	
	for(long i=0; i< perturbazioni.cols(); i++)       
	  T_A_perturbazioni.col(i+2) = perturbazioni.col(i);

	cout << "Evolving....." << endl;
	evolve(T_A_perturbazioni, 0.05);
	
	truth.col(0) =  T_A_perturbazioni.col(0);
	analysis.col(0) =  T_A_perturbazioni.col(1);
	
	for(long i=0; i< perturbazioni.cols(); i++)
	  perturbazioni.col(i) = T_A_perturbazioni.col(i+2);
	
	errore = analysis - truth;
	
	alog << "Timestep = " << kk << " Forecast Error = " << errore.norm() / sqrt(N0) << endl;
	
	Xa = ekf_aus.PrepareForAnalysis(analysis, perturbazioni, gmunu);
	
	measure = nonlinH(truth);
	
	for(long i=0; i<ekf_aus.P(); i++) 
	  measure(i,0) += sqrt(R(i,0)) * gaussiano(); 
	
	ekf_aus.Assimilate(measure, meas_operator, R, analysis, Xa);
	
	writeFile("./truth",truth);
	writeFile("./analysis",analysis);

	traj << kk << " " << truth(0,0) << " " << truth(1,0) << " " << truth(2,0);
	traj << " " << analysis(0,0) << " " << analysis(1,0) << " " << analysis(2,0) << endl;

      }

    traj.close();
    alog.close();
    
    return 0;

    /* SONO QUI*/
  }
  
  long dimensions(const char* forecast)
  {
    long ci = 0, n0;
    double x,y,phi,V,G;
    
    ifstream in(forecast);
    
    in >> x;  in >> y; in >> phi;
    in >> V; in >> G;
    
    in >> n0;
    
    in.close();

    return 2*n0 + 5; /* + x,y,phi,V,G; V,G at the end*/
  }
  
  void readFile(const char* forecast, MatrixXd& xf)
  {
    long ci = 0, N;
    double x,y,phi,V,G, xn, yn;
    ifstream in(forecast);
    
    in >> x;  in >> y; in >> phi;
    in >> V; in >> G;
    
    in >> N;
    
    cout << "there are " << N << " landmarks" << endl;
    
    xf.resize(2*N + 5,1);
    
    xf(0,0) = x; xf(1,0) = y; xf(2,0) = phi;
    
    for(long i=0; i<N; i++)
      {
	in >> xn; in >> yn;
	xf(3+2*i,0) = xn;
	xf(4+2*i,0) = yn;
      }
    
    xf(2*N + 3,0) = V;
    xf(2*N + 4,0) = G;
    
    in.close();
  }
  
  void writeFile(const char* forecast, MatrixXd& xf)
  {
    long ci = 0, N;
    
    ofstream out(forecast);
   
    out.setf(ios_base::fixed);

    out << xf(0,0) << endl;
    out << xf(1,0) << endl;
    out << xf(2,0) << endl;
    
    out << xf(xf.rows() - 2,0) << endl; /* V */
    out << xf(xf.rows() - 1,0) << endl; /* G */
    
    N = (xf.rows() - 5)/2;
    
    out << N << endl;
        
    for(long i=0; i<N; i++)
      {
	out << xf(3 + 2*i,0) << " " << xf(4+2*i,0) << endl;
      }
    
    out.close();
  }
  
  void evolve(MatrixXd& XX, double time)
  {
    int last_run, first_run, nsteps = time / 0.025; 
    char comando[200];

    MatrixXd colonna;

    for(long k=0; k<XX.cols(); k++)
      {
	cout << "Evolving column " << k << " ......." <<endl;
	colonna = XX.col(k);

	writeFile("statotemporaneo.dat",colonna);

	sprintf(comando,"external/my_slam/MySlam statotemporaneo.dat %d",nsteps);
	cout << "Executing the command: " << comando << endl;

	system(comando);    ///< system call to evolve the dynamical system state

	cout << "Evoluted column " << k << endl;
	cout.flush();

	readFile("statotemporaneo.dat",colonna);

	XX.col(k) = colonna;

	cout << "Terminated column " << k << endl;
	cout.flush();
      }
  }

};
