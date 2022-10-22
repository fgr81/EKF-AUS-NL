/*
Copyright (C) 2018 L. Palatella, F. Grasso

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

*/


/**
*
* @file KITTI_Assimilated.C
* @author L.Palatella & F. Grasso
* @date September 2018
*
*/

/**
*
* @class SLAM_Assimilated
* @brief Handle external SLAM program and it implements the interface class.
*
*/
long IAssimilate::LANDMARKS_PER_FRAME = 150;
long IAssimilate::NUMBER_OF_FRAMES = 4543; // todo parametro
long * IAssimilate::new_landmark_flag = new long[LANDMARKS_PER_FRAME]; // '-1' indica che il lm è nuovo, non presente nello stato
long * IAssimilate::meas_landmark_flag = new long[LANDMARKS_PER_FRAME]; //
long * IAssimilate::state_landmark_flag = new long[LANDMARKS_PER_FRAME]; //


class KITTI_Assimilated : public IAssimilate
{
public:

  /**
  * Constructor
  */
  KITTI_Assimilated(){};

  /**
  * Destructor
  */
  ~KITTI_Assimilated(){};

private:

  MatrixXd aux_Xa;
  MatrixXd aux_analysis;
  MatrixXd Xa;
  string mystr_2;   //

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


  double sigma_steer; /* global */

  //MatrixXd getMeasure(long int t_)
  // It works on matches.dat to pick up landmark relative position;
  // It appends in IAssimilate.measure the new landmarks and it updates the olds
  // And returns the number of new landmarks found
  // Return: an array with in first location the NUMBER of new landmark
  MatrixXd getMeasure(long int t_)
  {
    // Intestazione del file matches.dat:
    // t idd L_x L_y R_x R_y disparity vP[0] vP[1] vP[2]
    //
    
    string mystr;
    float L_x, L_y, R_x, R_y, disparity, vP_0, vP_1, vP_2 = 0.0;
    long t, idd;
    MatrixXd new_lm(0,3);
    short prepend = 0;
    double dummy, measured_steering;
    

    //ifstream steer("steering_2011_10_03_0027_dummy.dat");
    // ifstream steer("steering_2011_10_03_0027_sigma.dat");
    ifstream steer("steering_2011_10_03_0027_22ott19.dat");
    //ifstream steer("steering_NO-CROP_2011_10_03_0027_new.dat");
    // ifstream steer("steering_2011_10_03_0027_24giu.dat");

    for(long k = 0; k <= t_; k++)
      {    
	steer >> dummy;
	steer >> measured_steering;
	steer >> sigma_steer;
	//	cout << "t = " << t_ << " steering = " << measured_steering << endl; 
      }
    steer.close();


    if ( mystr_2 != "")
      prepend = 1;
    for ( int k = 0; k < LANDMARKS_PER_FRAME; k++ )
    {
      if ( prepend == 1)
      {
        mystr = mystr_2;
        mystr_2 = "";
        prepend = 0;
      }
      else
      {
        getline(match_stream,mystr);
      }
      cout << mystr << endl;
      stringstream ss(mystr);
      ss >> t;
      ss >> idd;
      ss >> L_x;
      ss >> L_y;
      ss >> R_x;
      ss >> R_y;
      ss >> disparity;
      ss >> vP_0;
      ss >> vP_1;
      ss >> vP_2;
      /*
        fmg 191218
        non metto piu in questo array tutti i landmark della misura ma solo quelli in tracking
      new_landmark_flag[k] = idd;
      */
      /*
        fmg 030219
        devo fare un controllo su t  per assicurarmi che sia uguale a t_
        se è != t_ , metto in measure un valore null, cioè 0,0 e idd = -1
        devo fare anche un controllo sulla disparity, per assicurarmi che sia ragionevole ?
          un controllo serio imporre di assicurarsi che non superi molto un certo percentile, ma
          questo imporrebbe di acquisire la misura in due step
      */
      if ( t != t_)
      {
        // In questo frame ci sono meno di LANDMARKS_PER_FRAME lm  !
        // Bisogna fare un padding della misura
        // La stringa appena acquisita bisogna metterla da parte per la prox misura
        mystr_2 = mystr ;
        cout << "in getmeasure() trovato un frame con meno lm. k= , mystr_2: " << k << "  " << mystr_2 << endl;
        // Padding di measure
        for (int kk = k; kk < LANDMARKS_PER_FRAME; kk++)
        {
          measure( kk*2 , 0) = 0. ;
          measure( kk*2 + 1 , 0) = 0. ;
          meas_landmark_flag[kk] = -999;
          new_landmark_flag[kk] = -999;
        }
        k = LANDMARKS_PER_FRAME; // lo faccio uscire dal ciclo di misura
      }
      else
      {
        if ( idd == last_landmark_idd + 1)  // It is a new landmark
        {
          last_landmark_idd = idd;
          new_lm.conservativeResize(new_lm.rows()+1,3);
          new_lm(new_lm.rows()-1, 0) = idd;
          //        new_lm(new_lm.rows()-1, 1) = vP_0;
          //        new_lm(new_lm.rows()-1, 2) = vP_2;
          new_lm(new_lm.rows()-1, 1) = vP_2;  /* LP 261218 */
          new_lm(new_lm.rows()-1, 2) = vP_0;
          new_landmark_flag[k] = -1;
        }
        else  // It is an old landmark
        {
          new_landmark_flag[k] = idd;
        }
        measure( k*2 , 0) = vP_2 ;
        measure( k*2 + 1 , 0) = vP_0 ;
        meas_landmark_flag[k] = idd;
      }
    }


    new_lm.conservativeResize(new_lm.rows()+1,3);

    //    new_lm(new_lm.rows()-1,0) = measured_steering;

    new_lm(new_lm.rows()-1,0) = measured_steering;

    cout << "Cocco bell ci sono\n\n"; 
    return new_lm;
  }


  static MatrixXd nonlinH(MatrixXd& state)
  {
    /* The state variabiles are sorted in this way
    x, y, phi, x1, y1, x2, y2, x3, y3, ......, V, G
    */
    /*
    fmg 231218
    usando l'array di indici new_landmark_flag, vai a vedere quali sono i lm in state che sono
    in fase di tracking ed attualizzali in modo che possano essere confrontati
    con i valori misurati attraverso le cam
    */
    long i ;
    double d, angle, dx, dy, ang;
    int n_track_lm = 0;
    /*
    fmg 231218
    prima fase: capisci quanti siano i lm in tracking
    */
    for(int k = 0; k < LANDMARKS_PER_FRAME; k++)
    {
      if ( new_landmark_flag[k] > -1 )
        n_track_lm ++;
    }
    MatrixXd mis(n_track_lm*2 + 1,1); /*ci aggiungo lo steering misurato alla fine*/
    long kk = 0;
    int trovato;
    long ii ;

    for(int k = 0; k < LANDMARKS_PER_FRAME; k++)
    {
      if ( new_landmark_flag[k] > -1)  // vado a capire in che posizione dello stato sia
      {
        trovato = -1;
        ii = 0;
        for(; ii < LANDMARKS_PER_FRAME && trovato == -1; ii ++)
        {
          if ( state_landmark_flag[ii] == new_landmark_flag[k])  trovato = 0 ;
        }
        if ( trovato == -1 && ii >= LANDMARKS_PER_FRAME)
        {
            // todo mandare in eccezione
            cout << "Eccezione in nonlinH(), analysis non coerente ii k " << ii << " " << kk <<  endl;
            exit(EXIT_FAILURE);
        }
        ii += -1 ;
        dx = state( (ii * 2) + 3,0) - state(0,0);
        dy = state( (ii * 2) + 4,0) - state(1,0);
        d = sqrt(dx*dx + dy*dy);
        angle = atan2(dy,dx) - state(2,0);
        mis(kk*2,0) = d * cos(angle);
        mis(kk*2+1,0) = -d * sin(angle);      //LP, il - ci va perché per kitti la x è a destra
        kk ++ ;
      }
    }

    mis(mis.rows()-1,0) = state(state.rows()-1,0); /*Ci aggiungo qui lo steering*/

    return mis;
  }


  MatrixXd pose(MatrixXd state)
  {
    MatrixXd pose_tmp(3,1);
    pose_tmp(0,0) = state(0,0);
    pose_tmp(1,0) = state(1,0);
    pose_tmp(2,0) = state(2,0);

    return pose_tmp;
  }

  int LoopDetector(MatrixXd& state, long time, MatrixXd* history)
  {
    ifstream loop("prova_cross_matches_2011_10_03_0027.dat");
    long jmax,i,j,k, t;
    double f, max, dx, dy; /* va aggiunto l'effetto della rotazione LP 23-9-19 */
    MatrixXd old_pose;

    while(loop >> i >> j >> k >> f)
      {
	if(j==time && time - i > 500)
	  {
	    old_pose = pose(history[i]);
	    
	    dx = old_pose(0,0) - state(0,0);
	    dy = old_pose(1,0) - state(1,0);

	    state(0,0) = old_pose(0,0);
	    state(1,0) = old_pose(1,0);
	    state(2,0) = old_pose(2,0); // - f;
	    
	    for(t=3; t<state.rows()-2; t+=2)
	      {
		state(t,0) += dx;
		state(t+1,0) += dy;
	      }
	    return 1;
	  }
      }
  
    return 0; /* ritorna 0 se non ci sono loop, 1 se ci sono*/
  }





  int perfect(const char* name)
  {
    last_landmark_idd = -1;
    N0 = 5; // X Y Phi V G
    // int new_lm = 0;
    long Assimilationcycles = ( NUMBER_OF_FRAMES - 1 ), mem = 0;
    //long NL = 0; // Number of LandMark
    double sigmad = 5, sigmaA = 3 * ppi / 360., xabs, yabs; /* 1 grado sessagesimale */
    MatrixXd new_lm ;
    ofstream traj("traiettorie.dat");
    ofstream alog("log.txt");
    //char nomepert[100];
    meas_operator = &nonlinH;
    match_stream.open(name);

    string trash;
    getline(match_stream,trash);
    MatrixXd gmunu(N0 + LANDMARKS_PER_FRAME*2,1), ModelError(2,1);
    analysis.resize(N0 + (LANDMARKS_PER_FRAME*2),1);
    aux_analysis.resize(N0 + (LANDMARKS_PER_FRAME*2),1);
    startpert.resize(N0 + (LANDMARKS_PER_FRAME*2),1);
    measure.resize(LANDMARKS_PER_FRAME*2+1,1);
    double sigma_estimate = 1.;   // nei componenti della matrice Xa, è la stima iniziale della sigma
    double gmunu_estimate = 1.;   // valore iniziale delle componenti di gmunu
    double velocita_iniziale = 0.;//.16;
    double sterzata_iniziale = 0.0;


    MatrixXd* history = new MatrixXd[Assimilationcycles];
    for(long j=0; j<Assimilationcycles; j++)
      history[j].resize(3,1);
      


    //double VV, GG = 0.;
    //cout << "read V (velocity) = " << VV << " G (steering) = " << GG << endl << endl;
    /* initializing the EKF_AUS_NL class with N0 degrees of freedom,
    6 linear Lyapunov vectors, N0-2 number of measurements,
    no nonlinear interaction*/
    //class EKF_AUS ekf_aus(N0, 6, 0, 0);
    // fmg 310119 class EKF_AUS ekf_aus(N0 + LANDMARKS_PER_FRAME*2, 6, 0, 0);
    //class EKF_AUS ekf_aus(N0 + LANDMARKS_PER_FRAME*2, 6, 0, 0);
    class EKF_AUS ekf_aus(N0 + LANDMARKS_PER_FRAME*2, 6, 0, 1);

    ekf_aus.AddInflation(1.e-1);
    ekf_aus.Lfactor(0.01);
    long nc = ekf_aus.TotNumberPert();
    Xa.resize(N0 + (LANDMARKS_PER_FRAME * 2),nc);
    aux_Xa.resize(N0 + (LANDMARKS_PER_FRAME * 2),nc);
    analysis(0,0) = analysis(1,0) = analysis(2,0) = 0.;
   
    analysis(3 + LANDMARKS_PER_FRAME*2 ,0) = velocita_iniziale;  // metri per secondo
    analysis(4 + LANDMARKS_PER_FRAME*2,0) = 0.;   // sterzata iniziale nulla
    for(long kk = 0; kk < N0 + LANDMARKS_PER_FRAME*2; kk++)
      gmunu(kk,0) = gmunu_estimate;
    // Inizializzazione, non devono essere allineati
    //srand((unsigned)time(NULL));
    
    Xa = MatrixXd::Random(Xa.rows(), Xa.cols());

    for(int j=0; j<Xa.cols();j++)
    {
      Xa(0,j) = 0.00;
      Xa(1,j) = 0.00;
      Xa(2,j) = 0.00;
      /*Xa(3,j) = 1 + j*.1 ;
      Xa(4,j) = .1 - j*.01 ;*/
      // Xa(Xa.rows()-2,j) =   j*.1 ;
      // Xa(Xa.rows()-1,j) = - j*.01 ;

    }

    new_lm = getMeasure(0);
    alog << " Misure iniziali:" << endl << new_lm << endl;

    for ( long ii = 0 ; ii < new_lm.rows(); ii ++ )
    {

      analysis(3 + ii*2,0) = new_lm(ii,1);
      analysis(4 + ii*2,0) = -new_lm(ii,2);
      for (long j=0; j < Xa.cols(); j ++)
      {
        Xa(3 + ii*2, j) = sigma_estimate;
        Xa(4 + ii*2, j) = sigma_estimate;
      }
    }
    for(long j=0; j < LANDMARKS_PER_FRAME; j ++) state_landmark_flag[j] = meas_landmark_flag[j];     
    // essendo la prima misura, tutti questi lm andranno nello stato
    //alog << endl << "analysis:" << endl << analysis << endl;
    //alog << endl << "Xa:" << endl << Xa << endl ;
    ekf_aus.P( new_lm.rows() * 2 );  //210119
    //NL += new_lm.rows();;
    cout << endl;
    //N0 = analysis.rows();
    R.resize(ekf_aus.P(),1);
    for(long i=0; i<ekf_aus.P(); i++)
    {
      if(i%2) // odd: angle
      R(i,0) = sigmad * sigmad ; // *10 è fmg 241218 LP , misuro solo posizioni, non angoli
      else // even: distance
      R(i,0) = sigmad * sigmad ;
    }
    //VectorXd errore;
    R(R.rows()-1,0) = 1.e-4;


    ModelError(0,0) = 2.; /* (m/s) error in the velocity  */
    //ModelError(1,0) = 5.*ppi/360.; /* 3 degrees error for the steering angle */
    ModelError(1,0) = 90*ppi/360.; /* 3 degrees error for the steering angle */
    //ModelError(1,0) = 90*ppi/360.; /* 3 degrees error for the steering angle */

    cout << "Initializing analyis and Xa" << endl;
    //for(long kk = 0; kk<N0; kk++) gmunu(kk,0) = 1.;

    /********************************************
    *********************************************
       Qui inizia il CICLO  delle ASSIMILAZIONI
    *********************************************
    *********************************************
    *********************************************/
    for(long kk = 0; kk < Assimilationcycles; kk++)
    {
      alog << endl << "iterazione # " << kk << endl;
      /* Preparo per l'evoluzione */
      ekf_aus.SetModelErrorVariable(Xa.rows()-2, Xa.rows()-1, ModelError, Xa);
      cout << "Iterazione #:" << kk << " kiki Xa.cols():" << Xa.cols() << endl;
      perturbazioni = ekf_aus.PrepareForEvolution(analysis, Xa, gmunu);
      //cout << "Evolving....." << endl;

      /* Evolvo.....*/
      evolve(perturbazioni, 0.1);
      evolve(analysis, 0.1);
      /* Preparo per l'analisi*/
      Xa = ekf_aus.PrepareForAnalysis(analysis, perturbazioni, gmunu);
      new_lm = getMeasure( kk + 1);

      // meas2  è una sottoparte della misura, sarebbero i lm 'in tracking'
      // MatrixXd meas2((LANDMARKS_PER_FRAME - new_lm.rows()) * 2, 1);
      int lm_mancanti = 0;
      for (int ii = 0 ; ii < LANDMARKS_PER_FRAME; ii++)
      {
        if (new_landmark_flag[ii] == -999)
          lm_mancanti ++;
      }
      MatrixXd meas2((LANDMARKS_PER_FRAME - new_lm.rows() + 1 - lm_mancanti) * 2 + 1, 1); /* +1 per lo steering */

      long iii = 0;
      for ( long ii = 0; ii < LANDMARKS_PER_FRAME; ii ++)
      {
        if (new_landmark_flag[ii] > -1)
        {
          meas2(iii*2,0) = measure( ii*2 , 0) ;
          meas2(iii*2+1,0) = measure( ii*2 + 1 , 0);
          iii ++ ;
        }
      }

      meas2(meas2.rows()-1,0) = new_lm(new_lm.rows()-1,0); /* LP , 080619, sempre steering*/


      ekf_aus.P(meas2.rows());
      R.resize(ekf_aus.P(),1);
      for(long i=0; i<ekf_aus.P(); i++)
      {
        if(i%2) // odd: angle
        R(i,0) = sigmad * sigmad ; // *10 è fmg 241218 LP , misuro solo posizioni, non angoli
        else // even: distance
        R(i,0) = sigmad * sigmad ;
      }

      //      R(R.rows()-1,0) = sigma_steer; //*sigma_steer; 
      R(R.rows()-1,0) = 0.1; //*sigma_steer; 
      //R(R.rows()-1,0) = .1; 

     
      cout << "state_landmark_flag:" << endl;
      for(long ii = 0 ; ii < LANDMARKS_PER_FRAME; ii++) cout << ii << " " << state_landmark_flag[ii] << endl ;
      cout << "new_landmark_flag:" << endl;
      for(long ii = 0 ; ii < LANDMARKS_PER_FRAME; ii++) cout << ii << " " <<  new_landmark_flag[ii] << endl ;
      cout << "meas_landmark_flag:" << endl;
      for(long ii = 0 ; ii < LANDMARKS_PER_FRAME; ii++) cout << ii << " " <<  meas_landmark_flag[ii] << endl ;

      //      if(meas2.rows()>10)

      //      if(kk<1400 || kk > 1410)

      cout << "numero di misure assimilate " << meas2.rows() << endl;

      ekf_aus.Assimilate(meas2, meas_operator, R, analysis, Xa);
 

      if(kk - mem > 10)
	{
	  if(LoopDetector(analysis, kk, history)) mem = kk;
	  /* cerco dei loop */      
	  cout << "Loop detected at time " << kk << endl; 
	}
      history[kk] = pose(analysis); /* memorizza nella history solo la pose */

      // copio analysis e Xa nei buffer ausialiari
      for(long ii = 0; ii < analysis.rows(); ii ++ ) aux_analysis(ii,0) = analysis(ii,0);
      for(long ii = 0; ii < Xa.rows(); ii ++){ for (long iii = 0 ; iii < Xa.cols() ; iii ++ ) aux_Xa(ii,iii) = Xa(ii,iii); }
      //alog << endl << "dopo assinilate , analysis:" << endl << analysis << endl;
      for (long ii = 0 ; ii < LANDMARKS_PER_FRAME; ii ++ )  // ciclo su meas_landmark_flag
      {
        int trovato = 0;
        long iii = 0;
        // verifico in state_landmark_flag[] se questo lm è già nello stato
        for (; iii < LANDMARKS_PER_FRAME && trovato == 0; iii ++)  //   ciclo su state_landmark_flag
        {
            if (state_landmark_flag[iii] == meas_landmark_flag[ii] && meas_landmark_flag[ii] != -999)  trovato = 1 ;
        }
        if (trovato == 1 )  // il lm è presente nello stato  alla posizione  iii
        {
          iii += -1 ; // fmg 220119
          analysis(3 + ii*2,0) = aux_analysis(3 + iii*2,0);   analysis(4 + ii*2,0) = aux_analysis(4 + iii*2,0);
          for(long jj = 0 ; jj < Xa.cols(); jj ++)   Xa(iii,jj) = aux_Xa(iii,jj);  // copio anche la riga di Xa
        }
        else  // metto nello stato il nuovo punto
        {
          if ( meas_landmark_flag[ii] != -999)
          {
            xabs = analysis(0,0) + measure(ii*2,0) * cos(analysis(2,0)) + measure(ii*2+1,0) * sin(analysis(2,0));
            yabs = analysis(1,0) + measure(ii*2,0) * sin(analysis(2,0)) - measure(ii*2+1,0) * cos(analysis(2,0));
            analysis(3 + ii*2,0) = xabs; analysis(4 + ii*2,0) = yabs;
          }
          else
          {
            // Padding
            analysis(3 + ii*2,0) = analysis(0,0);
            analysis(4 + ii*2,0) = analysis(1,0);
          }
	  //          for(long iii=0; iii< Xa.cols(); iii++)   Xa(ii,iii) = sigma_estimate;
          Xa(ii,0) = sigma_estimate;
        }
        state_landmark_flag[ii] = meas_landmark_flag[ii] ;
        alog << endl << " ii, state_landmark_flag[ii]: " << ii << " " << state_landmark_flag[ii] << endl;
        //alog << endl << ii << " " << state_landmark_flag[ii] << " " << meas_landmark_flag[ii] << endl;
        alog << analysis(3+ii*2,0) << " " << analysis(4+ii*2,0) << endl;

      }



      traj << analysis(0,0) << " " << analysis(1,0) << " " << analysis(2,0) << " " << analysis(analysis.rows()-2,0) << \
      " " << analysis(analysis.rows()-1,0) << endl;
    }

    traj.close();
    alog.close();
    return 0;
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

  return 2*n0 + 5; // + x,y,phi,V,G; V,G at the end
}


void readFile(const char* forecast, MatrixXd& xf)
{
  long ci = 0, N;
  double x,y,phi,V,G, xn, yn;
  ifstream in(forecast);

  in >> x;  in >> y; in >> phi;
  in >> V; in >> G;

  in >> N;

  //cout << "there are " << N << " landmarks" << endl;

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
    //cout << "Evolving column " << k << " ......." <<endl;
    //cout << XX.col(k) << endl;
    //cout << endl;
    colonna = XX.col(k);

    writeFile("statotemporaneo.dat",colonna);

    sprintf(comando,"external/my_slam/MySlam statotemporaneo.dat %d",nsteps);
    //cout << "Executing the command: " << comando << endl;

    system(comando);    ///< system call to evolve the dynamical system state

    //cout << "Evoluted column " << k << endl;
    //cout.flush();

    readFile("statotemporaneo.dat",colonna);

    XX.col(k) = colonna;

    //cout << "Terminated column " << k << endl;
    //cout.flush();
  }
}

};
