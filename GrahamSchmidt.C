/*
    Copyright (C) 2017 L. Palatella, F. Grasso

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/. 

*/



/** 
 *  
 * @file  GrahamSchmidt.C
 * @author L.Palatella  
 * @date September 2017 
 *
 */




using namespace Eigen;

MatrixXd gramsh(MatrixXd dx)
{
  int N = dx.rows();
  int N_aus = dx.cols();
  int j,k,l;
  double norma, ps;
  Eigen::MatrixXd t1, t2, sum, pps;
  Eigen::MatrixXd dx_out = dx;

  //  t1.resize(N,1);
  t2.resize(N,1);

  t1 = dx.col(0);
  norma = t1.norm();
  
  if(norma > 1.e-14)   t1 /= norma;
  else
    {
      cout << "vettore troppo piccolo da normalizzare!\nnorma = " << norma << endl;
      cout << t1 << endl;
      // throw int(123);
    }



  //  cout << "norma = " << norma << endl;

  for(j=0;j<N;j++) dx_out(j,0) = t1(j,0);

  ///< inizio il ciclo su tutti i vettori proiettandoli sui precedenti
  for(j=1;j<N_aus;j++)
    {
      //  cout << j << " / " << N_aus << endl;
      sum.setZero(N,1); ///< azzero il vettore
      t1 = dx.col(j); ///< copio il nuovo vettore in t1
      
      for(k=0;k<j;k++)
	{
	  t2 = dx_out.col(k);
	  ///< ricopio i vettori gia' ortonormalizzati in t2
	  pps = t2.transpose() * t1;
	  ps = pps(0,0);
	  //	  cout << "ps = " << ps << endl;

	  ///< proietto e sottraggo la componente parallela a t2
	  sum += t2 * pps; 
	}
      t1 -= sum;
      norma = t1.norm();
      if(norma>1e-12) t1 /= norma;
      else
	{ 
	  cout << "vettore troppo piccolo da normalizzare!\n\n";
	  cout << t1 << endl;
	  //	  throw int(123);
	}


      for(l=0;l<N;l++) dx_out(l,j) = t1(l,0); ///< salvo il novo ortonormalizzato in t1
    }
  
  return dx_out;
}
