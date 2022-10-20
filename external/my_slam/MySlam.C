#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>

#include "../../Eigen/Dense"
#include "../../Eigen/Eigenvalues"

#define ppi 6.28318530717959
#define pi  3.14159265358979


using namespace std;
using namespace Eigen;



//--------- Fine definizioni ---------------------------

double riportapi2(double angle)
{
  /* modify the angle value to mantain it in 
     the interval -pi pi */


  if(angle < pi && angle > -pi) return angle;

  if(angle < -pi)
    {
      while(angle < -pi) angle += ppi;
      return angle;
    }

  if(angle > pi)
    {
      while(angle < pi) angle -= ppi;
      return angle;
    }
  

}

int main(int argc, char* argv[])
{
  if(!argv[1])
    {
      printf("\nUsage: %s <nomefile> <timestep>\n\n", argv[0]);
      exit(0);
    }

  double x,y,phi, xn, yn, phin;
  double Dt = 0.025, V, sigmaV, G, sigmaG = 1. * ppi / 360.; 
  /* G is the steering angle */
  double B = 1., sigmaR, sigmaB, R;
  double Trev, Gmax = 0.05;
  long N;
  
  ifstream in(argv[1]);
  
  in >> x;  in >> y; in >> phi;
  in >> V; in >> G;

  in >> N; 

  MatrixXd lm(N,2);

  for(long i=0; i<N; i++)
    {
      in >> xn; in >> yn;

      lm(i,0) = xn;
      lm(i,1) = yn;
    }

  in.close();

  long NT;

  sscanf(argv[2],"%ld",&NT);

  Trev = ppi * B / Gmax / V;

  for(long j=0; j<NT; j++)
    {
      xn = x + V*Dt*cos(G + phi);
      yn = y + V*Dt*sin(G + phi);
      phin = phi + V*Dt/B * sin(G);

      x = xn; y = yn; phi = phin;

    }

  phi = riportapi2(phi);
  
  ofstream out(argv[1]);
  
  out.setf(ios_base::fixed);

  out << x << endl; 
  out << y << endl;
  out << phi << endl;  
  out << V << endl;  
  out << G << endl;  
  out << N << endl;  

  for(long i=0; i<N; i++)
    out << lm(i,0) << " " << lm(i,1) << endl;
  
  out.close();
  
  return 0;
}






