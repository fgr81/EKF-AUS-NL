#include <stdio.h>
#include <iostream>
#include <fstream>

#include "../../Eigen/Dense"
#include "../../Eigen/Eigenvalues"

#define ppi 6.28318530717959
#define pi  3.14159265358979

using namespace std;
using namespace Eigen;


MatrixXd xdlor(MatrixXd& x)
{
  double F=8.;
  size_t i, j1, j2, N = x.rows();
  MatrixXd out(N,1);
  
  for(i=0;i<N;i++)
    {
      j1 = i-1+N; j2 = i-2+N;
      j1 = j1%N; j2=j2%N;
      
      //      out(i,0) = ( (*x)((i+1)%N,0) - (*x)(j2,0) ) * (*x)(j1,0) - (*x)(i,0) + F;

      out(i,0) = ( x((i+1)%N,0) - x(j2,0) ) * x(j1,0) - x(i,0) + F;

    }
  return out;
}


void HenonEvolve(MatrixXd& state, double time_evolution, double dt)
{
  double t = 0.;
  MatrixXd xd, xp, xdp;

  
  while(t <= time_evolution)
    {
      xd = xdlor(state);
      xp = state + xd*dt;
      xdp = xdlor(xp);
	  
      state += 0.5*(xd + xdp)*dt; 
      t += dt;

    }
}



int main(int argc, char* argv[])
{
  long int N0;
  double tfin; 

  if(!argv[1])
    {
      printf("\nUsage: %s <nomefile> <timestep>\n\n", argv[0]);
      exit(0);
    }


  ifstream in(argv[1]);

  in >> N0;

  cout << "Dimensione sistema N0 = " << N0 << endl;

  sscanf(argv[2],"%lf",&tfin);

  MatrixXd state(N0,1);

  for(long i=0; i<N0; i++) in >> state(i,0);

  in.close();

  HenonEvolve(state, tfin, 0.0125);
  
  ofstream out(argv[1]);

  out << N0 << endl;

  for(long i=0; i<N0; i++) out << state(i,0) << endl;
  
  out.close();
}





