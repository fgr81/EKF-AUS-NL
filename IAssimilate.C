/*
  Copyright (C) 2017 L. Palatella, F. Grasso
  
  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/. 
  
*/


/**
 *
 * @file  IAssimilate.C
 * @author L.Palatella & F. Grasso 
 * @date September 2017 
 *
 */


#define ppi 6.28318530717959

#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

using namespace std;
using namespace Eigen;

ofstream alog("AssimilationLog.dat");

typedef MatrixXd (*FP)(MatrixXd&);

#include "randoknuth.c"
#include "EKF_AUS_NL.C"
#include <dirent.h>
#include <iomanip>

/**
 *
 * @class IAssimilate
 * @brief It is the skeleton that implementation classes have to hold up. 
 *
 */
class IAssimilate{
  
protected:
  
  MatrixXd truth;	   ///<  Truth vector. 
  MatrixXd analysis;    ///<  Analysis vector.  	
  MatrixXd R;    ///< Error covariance matrix.
  MatrixXd startpert;    ///< Start Perturbation 
  MatrixXd tmpi;    ///< todo
  MatrixXd measure;    ///< Measurements matrix
  MatrixXd noise;    ///< Noise matrix
  MatrixXd perturbazioni;    ///< Perturbations matrix
  MatrixXd T_A_perturbazioni;    ///< 
  FP meas_operator;    ///< Function pointer to measurement operator
  long m;    ///< todo
  long n;    ///< todo
  size_t N0; ///< Number of rows of state vector 
  long mm;    ///<
  long SEED;    ///< todo

public:


  /**
     @brief Internal implementation of gaussian ditribution.
   */
  virtual double gaussiano() = 0;
  
  
  /**
     @brief It's the ENGINE of the program.
     * It initializes the variables and it follows the algorithm loop: 
     * PrepareForEvolution -> evolve -> PrepareForAnalysis -> Assimilate 
     *
     *@param name: Input filename.
  */
  virtual int perfect(const char*) = 0;
  
  
  /**
     @brief It performs the time evolution. 
     *
     *@param XX: The time evolution is performed over this matrix's columns.
     *@param time: Useful to perform time dependant evolution.
     *@return void
     *
  */
  virtual void evolve(MatrixXd&, double) = 0;

  //  virtual int ActualAssimilation(const char*) = 0;

  /**
     @brief This is the routine that performs the actual data
     assimilation esperiment.
     *
     @param filename with the initial condition 
     @return int
  */

  // virtual MatrixXd GetMeasure() = 0;


  /**
     @brief This routine does the actual read of measured values
     from the suitable external device (camera, radar, ultrasound sensor,...)

     @param nothing, can be modified if the port number or similar information
     is needed
     @return a px1 MatrixXd with the measured values 
  */

  virtual long dimensions(const char*) = 0;
  
  
  /**
     @brief Handle program parameter file.
  */
  virtual void readFile(const char*, MatrixXd&) = 0;
  
  
  /**
     @brief Handle program parameter file. 
   */
  virtual void writeFile(const char*, MatrixXd&) = 0;
  
};
