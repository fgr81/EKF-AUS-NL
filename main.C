/*
    Copyright (C) 2017 L. Palatella, F. Grasso

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/. 

*/




/**
 *
 * @file  main.C
 * @brief Handle to manage the experience
 * @author L.Palatella & F. Grasso 
 * @date September 2017 
 *
 */


#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "IAssimilate.C"
#include "SLAM_Assimilated.C"
#include "L96_Assimilated.C"

using namespace std;

const static std::string help = " \
\n\
This implementation of EKF-AUS-NL could be applied to several systems, described by proper dynamical equations. \n\
In ./external there are the implementations of two systems: L96 and SLAM; they can be used to test the filter. \n\
The main routine manages these two options on command line; the user has to indicate also a text file that include the initial condition, e.g. initial_SLAM.dat and initial_l96.dat. \n\
\n\
To tun on L96 model:\n\
\n\
./EKF-AUS-NL L96 initial_l96.dat > log &\n\
\n\
To tun on SLAM:\n\
\n\
./EKF-AUS-NL SLAM initial_SLAM.dat > log &\n\
\n\
The log's of the assimilation algorithm can be found in the files log and AssimilationLog.dat .\n\
";


static void show_usage(std::string name)
{
	if (name == "--help" || name == "-h") {
		cerr << help << endl;
	  	return;
	}

	cerr << "Usage: " << name << " [-h] SLAM|L96 Filename\n"
        	<< "Options:\n"
              	<< "\t-h,--help\t\tShow this help message\n"
              	<< endl;
}




/**
 *
 * @brief Main function
 * @param  argc An integer argument count of the command line arguments
 * @param  argv[1] [SLAM|L96] Type of implementation
 * @param  argv[2] Nomefile 
 * @return an integer 0 upon exit success
 *
 */
int main(int argc, char* argv[])
{
  const char* nomefile;
  static SLAM_Assimilated slam;
  static L96_Assimilated l96;
  static IAssimilate* ass;
  string arg;
  
  if (argc == 2){
    arg = argv[1];
    if (arg == "--help" || arg == "-h"){
      show_usage(argv[1]);
      return 1;
    }
  }


  if (argc < 3) {
    show_usage(argv[0]);
    return 1;
  }

  arg = argv[1];
  vector <string> sources;
  string destination;
  
  if (arg == "SLAM"){
    ass = &slam;
  }
  else 
    {
      if (arg == "L96")
	{
	  ass = &l96;
	}
      else 
	{
	  show_usage(argv[0]);
	  return 1;
	}
    }
  nomefile = argv[2];
  
  ass->perfect(nomefile);
  
  //delete ass;
  
  return 0;
}
