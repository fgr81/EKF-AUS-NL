CFLAGS=-O
LIBS=-lm
MYSLAM=external/my_slam
MYL96=external/my_l96

EKF-AUS-NL: main.C SLAM_Assimilated.C randoknuth.c L96_Assimilated.C IAssimilate.C myslam myl96
	g++ -o EKF-AUS-NL main.C  $(LIBS) $(CFLAGS)


myslam: $(MYSLAM)/MySlam.C
	g++ -o $(MYSLAM)/MySlam $(MYSLAM)/MySlam.C


myl96:	$(MYL96)/MyL96.C
	g++ -o $(MYL96)/MyL96 $(MYL96)/MyL96.C 


.PHONY: clean

clean:
	rm -f EKF-AUS-NL $(MYSLAM)/MySlam $(MYL96)/MyL96 

