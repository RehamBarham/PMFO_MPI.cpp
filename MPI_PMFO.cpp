/* 
   PMFO-LP: Parallel Moth Flame Optimization (MFO) for Link Prediction problem
   Developed in Microsoft visual studio environment using C++ and utilizing the OpenMPI  
   @Author: Reham Barham
   Date: 10 May 2018  
*/  
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include<complex>
#include<tuple>
#include <stdexcept>
#include <mpi.h> 
using namespace std;
int dim=4;
int k=2;
int popSize=50;
const double PI = 3.1415927;
int maxIter=200;
//int h=0;

//-------------------------------------------------------------------
//            struct type for individual:
//-------------------------------------------------------------------
struct population{
	
	double **position;
    double fitness;
	double gmean;
	double auc;
  //deafult constructor
	population(){
		double** position1= new double *[k];
	        for(int i = 0; i <k; i++)
	            {position1[i] = new double[dim];}
		for(int i=0;i<k;i++){
			for(int j=0;j<dim;j++){
				
				position1[i][j]=0;
			}
		}
		position=position1;
		fitness=0;
	}
  // constructor:
  /*	population(double** position_,double fitness_){
		
		for(int i=0;i<k;i++){
			for(int j=0;j<dim;j++){
				
				position[i][j]=position_[i][j];
			}
		}
		position;
		fitness=fitness_;
		}*/
  //one member constructor
  population(double** position_){
		double **pos= new double *[k];
	        for(int i = 0; i <k; i++)
	            {pos[i] = new double[dim];}
		for(int i=0;i<k;i++){
			for(int j=0;j<dim;j++){
				
				pos[i][j]=position_[i][j];
			}
		}
		 position=pos;
		
	}
    population(double fitness_){
		fitness=fitness_;
		
	}
void printIndividualInfo();	
// initialization


};

//-------------------------------------------------------------------
//            struct type for dataset:
//-------------------------------------------------------------------
/*struct dataSet{
	
	double **subFeatures;
    double **subTarget;
  //deafult constructor
	dataSet(){
		double** subF= new double *[subDS_size];
	        for(int i = 0; i <subDS_size; i++)
	            {subF[i] = new double[dim];}
		for(int i=0;i<subDS_size;i++){
			for(int j=0;j<dim;j++){
				
				subF[i][j]=0;
			}
		}
		subFeatures=subF;
		////////////////////////////////
		double** subT= new double *[subDS_size];
	        for(int i = 0; i <subDS_size; i++)
	            {subF[i] = new double[1];}
		for(int i=0;i<subDS_size;i++){		
				subF[i][1]=0;
		}
		subTarget=subT;
		
	}
  //two member constructor
	/*dataSet(double** subFeatures_, double ** subTarget_){
		
		for(int i=0;i<subDS_size;i++){
			for(int j=0;j<dim;j++){
				
				subFeatures[i][j]=subFeatures_[i][j];
			}
		}
		subFeatures;

		for(int i=0;i<subDS_size;i++){
				
				subTarget[i][1]=subTarget_[i][1];
		}
		subTarget;
		}
 
};*/


//-------------------------------------------------------------------
//            For fitness comparison
//-------------------------------------------------------------------
bool acompare(population lhs, population rhs) { return lhs.fitness < rhs.fitness; }
       
//-----------------------------------------------------------------------
//                  upper bound function
//-----------------------------------------------------------------------
double* upperBound(double ** feat, int subDS_size){
	double m;
	double* varMx;
	varMx = new double[dim];
	for(int j=0;j<dim;j++){
		m=feat[1][j];
		for(int i=0;i<subDS_size;i++){
			if(m<feat[i][j])
		{
			m=feat[i][j];
		}
		}
		varMx[j]=m;
	}
	return varMx;

}
//-----------------------------------------------------------------------
//                  lower bound function
//-----------------------------------------------------------------------
double* lowerBound(double ** feat, int subDS_size){
	double m;
	double* varMn;
	varMn = new double[dim];
	for(int j=0;j<dim;j++){
		m=feat[1][j];
		for(int i=0;i<subDS_size;i++){
			if(m>feat[i][j])
		{
			m=feat[i][j];
		}
		}
	    varMn[j]=m;
	}
	return varMn;

}

//-----------------------------------------------------------------------
//                  display position matrix function
//-----------------------------------------------------------------------
void displayp(population individ){
	cout<<"display again"<<"\n";
int i1,j1;
            for(i1 = 0;i1 < k;i1++)
            {
                for(j1 = 0; j1 < dim; j1++)
                    cout<<individ.position[i1][j1]<<"  ";
                cout<<endl;
            }
			//cout<<indiv.fitness;
}
//-----------------------------------------------------------------------
//                  display  a matrix 
//-----------------------------------------------------------------------
void display(double ** m, int row, int col){
	cout<<"display matrix"<<"\n";

            for(int i = 0;i < row;i++)
            {
                for(int j = 0; j < col; j++)
                    cout<<m[i][j]<<"  ";
                cout<<endl;
            }
			//cout<<indiv.fitness;
}
//-----------------------------------------------------------------------
//                  display  avector  matrix 
//-----------------------------------------------------------------------
void displayV(double ** m, int row, int col){
	cout<<"displayvector matrix"<<"\n";

            for(int i = 0;i < row;i++)
            {
                
                    cout<<m[i][0]<<"  ";
                cout<<endl;
            }
			
}

//-------------------------------------------------------------------
//                        print individual vector information
//--------------------------------------------------------------------
// print vector information
void population:: printIndividualInfo(){
  for(int i1 = 0;i1 < k;i1++)
            {
                for(int j1 = 0; j1 < dim; j1++)
                    cout<<position[i1][j1]<<"  ";
                cout<<endl;
            }
  cout<< fitness<<endl;
  
}




 //--------------------------------------------------------------------------
//                        initialization
//---------------------------------------------------------------------------
double** initialization(double * vmn,double* vmx){
	double** pos;
	pos = new double*[k];
    for (int k3 = 0; k3 < k; ++k3)
      {
         pos[k3] = new double[dim];
      }

	for(int i=0; i<k;i++){
		for(int j=0;j<dim;j++){
			pos[i][j]=((double) rand() / (RAND_MAX+1)) * (vmx[j]-vmn[j]) + vmn[j];

		}
	}
	 
	return pos; 
 }

 
//--------------------------------------------------------------------
//                     cost function
//--------------------------------------------------------------------

 tuple<double, double, double> costFunction(double** indv,double** f, double** tar, int e){ 
	
	 double dist1=0;
	 double dist2=0;
	 double Eucldist1;
	 double Eucldist2;
	 double** distance= new double *[e];
	        for(int i = 0; i <e; i++)
	            {distance[i] = new double[k];}

     double** index= new double *[e];
	        for(int i = 0; i <e; i++)
	            {index[i] = new double[1];}

//---------------fisrst centroid SSD---------------
			for(int i=0; i<e;i++){
				for (int j=0; j<dim; j++){
					dist1 += pow(indv[0][j]-f[i][j],2);
					dist2 += pow(indv[1][j]-f[i][j],2);
				}
				Eucldist1=sqrt(dist1);
				distance[i][0]=Eucldist1;
				Eucldist2=sqrt(dist2);
				distance[i][1]=Eucldist2;
			}
			// display distance matrix
			/*for(int i = 0;i < e;i++)
            {
                for(int j = 0; j < k; j++)
                    cout<<distance[i][j]<<"  ";
                cout<<endl;
            }*/
			//check for minimum dist
			
			for (int i=0; i<e;i++){
				if(distance[i][0]<distance[i][1]){					
					index[i][0]=1;
				} 
				else{ 	
					index[i][0]=2;
				}			
			}
			/*	cout<<"the target is"<<endl;
			display(tar,e,1);
// display index prediction matrix
			cout<<"index    "<<endl;
			for(int i = 0;i < e;i++)
            {
                
                cout<<index[i][0]<<"  ";
                cout<<endl;
		}*/
//------confusion matrix-------
//std::tie(error, auc, gm) = confusionMatrix(index,tar, e);
			double tp=0, tn=0, fp=0, fn=0;

	for(int i=0;i<e;i++){
		 				 
						 if(index[i][0]==1 && tar[i][0]==1){
						 tp=tp+1;}
						 if(index[i][0]==2 && tar[i][0]==2){
						 tn=tn+1;
						 // cout<<"tn is woow"<<tn<<endl;
}
						 if(index[i][0]==1 && tar[i][0]==2){
						 fp=fp+1;}
						 if(index[i][0]==2 && tar[i][0]==1){
						 fn=fn+1;}
						  
						
					 }

	//cout<<"fp, tp, tn, fn  "<< fp<<"  " <<tp<<"  "<< tn<<"  "<< fn<<endl;
 double predictedYes=fp+tp;
 // double predictedNo=tn+fn;
 double actualYes= fn+tp;
 double actualNo=tn+fp;
 double precision=tp/predictedYes;
 double sensetivity=tp/actualYes;
 // double specificity=tn/actualNo;
 double fpr=fp/actualNo;
 //cout<<"sen is  "<<sensetivity<< " fpr is "<< fpr<<endl;
 double A1=(sensetivity*fpr)/2;
 double A2=((1-fpr)*(1+sensetivity))/2; 
 double AUC=A1+A2;
 double accuracy=(tp+tn)/(tp+tn+fp+fn);
 double gMean = sqrt (sensetivity *precision);
 // double fMeasure=2*((precision*sensetivity)/(precision+sensetivity));
 double error= 1-accuracy;
//cout<< error<<"auc gm"<<AUC<<" "<<gMean;

	 return make_tuple(error, gMean, AUC);
 }

//---------------------------------------------------------------------
//                 display an array
//---------------------------------------------------------------------
 void displayArray(int * arr){
	 for (int i=0;i<popSize;i++){
		 cout<< arr[i];
	 }
 }


 //---------------------------------------------------------------------
                       // matrix assignment
//-----------------------------------------------------------------------


 double** matrixAssignment(double** matrix, int ro,int co){
	 double** mat= new double *[ro];
	        for(int i = 0; i <ro; i++)
	            {mat[i] = new double[co];}
	 for(int i=0;i<ro;i++){
		 for(int j=0;j<co;j++){
			 mat[i][j]=matrix[i][j];

		 }
	 }
	 return mat;
 }


 //---------------------------------------------------------------------
                       // matrix assignment
//-----------------------------------------------------------------------


 int** matrixAssignmentInt(int** matrix, int r, int c){
	 int** mat= new int *[r];
	        for(int i = 0; i <r; i++)
	            {mat[i] = new int[c];}
	 for(int i=0;i<r;i++){
		 for(int j=0;j<c;j++){
			 mat[i][j]=matrix[i][j];

		 }
	 }
	 return mat;
 }

 //----------------------------------------------------------------------
//                        DS division
//----------------------------------------------------------------------
double** division(int subDS_size, double** dataset, int r){
      int h=r*subDS_size;
      double** subDataset = new double*[subDS_size];
      for (int i3 = 0; i3 < subDS_size; i3++)
        {
           subDataset[i3] = new double[dim+1];
        }
	 for(int i=0; i<subDS_size; i++){
			   for(int j=0;j<dim+1; j++){
			      subDataset[i][j]=dataset[h][j];
			   }
			   h=h+1;

		   }
	 return subDataset;
}

 //----------------------------------------------------------------------
//                        features partition
//----------------------------------------------------------------------
 double** featureExtraction(int size, double** subDS){
	 double** subfeat = new double*[size];
      for (int i3 = 0; i3 < size; i3++)
        {
           subfeat[i3] = new double[dim];
        }
	  //-------------------------------------
       for(int i = 0; i < size; i++)
         { 
			for(int j = 0; j<dim ; j++)
              {
				subfeat[i][j]=subDS[i][j];
			    
			  }	     
		   }
	   return subfeat;
 }
//----------------------------------------------------------------------
//                        target partition
//----------------------------------------------------------------------
	  
     double** targetExtraction(int size, double** subDS){
	 double** subtar = new double*[size];
      for (int i3 = 0; i3 < size; i3++)
        {
           subtar[i3] = new double[1];
        }
	  //-------------------------------------
       for(int i = 0; i < size; i++)
         { 	
				subtar[i][0]=subDS[i][dim];  	     
		   }
	   return subtar;
 }
//----------------------------------------------------------------------
//                        MFO function 
//         MFO as in (Mirjalili, S.: Moth-flame optimization algorithm: A novel nature inspired heuristic paradigm. Knowledge-Based Systems, 89, 228-249 (2015))
//----------------------------------------------------------------------

tuple<double, double, double> MFO(double** target, double** features, int dim, int k, int maxIter, int popSize, int e1){
double error;
double gm;
double Auc;
double* varMin;
varMin = new double[dim];
double* varMax;
varMax = new double[dim];
double**Best_flame_pos;
double Best_flame_score=0;
double Best_gm=0;
double Best_auc=0;
vector< population > individual(popSize);
vector<population>sorted_individual;
vector<population>previous_population;
vector<population>best_positions;
vector<population>sorted_best_positions;
vector<population>::iterator it;
//cout<<"this is the feature"<<endl;
// display(features,e1,dim);
double** position1;
	position1 = new double*[k];
    for (int k3 = 0; k3 < k; ++k3)
      {
         position1[k3] = new double[dim];
      }
// compute upper and lower bounds array---------------------
	varMin =lowerBound(features, e1);
	/*	cout<<"lower bounds";
	for (int i=0;i<dim;i++){
	  //cout<<varMin[i];
	  }*/
	varMax =upperBound(features,  e1);
	/*	cout<<"upper bounds";
	for (int i=0;i<dim;i++){
	  cout<<varMax[i];
	  }*/
// individual initialization---------------------------------
   for(int i=0; i< popSize; i++){
	 position1=initialization(varMin,varMax);
	 individual[i]=population(position1);
	 //cout<<"the individual at bigining  "<< i <<"is " ;
	 // display(individual[i].position, k, dim);
	 //cout<<endl;
	 // cout<<"\n";
        }

  //cout <<"a new print again"<<"\n";

// Now setup an iterator loop through the vector
	 /*  vector<population>::iterator it;
	   for ( it = individual.begin(); it != individual.end(); ++it ) {
	      // For each friend, print out their info
	      it->printIndividualInfo();
	   }*/
  
// MFO optimizaton algorithm----------------------------------------------------------
 //main looop
	 int iteration=1;
	 while (iteration<(maxIter+1)){
		 //cout<<"the iteration no is  "<< iteration<<endl;
		 //Number of flames
		int Flame_no=(int)((maxIter-iteration*((popSize-1)/maxIter)));
		//cout<<"flame no. is "<<Flame_no<<endl;

// Check if moths go out of the search spaceand bring it back
		for(int y=0;y<popSize;y++){
			for(int e=0;e<k;e++){
				for(int f=0;f<dim;f++){
					if (individual[y].position[e][f]<varMin[f]){
						double mnn=varMin[f];
						individual[y].position[e][f]=mnn;
					}
					if (individual[y].position[e][f]>varMax[f]){
						double mxx=varMax[f];
						individual[y].position[e][f]=mxx;
					} 
				}
			}
			//	cout<<" the individual after corrections is  "<< y<<"\n";
			// display(individual[y].position, k, dim);
			//calculate fitness of moths
			std::tie(error, gm, Auc)=costFunction(individual[y].position, features,target, e1);
			//	cout<<""<<error<<endl;
			individual[y].fitness=error;
			individual[y].gmean=gm;
			individual[y].auc=Auc;
			//cout<< "fitness for i =  "<<y<<"  is  "<< individual[y].fitness<<"\n";
		}// end of for loop
		
		if(iteration==1){
			//Sort the first population of moths
			
			std::sort(individual.begin(), individual.end(), acompare);
			sorted_individual.assign(individual.begin(),individual.end());
                	best_positions.assign(sorted_individual.begin(),sorted_individual.end());
			//sorted_individual=individual;
			/*cout<<"after sorting"<<endl;
			for(int i=0;i<popSize;i++){
				display(sorted_individual[i].position);
				cout<<sorted_individual[i].fitness<<endl;
			}*/
		}
		else{
			//merge the population, then sort
			/*cout<<"second if else"<<endl;
			for(int i=0;i<popSize;i++){
				display(individual[i].position);
				cout<<individual[i].fitness<<endl;
			}*/
			//double_sorted_individual=previous_population;
			best_positions.insert(best_positions.end(), previous_population.begin(), previous_population.end());
	sorted_best_positions.assign(best_positions.begin(),best_positions.end());
			//print after concatenating and before sorting
			
	   
			//sorting after concatenating
			std::sort(sorted_best_positions.begin(), sorted_best_positions.end(), acompare);
			//std::sort(previous_population.begin(), previous_population.end(), acompare);
			/*cout<<"after cocatenating and sorting"<<endl;
			
	   for ( it = individual.begin(); it != individual.end(); ++it ) {
	      // For each friend, print out their info
	      it->printIndividualInfo();
	   }*/
		sorted_individual.assign(sorted_best_positions.begin() + 0, sorted_best_positions.begin() + popSize);
		best_positions.assign(sorted_individual.begin(),sorted_individual.end());
		/*cout<<"the sorted after taking elements from 1--popsize "<<endl;
		for ( it = sorted_individual.begin(); it != sorted_individual.end(); ++it ) {
	      // For each friend, print out their info
	      it->printIndividualInfo();
	   }*/
		
		}
		//Update the position best flame obtained so far
 		Best_flame_score=sorted_individual[0].fitness;
		Best_gm=sorted_individual[0].gmean;
		Best_auc=sorted_individual[0].auc;
		Best_flame_pos=matrixAssignment(sorted_individual[0].position, k, dim);
		//cout<<"best flame is is"<<endl;
		//display(Best_flame_pos);
		
		previous_population.assign(individual.begin(),individual.end());
		//previous_population=individual;
		/*cout<<"previus population"<<endl;
			for(int i=0;i<popSize;i++){
				display(previous_population[i].position);
				cout<<previous_population[i].fitness<<endl;
			}*/
		//a linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
		double a=(-1)+iteration*((-1)/maxIter);
		//cout<<"a is "<<a<<endl;
		for(int i=0;i<popSize;i++){
			
			for(int h=0;h<k;h++){
				for(int f=0;f<dim;f++){
					if(i<=Flame_no){//Update the position of the moth with respect to its corresponsing flame
						// D 
					  	double distance_to_flame=abs(sorted_individual[i].position[h][f]-individual[i].position[h][f])+1;
						 int b=1;
						double t=(a-1)*((double)rand()/(double)RAND_MAX)+1;
						//cout<<distance_to_flame<<"  "<<t<<endl;
						// 
						
						individual[i].position[h][f]=((distance_to_flame*(exp(b*t))*(cos(t*2*PI)*t))+(sorted_individual[i].position[h][f]));
						//cout<<"element is "<<individual[i].position[h][k]<<endl;

					}
					if(i>Flame_no){
					  	double distance_to_flame=abs(sorted_individual[i].position[h][f]-individual[i].position[h][f]);
						cout<< distance_to_flame<<endl;
						int b=1;
						double t=(a-1)*((double)rand()/(double)RAND_MAX)+1;
						
						// Eq. (3.12)
					individual[i].position[h][f]=(distance_to_flame*(exp(b*t))*(cos(t*2*PI)/t))+(sorted_individual[Flame_no].position[h][f]);
						//cout<<"element 2 is "<<individual[i].position[h][k]<<endl;
					}
				}
			}
			//cout<<"before"<<endl;
//display(Best_flame_pos);
		}
		
//cout<<"iteration no. is=  "<< iteration<< "  best obtained so far is  "<<Best_flame_score<<"\n";
//display(Best_flame_pos);
//cout<<endl;

iteration=iteration+1;
	 }
	
     cout<<endl;
     //cout<<"best fittness is   "<<Best_flame_score<<endl;
	 
return make_tuple(Best_flame_score, Best_gm, Best_auc);
}

//----------------------------------------------------------------------
//                        main function
//----------------------------------------------------------------------
int main(int argc, char** argv){


 double**results= new double*[1];
      for (int i3 = 0; i3 < 1; i3++)
        {
           results[i3] = new double[6];
        }


	
int number_of_lines = 0;
std::string line;
int E5;
int subDS_size;
 clock_t  start,end,startcomp, endcomp;
double total;
double totalf;
 double compTime;
//double** results;
double subBest_error, subBest_gM, subBest_auc;
// double subBest_error1,subBest_gM1, subBest_auc1;
double subBest_error_0, subBest_gM_0, subBest_auc_0;

double final_best_error;
double final_best_gM;
double final_best_auc;
srand ( time(NULL) );
ifstream file;
// double** fea, targ;

//vector< dataSet > sub;

MPI_Init(NULL, NULL);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  
 // cout<<"iam processor number  "<< world_rank<<endl;
  //cout<<"reham1"<<endl;
// get the number of instances and number of shared instances
	
std::ifstream myfile("wiki-vote.txt");
//cout<<"reham2"<<endl;

while (std::getline(myfile, line)){ ++number_of_lines;}
E5=number_of_lines;
subDS_size=E5/world_size;
 cout<< E5<<subDS_size<<endl;
 // cout<<"reham3"<<endl;
myfile.close();
 // read from file dataset and then store data into a matrix
  file.open("wiki-vote.txt", ios::in); 	
	//////

//////////////////////////////////////////////////////////////////
double** dataset = new double*[E5];
      for (int i3 = 0; i3 < E5; i3++)
        {
           dataset[i3] = new double[dim+1];
        }
// sub dataset declaration////////////////////////////////
	   double** subDataset = new double*[subDS_size];
      for (int i3 = 0; i3 < subDS_size; i3++)
        {
           subDataset[i3] = new double[dim+1];
        }
//////////////////////////////////////////////////////////////////
double** subfeatures;
 subfeatures= new double*[subDS_size];
      for (int i3 = 0; i3 < subDS_size; i3++)
        {
           subfeatures[i3] = new double[dim];
        }
////////////////////////////////////////////////////
double** subtarget;
subtarget = new double*[subDS_size];
       for (int i3 = 0; i3 < subDS_size; i3++)
        {
           subtarget[i3] = new double[1];
        }
/////////////////////////////////////////////////////
// dataset matrix
       for(int i = 0; i < E5; i++)
         { cout<<"\n";
			for(int j = 0; j<dim+1 ; j++)
              {
			    file >> dataset[i][j];
			    //cout<< dataset[i][j]<< " ";
			  }	     
		   }
       
///////////////////////////////////////////
file.close();	   
 //for(int w=0; w<world_size;w++){
 subDataset=division(subDS_size,  dataset, world_rank);
for(int i=0;i<E5;i++){
delete [] dataset[i];
}
delete [] dataset;
//delete dataset;		  
		   cout<<endl;
		   subfeatures= featureExtraction(subDS_size, subDataset);

		   subtarget= targetExtraction(subDS_size, subDataset);
for(int i=0;i<subDS_size;i++){
delete [] subDataset[i];
}
//delete [] subDataset;
//delete subDataset;
		   /* if(world_rank==0){
		     display(subDataset, subDS_size,dim+1);
		     cout<<endl;
		     display(subfeatures, subDS_size,dim);
		     cout<<endl;
		     displayV(subtarget, subDS_size,1);
		     cout<<endl;


		     }*/
		   //sub[w]=dataSet(subfeatures,subtarget);
		   // sub[w].subFeatures=subfeatures;
		   // sub[w].subTarget=subtarget;
		   //}
		    cout<<endl;
//for(int ii=0;ii<1;ii++){
double best_error=0;
double best_gM=0;
double best_auc=0;
//Master processor tasks:------------//////////////////////////////
 start=clock();
if(world_rank > 0)

  {
 std::tie(subBest_error, subBest_gM, subBest_auc)= MFO(subtarget, subfeatures, dim, k, maxIter, popSize, subDS_size);
 //cout<<"hi"<<endl;
 cout<<" I am processor number "<<world_rank<<"my error is  "<<subBest_error<<endl;
for(int i=0;i<subDS_size;i++){
delete [] subfeatures[i];
}
//delete [] subfeatures;

for(int i=0;i<subDS_size;i++){
delete [] subtarget[i];
}
//delete [] subtarget;
 //display(subDataset, subDS_size,dim+1);
    				 MPI_Send(&subBest_error, 1, MPI_DOUBLE, 0, 0,MPI_COMM_WORLD);
				 MPI_Send(&subBest_gM, 1, MPI_DOUBLE, 0, 0,MPI_COMM_WORLD);
				 MPI_Send(&subBest_auc, 1, MPI_DOUBLE, 0, 0,MPI_COMM_WORLD);

  }




 else 
{

	  

	  for( int i=1;i<world_size;i++)
			{	

			  MPI_Recv(&subBest_error, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			  // cout<<"subBest_error"<<subBest_error <<endl;
			  MPI_Recv(&subBest_gM, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			  MPI_Recv(&subBest_auc, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			 
				best_error  = best_error + subBest_error;
				best_gM  = best_gM + subBest_gM;
				best_auc  = best_auc + subBest_auc;
			}

          //cout <<"total profit is "<<  total_sol +  knapsack(0 ,cpu_shar + main_cpu_share ,p,w,cpu_shar_w + main_cpu_share_w)<< endl;
	  startcomp=clock();
      std::tie(subBest_error_0, subBest_gM_0, subBest_auc_0)= MFO(subtarget, subfeatures, dim, k, maxIter, popSize, subDS_size);
for(int i=0;i<subDS_size;i++){
delete [] subfeatures[i];
}
//delete [] subfeatures;

for(int i=0;i<subDS_size;i++){
delete [] subtarget[i];
}
//delete [] subtarget;
		 final_best_error=(subBest_error_0 + best_error)/world_size;
		 final_best_gM=(subBest_gM_0 + best_gM)/world_size;
		 final_best_auc=(subBest_auc_0 + best_auc)/world_size;
		 endcomp=clock();
end=clock();
cout<<" error rate    "<<final_best_error<<"   g-mean is  "<<final_best_gM<< " area under curve is "<<final_best_auc<<endl;

total=((double)end-start);
totalf=total/CLOCKS_PER_SEC;
compTime=((double)endcomp-startcomp)/CLOCKS_PER_SEC;
cout<<"computation time is "<< compTime<<endl;
cout<<"total exuction time is "<< totalf<<endl;
  // the end of master


results[0][0]=1;
results[0][1]=final_best_error;
results[0][2]=final_best_gM;
 results[0][3]=final_best_auc;
 results[0][4]=compTime;
results[0][5]=totalf;
}
// }

 if(world_rank==0){
// write the results into a txt file
std::ofstream output("wiki-vote_results/wiki-vote_16.txt", ios::app); 
for (int i=0;i<1;i++)
{
	for (int j=0;j<6;j++)
	{
		output << results[i][j] << " "; 
	}
	output << "\n";
}
output.close();
 }	 

MPI_Finalize();
 return 0;
}
