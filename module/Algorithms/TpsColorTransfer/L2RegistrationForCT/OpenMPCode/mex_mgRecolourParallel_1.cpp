#include "mex.h"
#include <omp.h>
#include <math.h>
#include <iostream>

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	double *pix, *A, *Nv, *ctrl, *recolour;    
	int num_pix, num_ctrl;   

	if (nrhs != 4) {
	mexErrMsgTxt("Four input arguments required.");
    } 
    if (nlhs > 1){
	mexErrMsgTxt("Too many output arguments.");
    }

	if (!(mxIsDouble(prhs[0]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
    if (!(mxIsDouble(prhs[1]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
    if (!(mxIsDouble(prhs[2]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
	 if (!(mxIsDouble(prhs[3]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
    

	//inputs   
	pix = (double *)mxGetPr(prhs[0]);
	A = (double *)mxGetPr(prhs[1]);
	Nv = (double *)mxGetPr(prhs[2]);
	ctrl = (double *)mxGetPr(prhs[3]);

	//figure out dimensions   
	num_pix = mxGetN(prhs[0]);
	num_ctrl = mxGetN(prhs[3]);  
	
	//associate outputs   
	plhs[0] = mxCreateDoubleMatrix(3,num_pix, mxREAL);    
	recolour = mxGetPr(plhs[0]);
	
	//std::cout << "FUCK" << std::endl;
	//std::cout << A[3] << std::endl;
	//std::cout << A[6] << std::endl;
	//std::cout << A[9] << std::endl;
	//for(int k = 0; k < 375; k++) {
	// 	if(k%3 == 0){
	//		std::cout << std::endl;
	//	}
	//	std::cout << Nv[k] << ", ";
	//}
	//std::cout << "FUCKEND" << std::endl;

	long int i, tmp_indx0,tmp_indx1,tmp_indx2;
	int j;
	double norm;
	int nThreads = omp_get_max_threads();
	#pragma omp parallel for shared(pix, A, Nv, ctrl, num_ctrl) private(j,norm,tmp_indx0, tmp_indx1, tmp_indx2)
	for(i = 0; i < num_pix; i++)
		{
			tmp_indx0 = 3*i;
			tmp_indx1 = 3*i+1;
			tmp_indx2 = 3*i+2;
			recolour[tmp_indx0] = A[0] + A[3]*pix[tmp_indx0] + A[6]*pix[tmp_indx1] + A[9]*pix[tmp_indx2]; // t+Ax1
			recolour[tmp_indx1] = A[1] + A[4]*pix[tmp_indx0] + A[7]*pix[tmp_indx1] + A[10]*pix[tmp_indx2]; //t+Ax2
			recolour[tmp_indx2] = A[2] + A[5]*pix[tmp_indx0] + A[8]*pix[tmp_indx1] + A[11]*pix[tmp_indx2]; //t+Ax3
			//recolour[tmp_indx0] = A[3]*pix[tmp_indx0] + A[6]*pix[tmp_indx1] + A[9]*pix[tmp_indx2]; // t+Ax1
			//recolour[tmp_indx1] = A[4]*pix[tmp_indx0] + A[7]*pix[tmp_indx1] + A[10]*pix[tmp_indx2]; //t+Ax2
			//recolour[tmp_indx2] = A[5]*pix[tmp_indx0] + A[8]*pix[tmp_indx1] + A[11]*pix[tmp_indx2]; //t+Ax3
			//recolour[tmp_indx0] = A[0] + pix[tmp_indx0];
			//recolour[tmp_indx1] = A[1] + pix[tmp_indx1];
			//recolour[tmp_indx2] = A[2] + pix[tmp_indx2];
		
		
		
			
			for(j = 0; j < num_ctrl; j++)
			{
				norm = -sqrt((pix[tmp_indx0] - ctrl[3*j])*(pix[tmp_indx0] - ctrl[3*j]) + (pix[tmp_indx1] - ctrl[(3*j)+1])*(pix[tmp_indx1] - ctrl[(3*j)+1]) + (pix[tmp_indx2] - ctrl[(3*j)+2])*(pix[tmp_indx2] - ctrl[(3*j)+2])); //|| Xi - cj||
				recolour[tmp_indx0] += norm*(Nv[j*3]); //|| Xi - cj||*Nvj1
				recolour[tmp_indx1] += norm*(Nv[(j*3) + 1]); //|| Xi - cj||*Nvj2
				recolour[tmp_indx2] += norm*(Nv[(j*3) + 2]);//|| Xi - cj||*Nvj3

				
			}
			
			
		}
	
	mexPrintf("nThreads = %i\n",nThreads);mexEvalString("drawnow");
}
