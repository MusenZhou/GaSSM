#include <stdio.h>
#include <math.h>
#include <stdlib.h>


//convert fractional coordinate to cartesian coordinate
void frac2car(double frac_a, double frac_b, double frac_c, double frac2car_a[], double frac2car_b[], double frac2car_c[],
                double cart_x[], double cart_y[], double cart_z[])
{
    cart_x[0] = frac_a*frac2car_a[0] + frac_b*frac2car_a[1] + frac_c*frac2car_a[2];
    cart_y[0] = frac_a*frac2car_b[0] + frac_b*frac2car_b[1] + frac_c*frac2car_b[2];
    cart_z[0] = frac_a*frac2car_c[0] + frac_b*frac2car_c[1] + frac_c*frac2car_c[2];
}

double trapz(double x[], double y[], int N)
{
	int i;
	double result = 0;
	for (i=1; i<N; i++)
	{
		result = result + (x[i] - x[i-1])*(y[i] + y[i-1])/2;
	}
	return result;
}

double max(double x[], int n)
{
	int i;
	double result = x[0];
	for (i=0; i<n; i++)
	{
		if (x[i]>result)
		{
			result = x[i];
		}
	}
	return result;
}

int main(int argc, char *argv[])
{
	// constant define
	double kb = 1.38e-23;
	double pi = 3.14159265359;
	double T;
	// file I/O variable
	FILE *fp1;
	int buffersize = 512;
	char str[buffersize];
	double rand_a, rand_b;
	int i;
	// read-in variables
	double alpha_degree, beta_degree, gamma_degree, alpha_rad, beta_rad, gamma_rad;
	double cell_a, cell_b, cell_c;
	int direction;
	int N;
	double mass_g_mol, mass_used;
	fp1 = fopen(argv[1],"r");
	fgets(str, buffersize, fp1);
	fgets(str, buffersize, fp1);
	fgets(str, buffersize, fp1);
	fscanf(fp1, "%lf %lf %lf", &cell_a, &cell_b, &cell_c);
	fgets(str, buffersize, fp1);
	fgets(str, buffersize, fp1);
	fscanf(fp1, "%lf %lf %lf\n", &alpha_degree, &beta_degree, &gamma_degree);
    alpha_rad = alpha_degree*pi/180;
    beta_rad = beta_degree*pi/180;
    gamma_rad = gamma_degree*pi/180;
	fgets(str, buffersize, fp1);
	fscanf(fp1, "%lf %lf %lf %lf", &rand_a, &rand_b, &mass_g_mol, &T);
	fgets(str, buffersize, fp1);
	fgets(str, buffersize, fp1);
	fgets(str, buffersize, fp1);
	fscanf(fp1, "%d\n", &direction);
	fgets(str, buffersize, fp1);
	fscanf(fp1, "%d", &N);
	fclose(fp1);

	// dependable varaibles
	double string_a[N], string_b[N], string_c[N], string_x[N], string_y[N], string_z[N];
	double string_alpha_rad[N], string_beta_rad[N], string_gamma_rad[N];
	double s[N], v[N];
	double length;
	double v_treated[N];
	fp1 = fopen(argv[2],"r");
	for (i=0; i<N; i++)
	{
		fscanf(fp1, "%lf %lf %lf %lf %lf %lf %lf\n", &string_a[i], &string_b[i], &string_c[i], &string_alpha_rad[i], &string_beta_rad[i], &string_gamma_rad[i], &v[i]);
		// printf("%lf %lf %lf %lf %lf %lf %lf\n", string_a[i], string_b[i], string_c[i], string_alpha_rad[i], string_beta_rad[i], string_gamma_rad[i], v[i]);
	}
	fclose(fp1);






	//frac2car parameter calculation
	double frac2car_a[3];
	double frac2car_b[3];
	double frac2car_c[3];
	double cart_x[1], cart_y[1], cart_z[1];
    frac2car_a[0] = cell_a;
    frac2car_a[1] = cell_b*cos(gamma_rad);
    frac2car_a[2] = cell_c*cos(beta_rad);
    frac2car_b[0] = 0;
    frac2car_b[1] = cell_b*sin(gamma_rad);
    frac2car_b[2] = cell_c*( (cos(alpha_rad)-cos(beta_rad)*cos(gamma_rad)) / sin(gamma_rad) );
    frac2car_c[2] = cell_a*cell_b*cell_c*sqrt( 1 - pow(cos(alpha_rad),2) - pow(cos(beta_rad),2) - pow(cos(gamma_rad),2) + 2*cos(alpha_rad)*cos(beta_rad)*cos(gamma_rad) );
	frac2car_c[2] = frac2car_c[2]/(cell_a*cell_b*sin(gamma_rad));
	//done!!!!!





	for (i=0; i<N; i++)
	{
		frac2car(string_a[i], string_b[i], string_c[i], frac2car_a, frac2car_b, frac2car_c, cart_x, cart_y, cart_z);
		string_x[i] = cart_x[0]*1e-10;
		string_y[i] = cart_y[0]*1e-10;
		string_z[i] = cart_z[0]*1e-10;
	}




	// calculate string length
	s[0] = 0;
	for (i=1; i<N; i++)
	{
		s[i] = s[i-1] + sqrt( pow((string_x[i]-string_x[i-1]), 2) + pow((string_y[i]-string_y[i-1]), 2) + pow((string_z[i]-string_z[i-1]), 2) );
	}
	// done!!!



	// determine the hopping length and mass
	switch (direction)
	{
		case 1:
			length = cell_a*1e-10;
			break;
		case 2:
			length = cell_b*1e-10;
			break;
		case 3:
			length = cell_c*1e-10;
			break;
	}
	mass_used = mass_g_mol/1e3/6.02214076e23;



	// calculate the diffusivity
	double k, D0;
	for (i=0; i<N; i++)
	{
		v_treated[i] = exp(-v[i]/T);
	}

	k = sqrt((kb*T)/(2*pi*mass_used)) *  (  exp( -max(v, N)/T ) / trapz(s, v_treated, N) );
	D0 = 0.5 * k * pow(length, 2);

	printf("%.5e\n", D0);
}
