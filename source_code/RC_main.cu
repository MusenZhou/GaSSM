#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cub/cub.cuh>
#include "debug.h"
#include "cpu_fxn.h"
#include "device_fxn.h"
#include "global.h"

#define PI 3.141592653589793
#define running_block_size 32
#define coul2Klevin 1.6710095663e+05










__global__
void print_convergence(int n, double *x1, double *x2, double *x3, double *x4)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                printf("%lf\t%lf\n", 1.0*x1[0]/x3[0], 1.0*x2[0]/x4[0]);
            }
        }
    }
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
    //To calculate the external potential field, two file strings are needed: input filename and output filename
	//define file varaiable
	FILE *fp1;
	int buffersize = 512;
	char str[buffersize];
    char conv_string[buffersize];
	//define read-in parameters
	// int Nmaxa, Nmaxb, Nmaxc;
	double La, Lb, Lc, dL;
	double alpha, beta, gamma;
    double alpha_rad, beta_rad, gamma_rad;
    int FH_signal;
    double mass, temperature[1];
    int set_running_step;
	double cutoff[1];
    int N_string[1];
    int int_N_string;
    int direction[1];
    double move_angle_degree[1], move_angle_rad[1], move_frac[1];
	int N_atom_frame[1], N_atom_adsorbate[1];
    double set_conv_trans_percent, set_conv_rot_percent;
    //define ancillary parameters
    double center_of_mass_x[1], center_of_mass_y[1], center_of_mass_z[1], total_mass_adsorbate;
    double temp_x[1], temp_y[1], temp_z[1];
    double cart_x, cart_y, cart_z;
    double cart_x_extended[1], cart_y_extended[1], cart_z_extended[1];
    int times_x[1], times_y[1], times_z[1], times;
    double a;
    // int a_N, b_N, c_N;
    double shift;
    double loc_a, loc_b, loc_c, loc_x, loc_y, loc_z, loc_u;
    double temp_frame_a, temp_frame_b, temp_frame_c;
    double temp_u;
    int i, ii, iii, iiii, j, jj, jjj, k, kk;
    double dis;
    //done!!!!!

    //read input file parameters
	fp1 = fopen(argv[1], "r");
	// fp1 = fopen("AMUWIP_charged.input", "r");
	fgets(str, buffersize, fp1);
    fgets(str, buffersize, fp1);
	// fscanf(fp1,"%d %d %d\n", &Nmaxa, &Nmaxb, &Nmaxc);
	fgets(str, buffersize, fp1);
	fscanf(fp1,"%lf %lf %lf %lf\n", &La, &Lb, &Lc, &dL);
	fgets(str, buffersize, fp1);
	fscanf(fp1,"%lf %lf %lf\n", &alpha, &beta, &gamma);
    alpha_rad = alpha*PI/180;
    beta_rad = beta*PI/180;
    gamma_rad = gamma*PI/180;
	fgets(str, buffersize, fp1);
    fscanf(fp1,"%lf %d %lf %lf %d\n", &cutoff[0], &FH_signal, &total_mass_adsorbate, &temperature[0], &set_running_step);
    // printf("running steps: %d\n", set_running_step);
    //read string calculation setting
    fgets(str, buffersize, fp1);
    fgets(str, buffersize, fp1);
    fscanf(fp1,"%d\n", &direction[0]);
    fgets(str, buffersize, fp1);
    fscanf(fp1,"%d %lf %lf\n", &N_string[0], &move_frac[0], &move_angle_degree[0]);
    fgets(str, buffersize, fp1);
    fscanf(fp1, "%s", conv_string);
    if (strcmp(conv_string, "default")==0)
    {
        set_conv_trans_percent = 30;
        set_conv_rot_percent = 30;
    }
    fgets(str, buffersize, fp1);
    //read adsorbate information
    fgets(str, buffersize, fp1);
    // printf("%s", str);
    fgets(str, buffersize, fp1);
    // printf("%s", str);
    fscanf(fp1,"%d\n", &N_atom_adsorbate[0]);
    // printf("N_atom_adsorbate: %d\n", N_atom_adsorbate[0]);
    double x_adsorbate[N_atom_adsorbate[0]], y_adsorbate[N_atom_adsorbate[0]], z_adsorbate[N_atom_adsorbate[0]];
    double epsilon_adsorbate[N_atom_adsorbate[0]], sigma_adsorbate[N_atom_adsorbate[0]], charge_adsorbate[N_atom_adsorbate[0]], mass_adsorbate[N_atom_adsorbate[0]];
    double vector_adsorbate_x[N_atom_adsorbate[0]], vector_adsorbate_y[N_atom_adsorbate[0]], vector_adsorbate_z[N_atom_adsorbate[0]];
    fgets(str, buffersize, fp1);
    center_of_mass_x[0] = 0;
    center_of_mass_y[0] = 0;
    center_of_mass_z[0] = 0;
    for (i=0; i<N_atom_adsorbate[0]; i++)
    {
        fscanf(fp1,"%lf %lf %lf %lf %lf %lf %lf\n", &x_adsorbate[i], &y_adsorbate[i], &z_adsorbate[i], &epsilon_adsorbate[i], 
            &sigma_adsorbate[i], &charge_adsorbate[i], &mass_adsorbate[i]);
        center_of_mass_x[0] += 1.0*x_adsorbate[i]*mass_adsorbate[i]/total_mass_adsorbate;
        center_of_mass_y[0] += 1.0*y_adsorbate[i]*mass_adsorbate[i]/total_mass_adsorbate;
        center_of_mass_z[0] += 1.0*z_adsorbate[i]*mass_adsorbate[i]/total_mass_adsorbate;
    }
    // printf("center of the mass:\t%lf\t%lf\t%lf\n", center_of_mass_x, center_of_mass_y, center_of_mass_z);
    //determin the vector of each atom with respect to the center of mass
    for (i=0; i<N_atom_adsorbate[0]; i++)
    {
        vector_adsorbate_x[i] = x_adsorbate[i] - center_of_mass_x[0];
        vector_adsorbate_y[i] = y_adsorbate[i] - center_of_mass_y[0];
        vector_adsorbate_z[i] = z_adsorbate[i] - center_of_mass_z[0];
    }
    //read framework information
	fgets(str, buffersize, fp1);
    fgets(str, buffersize, fp1);
	fscanf(fp1,"%d\n", &N_atom_frame[0]);
	fgets(str, buffersize, fp1);

    //frac2car parameter calculation
	double frac2car_a[3];
	double frac2car_b[3];
	double frac2car_c[3];
    frac2car_a[0] = La;
    frac2car_a[1] = Lb*cos(gamma_rad);
    frac2car_a[2] = Lc*cos(beta_rad);
    frac2car_b[0] = 0;
    frac2car_b[1] = Lb*sin(gamma_rad);
    frac2car_b[2] = Lc*( (cos(alpha_rad)-cos(beta_rad)*cos(gamma_rad)) / sin(gamma_rad) );
    frac2car_c[2] = La*Lb*Lc*sqrt( 1 - pow(cos(alpha_rad),2) - pow(cos(beta_rad),2) - pow(cos(gamma_rad),2) + 2*cos(alpha_rad)*cos(beta_rad)*cos(gamma_rad) );
	frac2car_c[2] = frac2car_c[2]/(La*Lb*sin(gamma_rad));
	//done!!!!!

    //expand the cell to the size satisfied cutoff condition
    //convert the fractional cell length to cartesian value;
    frac2car(1, 0, 0, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_x = temp_x[0];
    frac2car(0, 1, 0, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_y = temp_y[0];
    frac2car(0, 0, 1, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_z = temp_z[0];
    times_x[0] = (int) 2*cutoff[0]/cart_x + 1;
    times_y[0] = (int) 2*cutoff[0]/cart_y + 1;
    times_z[0] = (int) 2*cutoff[0]/cart_z + 1;
    times = times_x[0]*times_y[0]*times_z[0];
	double epsilon_frame[N_atom_frame[0]*times], sigma_frame[N_atom_frame[0]*times], charge_frame[N_atom_frame[0]*times], mass_frame[N_atom_frame[0]*times];
	double frac_a_frame[N_atom_frame[0]*times], frac_b_frame[N_atom_frame[0]*times], frac_c_frame[N_atom_frame[0]*times];
    for (i=0; i<N_atom_frame[0]; i++)
	{
		fscanf(fp1,"%lf %lf %lf %lf %lf %lf %lf %lf\n", &a, &sigma_frame[i], &epsilon_frame[i], &charge_frame[i], &mass_frame[i], &frac_a_frame[i], &frac_b_frame[i], &frac_c_frame[i]);
        fgets(str, buffersize, fp1);
    }
    fclose(fp1);
    pbc_expand(N_atom_frame, times_x, times_y, times_z, frac_a_frame, frac_b_frame, frac_c_frame, epsilon_frame, sigma_frame, charge_frame, mass_frame);
    frac2car(times_x[0], 0, 0, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_x_extended[0] = temp_x[0];
    frac2car(0, times_y[0], 0, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_y_extended[0] = temp_y[0];
    frac2car(0, 0, times_z[0], frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_z_extended[0] = temp_z[0];
    //done!!!!

    






    //define variables on device
    double *cart_x_extended_device, *cart_y_extended_device, *cart_z_extended_device;
    double *cutoff_device;
    int *N_atom_adsorbate_device;
    double *epsilon_adsorbate_device, *sigma_adsorbate_device, *charge_adsorbate_device;
    double *center_of_mass_x_device, *center_of_mass_y_device, *center_of_mass_z_device;
    double *vector_adsorbate_x_device, *vector_adsorbate_y_device, *vector_adsorbate_z_device;
    double *temperature_device;
    int *N_atom_frame_device;
    int *times_x_device, *times_y_device, *times_z_device;
    double *epsilon_frame_device, *sigma_frame_device, *charge_frame_device, *mass_frame_device;
    double *frac_a_frame_device, *frac_b_frame_device, *frac_c_frame_device;
    double *frac2car_a_device, *frac2car_b_device, *frac2car_c_device;
    int *direction_device;
    //allocate memory on device
    cudaMalloc((void **)&cart_x_extended_device, sizeof(double));
    cudaMalloc((void **)&cart_y_extended_device, sizeof(double));
    cudaMalloc((void **)&cart_z_extended_device, sizeof(double));
    cudaMalloc((void **)&cutoff_device, sizeof(double));
    cudaMalloc((void **)&N_atom_adsorbate_device, sizeof(int));
    cudaMalloc((void **)&epsilon_adsorbate_device, sizeof(double)*N_atom_adsorbate[0]);
    cudaMalloc((void **)&sigma_adsorbate_device, sizeof(double)*N_atom_adsorbate[0]);
    cudaMalloc((void **)&charge_adsorbate_device, sizeof(double)*N_atom_adsorbate[0]);
    cudaMalloc((void **)&center_of_mass_x_device, sizeof(double));
    cudaMalloc((void **)&center_of_mass_y_device, sizeof(double));
    cudaMalloc((void **)&center_of_mass_z_device, sizeof(double));
    cudaMalloc((void **)&vector_adsorbate_x_device, sizeof(double)*N_atom_adsorbate[0]);
    cudaMalloc((void **)&vector_adsorbate_y_device, sizeof(double)*N_atom_adsorbate[0]);
    cudaMalloc((void **)&vector_adsorbate_z_device, sizeof(double)*N_atom_adsorbate[0]);
    cudaMalloc((void **)&temperature_device, sizeof(double));
    cudaMalloc((void **)&N_atom_frame_device, sizeof(int));
    cudaMalloc((void **)&times_x_device, sizeof(int));
    cudaMalloc((void **)&times_y_device, sizeof(int));
    cudaMalloc((void **)&times_z_device, sizeof(int));
    cudaMalloc((void **)&epsilon_frame_device, sizeof(double)*N_atom_frame[0]*times);
    cudaMalloc((void **)&sigma_frame_device, sizeof(double)*N_atom_frame[0]*times);
    cudaMalloc((void **)&charge_frame_device, sizeof(double)*N_atom_frame[0]*times);
    cudaMalloc((void **)&mass_frame_device, sizeof(double)*N_atom_frame[0]*times);
    cudaMalloc((void **)&frac_a_frame_device, sizeof(double)*N_atom_frame[0]*times);
    cudaMalloc((void **)&frac_b_frame_device, sizeof(double)*N_atom_frame[0]*times);
    cudaMalloc((void **)&frac_c_frame_device, sizeof(double)*N_atom_frame[0]*times);
    cudaMalloc((void **)&frac2car_a_device, sizeof(double)*3);
    cudaMalloc((void **)&frac2car_b_device, sizeof(double)*3);
    cudaMalloc((void **)&frac2car_c_device, sizeof(double)*3);
    cudaMalloc((void**)&direction_device, sizeof(int));





    //define variables for Ewald summation
    double damping_a[1];
    damping_a[0] = 0.2;

    double *damping_a_device;
    cudaMalloc((void **)&damping_a_device, sizeof(double));



    // //copy and transfer arrary concurrently
    // cudaMemcpy(cart_x_extended_device, cart_x_extended, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(cart_y_extended_device, cart_y_extended, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(cart_z_extended_device, cart_z_extended, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(cutoff_device, cutoff, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(N_atom_adsorbate_device, N_atom_adsorbate, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(epsilon_adsorbate_device, epsilon_adsorbate, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice);
    // cudaMemcpy(sigma_adsorbate_device, sigma_adsorbate, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice);
    // cudaMemcpy(center_of_mass_x_device, center_of_mass_x, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(center_of_mass_y_device, center_of_mass_y, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(center_of_mass_z_device, center_of_mass_z, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(vector_adsorbate_x_device, vector_adsorbate_x, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice);
    // cudaMemcpy(vector_adsorbate_y_device, vector_adsorbate_y, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice);
    // cudaMemcpy(vector_adsorbate_z_device, vector_adsorbate_z, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice);
    // cudaMemcpy(temperature_device, temperature, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(N_atom_frame_device, N_atom_frame, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(times_x_device, times_x, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(times_y_device, times_y, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(times_z_device, times_z, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(epsilon_frame_device, epsilon_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice);
    // cudaMemcpy(sigma_frame_device, sigma_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice);



    //copy and transfer arrary asynchronously
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaMemcpyAsync(cart_x_extended_device, cart_x_extended, sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(cart_y_extended_device, cart_y_extended, sizeof(double), cudaMemcpyHostToDevice), stream1;
    cudaMemcpyAsync(cart_z_extended_device, cart_z_extended, sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(cutoff_device, cutoff, sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(N_atom_adsorbate_device, N_atom_adsorbate, sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(epsilon_adsorbate_device, epsilon_adsorbate, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(sigma_adsorbate_device, sigma_adsorbate, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(charge_adsorbate_device, charge_adsorbate, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(center_of_mass_x_device, center_of_mass_x, sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(center_of_mass_y_device, center_of_mass_y, sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(center_of_mass_z_device, center_of_mass_z, sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(vector_adsorbate_x_device, vector_adsorbate_x, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(vector_adsorbate_y_device, vector_adsorbate_y, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(vector_adsorbate_z_device, vector_adsorbate_z, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(temperature_device, temperature, sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(N_atom_frame_device, N_atom_frame, sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(times_x_device, times_x, sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(times_y_device, times_y, sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(times_z_device, times_z, sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(epsilon_frame_device, epsilon_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(sigma_frame_device, sigma_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(charge_frame_device, charge_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(mass_frame_device, mass_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(frac_a_frame_device, frac_a_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(frac_b_frame_device, frac_b_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(frac_c_frame_device, frac_c_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(frac2car_a_device, frac2car_a, sizeof(double)*3, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(frac2car_b_device, frac2car_b, sizeof(double)*3, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(frac2car_c_device, frac2car_c, sizeof(double)*3, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(direction_device, direction, sizeof(int), cudaMemcpyHostToDevice, stream1);


    cudaMemcpyAsync(damping_a_device, damping_a, sizeof(double), cudaMemcpyHostToDevice, stream1);
    // cudaMemcpy(direction_device, direction, sizeof(int), cudaMemcpyHostToDevice);





    //check whether data is properly transferred, uncomment only when it is debugging
    // cudaStreamSynchronize(stream1);
    // check_double<<<1,32>>>(1, cart_x_extended_device);
    // check_double<<<1,32>>>(1, cart_y_extended_device);
    // check_double<<<1,32>>>(1, cart_z_extended_device);
    // check_double<<<1,32>>>(1, cutoff_device);
    // check_int<<<1,32>>>(1, N_atom_adsorbate_device);
    // check_double<<<1,32>>>(N_atom_adsorbate[0], epsilon_adsorbate_device);
    // check_double<<<1,32>>>(N_atom_adsorbate[0], sigma_adsorbate_device);
    // check_double<<<1,32>>>(N_atom_adsorbate[0], charge_adsorbate_device);
    // check_double<<<1,32>>>(1, center_of_mass_x_device);
    // check_double<<<1,32>>>(1, center_of_mass_y_device);
    // check_double<<<1,32>>>(1, center_of_mass_z_device);
    // check_double<<<1,32>>>(N_atom_adsorbate[0], vector_adsorbate_x_device);
    // check_double<<<1,32>>>(N_atom_adsorbate[0], vector_adsorbate_y_device);
    // check_double<<<1,32>>>(N_atom_adsorbate[0], vector_adsorbate_z_device);
    // check_double<<<1,32>>>(1, temperature_device);
    // check_int<<<1,32>>>(1, N_atom_frame_device);
    // check_int<<<1,32>>>(1, times_x_device);
    // check_int<<<1,32>>>(1, times_y_device);
    // check_int<<<1,32>>>(1, times_z_device);
    // check_double<<<1,32>>>(N_atom_frame[0]*times, epsilon_frame_device);
    // check_double<<<1,32>>>(N_atom_frame[0]*times, sigma_frame_device);
    // check_double<<<1,32>>>(N_atom_frame[0]*times, charge_frame_device);
    // check_int<<<1,32>>>(1, direction_device);
    // cudaDeviceSynchronize();
    // return 0;







    clock_t t;





    double rot_alpha_angle, rot_beta_angle, rot_gamma_angle;
    double vector_adsorbate_x_rot[N_atom_adsorbate[0]], vector_adsorbate_y_rot[N_atom_adsorbate[0]], vector_adsorbate_z_rot[N_atom_adsorbate[0]];

    double delta_angle[1];
    // delta_angle[0] = 90;
    delta_angle[0] = 60;
    double delta_grid[1];
    delta_grid[0] = 0.1;
    int N_grid[1], N_angle_alpha[1], N_angle_beta[1], N_angle_gamma[1];
    double *ini_mapping_Vext;
    double double_variable;
    // int direction = 1;
    // direction[0] = 1;
    double local_a, local_b, local_c;
    double local_x, local_y, local_z;
    double local_alpha_angle, local_beta_angle, local_gamma_angle;
    N_grid[0] = (int) (floor(1.0/delta_grid[0])+1);
    N_angle_alpha[0] = (int) (floor(360/delta_angle[0]));
    N_angle_beta[0] = (int) (floor(180/delta_angle[0]));
    N_angle_gamma[0] = (int) (floor(360/delta_angle[0]));
    ini_mapping_Vext = (double *) malloc(sizeof(double_variable)*N_grid[0]*N_grid[0]*N_angle_alpha[0]*N_angle_beta[0]*N_angle_gamma[0]);

    double a_minimum, b_minimum, c_minimum, alpha_minimum_angle, beta_minimum_angle, gamma_minimum_angle;
    double V_min;
    int minimum_signal = 0;


    cudaStreamSynchronize(stream1);





    int N_points = N_grid[0]*N_grid[0]*N_angle_alpha[0]*N_angle_beta[0]*N_angle_gamma[0];




    // solution 1:
    int *N_grid_device, *N_angle_alpha_device, *N_angle_beta_device, *N_angle_gamma_device;
    double *delta_grid_device, *delta_angle_device;
    int *index_a_device, *index_b_device, *index_c_device;
    int *index_alpha_device, *index_beta_device, *index_gamma_device;
    int *index_adsorbate_device, *index_frame_device;
    double *cal_a_device, *cal_b_device, *cal_c_device;
    double *rot_alpha_rad_device, *rot_beta_rad_device, *rot_gamma_rad_device;
    double *loc_x_device, *loc_y_device, *loc_z_device;
    double *vector_adsorbate_x_rot_device, *vector_adsorbate_y_rot_device, *vector_adsorbate_z_rot_device;
    double *adsorbate_cart_x_rot_device, *adsorbate_cart_y_rot_device, *adsorbate_cart_z_rot_device;
    double *modify_frame_a_device, *modify_frame_b_device, *modify_frame_c_device;
    double *minimum_distance_device;
    double *V_total_1;
    // allocate memory
    cudaMalloc((void **)&N_grid_device, sizeof(int));
    cudaMalloc((void **)&N_angle_alpha_device, sizeof(int));
    cudaMalloc((void **)&N_angle_beta_device, sizeof(int));
    cudaMalloc((void **)&N_angle_gamma_device, sizeof(int));
    cudaMalloc((void **)&delta_grid_device, sizeof(double));
    cudaMalloc((void **)&delta_angle_device, sizeof(double));


    cudaMalloc((void **)&index_a_device, sizeof(int)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_b_device, sizeof(int)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_c_device, sizeof(int)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_alpha_device, sizeof(int)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_beta_device, sizeof(int)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_gamma_device, sizeof(int)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_adsorbate_device, sizeof(int)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_frame_device, sizeof(int)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);




    cudaMalloc((void **)&cal_a_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&cal_b_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&cal_c_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&rot_alpha_rad_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&rot_beta_rad_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&rot_gamma_rad_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&loc_x_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&loc_y_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&loc_z_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&vector_adsorbate_x_rot_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&vector_adsorbate_y_rot_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&vector_adsorbate_z_rot_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&adsorbate_cart_x_rot_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&adsorbate_cart_y_rot_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&adsorbate_cart_z_rot_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&modify_frame_a_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&modify_frame_b_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&modify_frame_c_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&minimum_distance_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&V_total_1, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);





    // memory transfer
    cudaMemcpy(N_grid_device, N_grid, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(N_angle_alpha_device, N_angle_alpha, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(N_angle_beta_device, N_angle_beta, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(N_angle_gamma_device, N_angle_gamma, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(delta_grid_device, delta_grid, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(delta_angle_device, delta_angle, sizeof(double), cudaMemcpyHostToDevice);

    int temp_add_frame_host[1];
    temp_add_frame_host[0] = N_atom_frame[0]*times;
    int *temp_add_frame_device;
    cudaMalloc((void **)&temp_add_frame_device, sizeof(int));
    cudaMemcpy(temp_add_frame_device, temp_add_frame_host, sizeof(int), cudaMemcpyHostToDevice);
    // check_int<<<1,32>>>(1, temp_add_frame_device);














    int num_segments = N_points;
    int *h_offset = (int *) malloc(sizeof(int)*(num_segments+1));
    h_offset[0] = 0;
    for (i=1; i<=num_segments; i++)
    {
        h_offset[i] = i*N_atom_adsorbate[0]*N_atom_frame[0]*times;
    }
    int *d_offset;
    cudaMalloc((void**)&d_offset, (num_segments+1)*sizeof(int));
    cudaMemcpy(d_offset, h_offset, (num_segments+1)*sizeof(int), cudaMemcpyHostToDevice);
    free(h_offset);
    double *V_out_test;
    double *V_out_print;
    V_out_print = (double *) malloc(sizeof(double)*num_segments);
    cudaMalloc((void**)&V_out_test, sizeof(double)*num_segments);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;



    // temp_1 = fopen("nohup.out", "w+");


    t = clock();


    Vext_cal<<<(int)((N_points*times*N_atom_adsorbate[0]*N_atom_frame[0]-1)/running_block_size+1),running_block_size>>>

    (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device, 
    charge_adsorbate_device, 
    vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device, 
    N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
    charge_frame_device, 
    frac_a_frame_device, frac_b_frame_device, frac_c_frame_device,
    times_x_device, times_y_device, times_z_device,
    cart_x_extended_device, cart_y_extended_device, cart_z_extended_device,
    frac2car_a_device, frac2car_b_device, frac2car_c_device,
    cutoff_device, damping_a_device, 

                direction_device, 


                N_grid_device, N_angle_alpha_device, N_angle_beta_device, N_angle_gamma_device,
                delta_grid_device, delta_angle_device,
                index_a_device, index_b_device, index_c_device,
                index_alpha_device, index_beta_device, index_gamma_device,
                index_adsorbate_device, index_frame_device,

                cal_a_device, cal_b_device, cal_c_device,
                rot_alpha_rad_device, rot_beta_rad_device, rot_gamma_rad_device,
                loc_x_device, loc_y_device, loc_z_device,
                vector_adsorbate_x_rot_device, vector_adsorbate_y_rot_device, vector_adsorbate_z_rot_device,
                adsorbate_cart_x_rot_device, adsorbate_cart_y_rot_device, adsorbate_cart_z_rot_device, 
                modify_frame_a_device, modify_frame_b_device, modify_frame_c_device,
                minimum_distance_device,
                V_total_1);


    cudaDeviceSynchronize();



    
    // return 0;
    // check_double_custom<<<1,32>>>(times*N_atom_adsorbate[0]*N_atom_frame[0], minimum_distance_device, V_total_1);

    

    // calculate potential energy at each grid by summing over the certain range
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_total_1, V_out_test, 
        num_segments, d_offset, d_offset+1);
    cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_total_1, V_out_test, 
        num_segments, d_offset, d_offset+1);
    cudaFree(d_offset);
    cudaFree(d_temp_storage);
    cudaDeviceSynchronize();
    t = clock() - t;
    // printf("%lf\t", ((double)t)/CLOCKS_PER_SEC);
    // printf("gpu time: %lf\n", ((double)t)/CLOCKS_PER_SEC);
    // cudaMemcpy(V_out_print, V_out_test, (num_segments)*sizeof(double), cudaMemcpyDeviceToHost);
    // printf("%d\n", N_points);


    
    // find the minimum energy value and index for the configuration on the side
    d_temp_storage = NULL;
    cub::KeyValuePair<int, double> *min_value_index_device;
    cudaMalloc((void**)&min_value_index_device, sizeof(cub::KeyValuePair<int, double>));
    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, V_out_test, min_value_index_device, num_segments);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, V_out_test, min_value_index_device, num_segments);
    // check_key<<<1,32>>>(1, min_value_index_device);
    cudaFree(d_temp_storage);

    cudaFree(V_out_test);
    cudaDeviceSynchronize();



    
    // return 0;




    // copy and save the information on the minimized configuration on the side




    


    


    // set up the variable for the string
    int *N_string_device;
    double *s0_a_device, *s0_b_device, *s0_c_device;
    double *s0_alpha_device, *s0_beta_device, *s0_gamma_device;
    double *s1_a_device, *s1_b_device, *s1_c_device;
    double *s1_alpha_device, *s1_beta_device, *s1_gamma_device;
    double *s1_cart_x_device, *s1_cart_y_device, *s1_cart_z_device;
    double *s1_length_coordinate_all_device, *s1_length_orientation_all_device;
    double *s1_length_coordinate_device, *s1_length_orientation_device;
    double *s1_l_abs_coordinate_device, *s1_l_abs_orientation_device;
    double *s1_length_coordinate_remap_device, *s1_length_orientation_remap_device;
    double *s1_length_coordinate_cumulation_device, *s1_length_orientation_cumulation_device;


    double *s2_a_device, *s2_b_device, *s2_c_device;
    double *s2_alpha_device, *s2_beta_device, *s2_gamma_device;
    // double *s2_a_device, *s2_b_device, *s2_c_device;
    double *s2_alpha_smooth_device, *s2_beta_smooth_device, *s2_gamma_smooth_device;


    double *s1_length_device, *s1_length_all_device, *s1_l_abs_device;
    double *s1_legnth_remap_device, *s1_length_cumulation_device;

    int  *index_s0_cal_Vext_s0_device;
    // double *index_a_cal_Vext_s0_device, *index_b_cal_Vext_s0_device, *index_c_cal_Vext_s0_device;
    // double *index_alpha_cal_Vext_s0_device, *index_beta_cal_Vext_s0_device, *index_gamma_cal_Vext_s0_device;
    int *index_adsorbate_cal_Vext_s0_device, *index_frame_cal_Vext_s0_device;

    double *a_cal_Vext_s0_device, *b_cal_Vext_s0_device, *c_cal_Vext_s0_device;
    double *alpha_rad_cal_Vext_s0_device, *beta_rad_cal_Vext_s0_device, *gamma_rad_cal_Vext_s0_device;
    double *loc_x_cal_Vext_s0_device, *loc_y_cal_Vext_s0_device, *loc_z_cal_Vext_s0_device;
    double *vector_adsorbate_x_rot_cal_Vext_s0_device, *vector_adsorbate_y_rot_cal_Vext_s0_device, *vector_adsorbate_z_rot_cal_Vext_s0_device;
    double *adsorbate_cart_x_rot_cal_Vext_s0_device, *adsorbate_cart_y_rot_cal_Vext_s0_device, *adsorbate_cart_z_rot_cal_Vext_s0_device;
    double *modify_frame_a_cal_Vext_s0_device, *modify_frame_b_cal_Vext_s0_device, *modify_frame_c_cal_Vext_s0_device;
    double *minimum_distance_cal_Vext_s0_device;
    double *V_s0_temp, *V_s0;
    double *V_s2;



    // allocate memory
    cudaMalloc((void**)&N_string_device, sizeof(int));
    cudaMalloc((void**)&s0_a_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s0_b_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s0_c_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s0_alpha_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s0_beta_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s0_gamma_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_a_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_b_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_c_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_alpha_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_beta_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_gamma_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_cart_x_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_cart_y_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_cart_z_device, sizeof(double)*N_string[0]);

    cudaMalloc((void**)&s1_length_coordinate_all_device, sizeof(double)*N_string[0]*3);
    cudaMalloc((void**)&s1_length_orientation_all_device, sizeof(double)*N_string[0]*3);
    cudaMalloc((void**)&s1_length_coordinate_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_length_orientation_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_l_abs_coordinate_device, sizeof(double));
    cudaMalloc((void**)&s1_l_abs_orientation_device, sizeof(double));
    cudaMalloc((void**)&s1_length_coordinate_remap_device, sizeof(double)*((int) (N_string[0]*(1+N_string[0])*0.5)));
    cudaMalloc((void**)&s1_length_orientation_remap_device, sizeof(double)*((int) (N_string[0]*(1+N_string[0])*0.5)));
    cudaMalloc((void**)&s1_length_coordinate_cumulation_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_length_orientation_cumulation_device, sizeof(double)*N_string[0]);


    cudaMalloc((void**)&s2_a_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s2_b_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s2_c_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s2_alpha_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s2_beta_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s2_gamma_device, sizeof(double)*N_string[0]);
    // cudaMalloc((void**)&s2_a_device, sizeof(double)*N_string[0]);
    // cudaMalloc((void**)&s2_b_device, sizeof(double)*N_string[0]);
    // cudaMalloc((void**)&s2_c_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s2_alpha_smooth_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s2_beta_smooth_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s2_gamma_smooth_device, sizeof(double)*N_string[0]);






    cudaMalloc((void**)&s1_length_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_length_all_device, sizeof(double)*N_string[0]*6);
    cudaMalloc((void**)&s1_l_abs_device, sizeof(double));
    cudaMalloc((void**)&s1_legnth_remap_device, sizeof(double)*((int) (N_string[0]*(1+N_string[0])*0.5)));
    cudaMalloc((void**)&s1_length_cumulation_device, sizeof(double)*N_string[0]);




    // copy and transfer memory
    cudaMemcpy(N_string_device, N_string, sizeof(int), cudaMemcpyHostToDevice);

    // check_int<<<1,32>>>(1, N_string_device);
    // check_int<<<1,32>>>(1, direction_device);


    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // printf("check\n");
    cudaDeviceSynchronize();
    int signal_straight_line = 1;
    int num_inidividual_ini_extra[1];
    num_inidividual_ini_extra[0] = 5;
    double limit_transition_frac[1], limit_rotation_angle[1];
    limit_transition_frac[0] = 0.15;
    limit_rotation_angle[0] = 0;

    int *i_cal_device;



    
    

    








    // printf("%d\n", argc);
    if (argc==4)
    {
        // there is also input of initital string

        // check the compatibility of the current input string
        fp1 = fopen(argv[2], "r");
        i=0;
        while (1)
        {
            if ( fgets(str, buffersize, fp1) != NULL)
            {
                i++;
            }
            else
            {
                break;
            }
        }
        fclose(fp1);
        if (i==N_string[0])
        {
            double *temp_input_load_a, *temp_input_load_b, *temp_input_load_c;
            double *temp_input_load_alpha, *temp_input_load_beta, *temp_input_load_gamma;

            temp_input_load_a = (double *) malloc(N_string[0]*sizeof(double));
            temp_input_load_b = (double *) malloc(N_string[0]*sizeof(double));
            temp_input_load_c = (double *) malloc(N_string[0]*sizeof(double));
            temp_input_load_alpha = (double *) malloc(N_string[0]*sizeof(double));
            temp_input_load_beta = (double *) malloc(N_string[0]*sizeof(double));
            temp_input_load_gamma = (double *) malloc(N_string[0]*sizeof(double));


            fp1 = fopen(argv[2], "r");
            for (ii=0; ii<i; ii++)
            {
                fscanf(fp1, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf", &temp_input_load_a[ii], &temp_input_load_b[ii], &temp_input_load_c[ii], 
                    &temp_input_load_alpha[ii], &temp_input_load_beta[ii], &temp_input_load_gamma[ii]);
                fgets(str, buffersize, fp1);
                // printf("%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\n", temp_input_load_a[ii], temp_input_load_b[ii], temp_input_load_c[ii], 
                //     temp_input_load_alpha[ii], temp_input_load_beta[ii], temp_input_load_gamma[ii]);
                // printf("%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\n", temp_input_load_a[ii], temp_input_load_b[ii], temp_input_load_c[ii], 
                //     temp_input_load_alpha[ii], temp_input_load_beta[ii], temp_input_load_gamma[ii]);


                // // debug non-stop rotation
                // temp_input_load_alpha[ii] = 0;
                // temp_input_load_beta[ii] = 0;
                // temp_input_load_gamma[ii] = 0;
                
            }
            fclose(fp1);
            cudaMemcpy(s0_a_device, temp_input_load_a, N_string[0]*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(s0_b_device, temp_input_load_b, N_string[0]*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(s0_c_device, temp_input_load_c, N_string[0]*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(s0_alpha_device, temp_input_load_alpha, N_string[0]*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(s0_beta_device, temp_input_load_beta, N_string[0]*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(s0_gamma_device, temp_input_load_gamma, N_string[0]*sizeof(double), cudaMemcpyHostToDevice);

        }
        else
        {
            printf("Warning!!!!\n");
            printf("Incompatible input string!!!\n");
            printf("Wrong line number!!!\n");
        }


    }
    else
    {
        if (signal_straight_line == 1)
        {
            // use straight line throught connecting the minimum energy point along the material
            ini_string_1<<<(int)((N_string[0]-1)/running_block_size+1),running_block_size>>>
            (N_string_device, cal_a_device, cal_b_device, cal_c_device, rot_alpha_rad_device, rot_beta_rad_device, rot_gamma_rad_device, 
                temp_add_frame_device, N_atom_adsorbate_device, direction_device, min_value_index_device, 
                s0_a_device, s0_b_device, s0_c_device, s0_alpha_device, s0_beta_device, s0_gamma_device);
        }
        else if (signal_straight_line == 0)
        {

            int *num_inidividual_ini_extra_device;
            double *ini_minimum_string_a_device, *ini_minimum_string_b_device, *ini_minimum_string_c_device;
            double *ini_minimum_string_alpha_device, *ini_minimum_string_beta_device, *ini_minimum_string_gamma_device;

            double *limit_transition_frac_device, *limit_rotation_angle_device;

            double *temp_partition_device;


            cudaMalloc((void**)&num_inidividual_ini_extra_device, sizeof(int));
            cudaMalloc((void**)&ini_minimum_string_a_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_string_b_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_string_c_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_string_alpha_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_string_beta_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_string_gamma_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&limit_transition_frac_device, sizeof(double));
            cudaMalloc((void**)&limit_rotation_angle_device, sizeof(double));
            cudaMalloc((void**)&i_cal_device, sizeof(double));
            cudaMalloc((void**)&temp_partition_device, sizeof(double)*N_string[0]*6);



            cudaMemcpy(num_inidividual_ini_extra_device, num_inidividual_ini_extra, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(limit_transition_frac_device, limit_transition_frac, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(limit_rotation_angle_device, limit_rotation_angle, sizeof(double), cudaMemcpyHostToDevice);




            copy_ini_upgrade<<<(int)((2*6)/running_block_size+1),running_block_size>>>
            (ini_minimum_string_a_device, ini_minimum_string_b_device, ini_minimum_string_c_device, 
            ini_minimum_string_alpha_device, ini_minimum_string_beta_device, ini_minimum_string_gamma_device, 

            direction_device,

            cal_a_device, cal_b_device, cal_c_device, 
            rot_alpha_rad_device, rot_beta_rad_device, rot_gamma_rad_device, 

            min_value_index_device, temp_add_frame_device, N_atom_adsorbate_device,
            num_inidividual_ini_extra_device);



            delta_angle[0] = 10;
            delta_grid[0] = 0.05;

            N_grid[0] = (int) (floor(2*limit_transition_frac[0]/delta_grid[0])+1);
            N_angle_alpha[0] = (int) (floor(2*limit_rotation_angle[0]/delta_angle[0]+1));
            N_angle_beta[0] = (int) (floor(2*limit_rotation_angle[0]/delta_angle[0]+1));
            N_angle_gamma[0] = (int) (floor(2*limit_rotation_angle[0]/delta_angle[0]+1));
            // N_angle_alpha[0] = (int) (floor(limit_rotation_angle[0]/delta_angle[0]+1));
            // N_angle_beta[0] = (int) (floor(limit_rotation_angle[0]/delta_angle[0]+1));
            // N_angle_gamma[0] = (int) (floor(limit_rotation_angle[0]/delta_angle[0]+1));





            cudaMemcpy(N_grid_device, N_grid, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(N_angle_alpha_device, N_angle_alpha, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(N_angle_beta_device, N_angle_beta, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(N_angle_gamma_device, N_angle_gamma, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(delta_grid_device, delta_grid, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(delta_angle_device, delta_angle, sizeof(double), cudaMemcpyHostToDevice);



            // printf("N_points: %d\n", N_points);
            N_points = N_grid[0]*N_grid[0]*N_angle_alpha[0]*N_angle_beta[0]*N_angle_gamma[0];
            // printf("N_points: %d\n", N_points);
            // printf("N_grid: %d\n", N_grid[0]);
            // printf("N_alpha: %d\n", N_angle_alpha[0]);
            // printf("N_beta: %d\n", N_angle_beta[0]);
            // printf("N_gamma: %d\n", N_angle_gamma[0]);
            int *ini_h_offset = (int *) malloc(sizeof(int)*(N_points+1));
            ini_h_offset[0] = 0;
            for (i=1; i<=N_points; i++)
            {
                ini_h_offset[i] = i*N_atom_adsorbate[0]*N_atom_frame[0]*times;
            }
            int *ini_d_offset;
            cudaMalloc((void**)&ini_d_offset, (N_points+1)*sizeof(int));
            cudaMemcpy(ini_d_offset, ini_h_offset, (N_points+1)*sizeof(int), cudaMemcpyHostToDevice);
            free(ini_h_offset);
            double *V_ini_test;
            cudaMalloc((void**)&V_ini_test, sizeof(double)*N_points);



            for (i=1; i<=num_inidividual_ini_extra[0]; i++)
            {
                cudaMemcpy(i_cal_device, &i, sizeof(int), cudaMemcpyHostToDevice);

                Vext_cal_ini<<<(int)((N_points*times*N_atom_adsorbate[0]*N_atom_frame[0]-1)/running_block_size+1),running_block_size>>>

                (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device,
                charge_adsorbate_device,
                vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
                N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
                charge_frame_device, 
                frac_a_frame_device, frac_b_frame_device, frac_c_frame_device,
                times_x_device, times_y_device, times_z_device,
                cart_x_extended_device, cart_y_extended_device, cart_z_extended_device,
                frac2car_a_device, frac2car_b_device, frac2car_c_device,
                cutoff_device, damping_a_device, 


                            direction_device,


                            N_grid_device, N_angle_alpha_device, N_angle_beta_device, N_angle_gamma_device,
                            delta_grid_device, delta_angle_device,
                            index_a_device, index_b_device, index_c_device,
                            index_alpha_device, index_beta_device, index_gamma_device,
                            index_adsorbate_device, index_frame_device,

                            limit_transition_frac_device, limit_rotation_angle_device,
                            ini_minimum_string_a_device, ini_minimum_string_b_device, ini_minimum_string_c_device, 
                            ini_minimum_string_alpha_device, ini_minimum_string_beta_device, ini_minimum_string_gamma_device,

                            i_cal_device, num_inidividual_ini_extra_device,

                            cal_a_device, cal_b_device, cal_c_device,
                            rot_alpha_rad_device, rot_beta_rad_device, rot_gamma_rad_device,
                            loc_x_device, loc_y_device, loc_z_device,
                            vector_adsorbate_x_rot_device, vector_adsorbate_y_rot_device, vector_adsorbate_z_rot_device,
                            adsorbate_cart_x_rot_device, adsorbate_cart_y_rot_device, adsorbate_cart_z_rot_device, 
                            modify_frame_a_device, modify_frame_b_device, modify_frame_c_device,
                            minimum_distance_device,
                            V_total_1);

                cudaDeviceSynchronize();




                d_temp_storage = NULL;
                cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_total_1, V_ini_test, 
                    N_points, ini_d_offset, ini_d_offset+1);
                cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
                cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_total_1, V_ini_test, 
                    N_points, ini_d_offset, ini_d_offset+1);
                cudaFree(d_temp_storage);
                // check_double<<<1,32>>>(N_points, V_ini_test);
                // check_double<<<1,32>>>(1, V_ini_test);
                d_temp_storage = NULL;
                cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, V_ini_test, min_value_index_device, N_points);
                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, V_ini_test, min_value_index_device, N_points);
                // check_key<<<1,32>>>(1, min_value_index_device);
                cudaFree(d_temp_storage);



                copy_ini_middle_upgrade<<<(int)((1*6)/running_block_size+1),running_block_size>>>
                (ini_minimum_string_a_device, ini_minimum_string_b_device, ini_minimum_string_c_device, 
                ini_minimum_string_alpha_device, ini_minimum_string_beta_device, ini_minimum_string_gamma_device, 

                direction_device,
                
                cal_a_device, cal_b_device, cal_c_device, 
                rot_alpha_rad_device, rot_beta_rad_device, rot_gamma_rad_device, 

                i_cal_device, 

                min_value_index_device, temp_add_frame_device, N_atom_adsorbate_device,
                num_inidividual_ini_extra_device);
            }

            cudaFree(V_ini_test);

            double *ini_minimum_string_cart_x_device, *ini_minimum_string_cart_y_device, *ini_minimum_string_cart_z_device;
            double *ini_minimum_length_all_device, *ini_minimum_length_device, *ini_minimum_l_abs_device;



            cudaMalloc((void**)&ini_minimum_string_cart_x_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_string_cart_y_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_string_cart_z_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_length_all_device, sizeof(double)*(num_inidividual_ini_extra[0]+2)*3);
            cudaMalloc((void**)&ini_minimum_length_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_l_abs_device, sizeof(double)*1);
            // double *ini_length_a_device, *ini_length_b_device, *ini_length_c_device;
            // sizeof(double)*(num_inidividual_ini_extra[0]+2)

            // cudaMalloc((void**)&ini_length_a_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));


            s1_frac2cart_ini<<<(int)(((num_inidividual_ini_extra[0]+2)*3-1)/running_block_size+1),running_block_size>>>
            (num_inidividual_ini_extra_device, 

            frac2car_a_device, frac2car_b_device, frac2car_c_device,

            ini_minimum_string_a_device, ini_minimum_string_b_device, ini_minimum_string_c_device,
            ini_minimum_string_cart_x_device, ini_minimum_string_cart_y_device, ini_minimum_string_cart_z_device);

            // check_double_custom2<<<1,32>>>((num_inidividual_ini_extra[0]+2), ini_minimum_string_a_device, ini_minimum_string_b_device, ini_minimum_string_c_device, 
            // ini_minimum_string_alpha_device, ini_minimum_string_beta_device, ini_minimum_string_gamma_device);


            // check_double_custom2<<<1,32>>>((num_inidividual_ini_extra[0]+2), ini_minimum_string_cart_x_device, ini_minimum_string_cart_y_device, ini_minimum_string_cart_z_device, 
            // ini_minimum_string_alpha_device, ini_minimum_string_beta_device, ini_minimum_string_gamma_device);


            cal_length_prep_ini<<<(int)(((num_inidividual_ini_extra[0]+2)*3-1)/running_block_size+1),running_block_size>>>

            (num_inidividual_ini_extra_device, 

            ini_minimum_string_cart_x_device, ini_minimum_string_cart_y_device, ini_minimum_string_cart_z_device,
            ini_minimum_length_all_device);

            // check_double_ini<<<1,32>>>(((num_inidividual_ini_extra[0]+2)*3), ini_minimum_length_all_device);


            int *add_ini_h_offset = (int *) malloc(sizeof(int)*((num_inidividual_ini_extra[0]+2)+1));
            add_ini_h_offset[0] = 0;
            for (i=1; i<=(num_inidividual_ini_extra[0]+2); i++)
            {
                add_ini_h_offset[i] = i*3;
            }
            int *add_ini_d_offset;
            cudaMalloc((void**)&add_ini_d_offset, ((num_inidividual_ini_extra[0]+2)+1)*sizeof(int));
            cudaMemcpy(add_ini_d_offset, add_ini_h_offset, ((num_inidividual_ini_extra[0]+2)+1)*sizeof(int), cudaMemcpyHostToDevice);
            free(add_ini_h_offset);


            d_temp_storage = NULL;
            cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, ini_minimum_length_all_device, ini_minimum_length_device, 
                (num_inidividual_ini_extra[0]+2), add_ini_d_offset, add_ini_d_offset+1);
            cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
            cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, ini_minimum_length_all_device, ini_minimum_length_device, 
                (num_inidividual_ini_extra[0]+2), add_ini_d_offset, add_ini_d_offset+1);
            cudaFree(d_temp_storage);


            // check_double<<<1,32>>>((num_inidividual_ini_extra[0]+2), ini_minimum_length_device);


            ini_length_sqrt_cal<<<(int)(((num_inidividual_ini_extra[0]+2)-1)/running_block_size+1),running_block_size>>>

            (num_inidividual_ini_extra_device, ini_minimum_length_device);


            // check_double<<<1,32>>>((num_inidividual_ini_extra[0]+2), ini_minimum_length_device);

            int *sum_ini_h_offset = (int *) malloc(sizeof(int)*((1)+1));
            sum_ini_h_offset[0] = 0;
            for (i=1; i<=(1); i++)
            {
                sum_ini_h_offset[i] = (num_inidividual_ini_extra[0]+2);
            }
            int *sum_ini_d_offset;
            cudaMalloc((void**)&sum_ini_d_offset, ((1)+1)*sizeof(int));
            cudaMemcpy(sum_ini_d_offset, sum_ini_h_offset, ((1)+1)*sizeof(int), cudaMemcpyHostToDevice);
            free(sum_ini_h_offset);

            // check_int<<<1,32>>>(2, sum_ini_d_offset);

            d_temp_storage = NULL;
            cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, ini_minimum_length_device, ini_minimum_l_abs_device, 
                (1), sum_ini_d_offset, sum_ini_d_offset+1);
            cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
            cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, ini_minimum_length_device, ini_minimum_l_abs_device, 
                (1), sum_ini_d_offset, sum_ini_d_offset+1);
            cudaFree(d_temp_storage);

            // check_double<<<1,32>>>(1, ini_minimum_l_abs_device);







            ini_2_s0<<<(int)(((N_string[0]*6)-1)/running_block_size+1),running_block_size>>>

            (num_inidividual_ini_extra_device, N_string_device, 


            ini_minimum_l_abs_device, 

            ini_minimum_string_a_device, ini_minimum_string_b_device, ini_minimum_string_c_device, 
            ini_minimum_string_alpha_device, ini_minimum_string_beta_device, ini_minimum_string_gamma_device, 

            temp_partition_device, ini_minimum_length_device, 

            s0_a_device, s0_b_device, s0_c_device, 
            s0_alpha_device, s0_beta_device, s0_gamma_device);










        }
        else
        {
            printf("fatal error!!!!!\n");
            printf("fatal error!!!!!\n");
            printf("fatal error!!!!!\n");
            printf("fatal error!!!!!\n");
            printf("fatal error!!!!!\n");
            printf("fatal error!!!!!\n");
            return 0;
        }
        
    }



    













    // free memory space used to calculate the potential energy on the side
    cudaFree(N_grid_device);
    cudaFree(N_angle_alpha_device);
    cudaFree(N_angle_beta_device);
    cudaFree(N_angle_gamma_device);
    cudaFree(delta_grid_device);
    cudaFree(delta_angle_device);

    cudaFree(index_a_device);
    cudaFree(index_b_device);
    cudaFree(index_c_device);
    cudaFree(index_alpha_device);
    cudaFree(index_beta_device);
    cudaFree(index_gamma_device);
    cudaFree(index_adsorbate_device);
    cudaFree(index_frame_device);
    
    cudaFree(loc_x_device);
    cudaFree(loc_y_device);
    cudaFree(loc_z_device);
    cudaFree(vector_adsorbate_x_rot_device);
    cudaFree(vector_adsorbate_y_rot_device);
    cudaFree(vector_adsorbate_z_rot_device);
    cudaFree(adsorbate_cart_x_rot_device);
    cudaFree(adsorbate_cart_y_rot_device);
    cudaFree(adsorbate_cart_z_rot_device);
    cudaFree(modify_frame_a_device);
    cudaFree(modify_frame_b_device);
    cudaFree(modify_frame_c_device);
    cudaFree(minimum_distance_device);
    cudaFree(V_total_1);





    
    // free memory space used to calculate the potential energy on the side
    cudaFree(cal_a_device);
    cudaFree(cal_b_device);
    cudaFree(cal_c_device);
    cudaFree(rot_alpha_rad_device);
    cudaFree(rot_beta_rad_device);
    cudaFree(rot_gamma_rad_device);



    // calculate energy for the potential along the string without calculating anything extra related to the derivative
    cudaMalloc((void **)&index_s0_cal_Vext_s0_device, sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_adsorbate_cal_Vext_s0_device, sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_frame_cal_Vext_s0_device, sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&a_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&b_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&c_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&alpha_rad_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&beta_rad_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&gamma_rad_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&loc_x_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&loc_y_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&loc_z_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&vector_adsorbate_x_rot_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&vector_adsorbate_y_rot_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&vector_adsorbate_z_rot_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&adsorbate_cart_x_rot_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&adsorbate_cart_y_rot_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&adsorbate_cart_z_rot_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&modify_frame_a_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&modify_frame_b_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&modify_frame_c_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&minimum_distance_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&V_s0_temp, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&V_s0, sizeof(double)*N_string[0]);
    cudaMalloc((void **)&V_s2, sizeof(double)*N_string[0]);





    Vext_cal_s0<<<(int)((N_string[0]*times*N_atom_adsorbate[0]*N_atom_frame[0]-1)/running_block_size+1),running_block_size>>>
    (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device, 
                charge_adsorbate_device, 
    vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
                N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
                charge_frame_device, 
                frac_a_frame_device, frac_b_frame_device, frac_c_frame_device, 
                times_x_device, times_y_device, times_z_device, 
                cart_x_extended_device, cart_y_extended_device, cart_z_extended_device, 
                frac2car_a_device, frac2car_b_device, frac2car_c_device, 
                cutoff_device, damping_a_device, 
                temp_add_frame_device, 


                // int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                // double *delta_grid_device, double *delta_angle_device,
                N_string_device,

                s0_a_device, s0_b_device, s0_c_device, 
                s0_alpha_device, s0_beta_device, s0_gamma_device,


                index_s0_cal_Vext_s0_device,
                // int *index_a_cal_Vext_s0_device, int *index_b_cal_Vext_s0_device, int *index_c_cal_Vext_s0_device,
                // int *index_alpha_cal_Vext_s0_device, int *index_beta_cal_Vext_s0_device, int *index_gamma_cal_Vext_s0_device,
                index_adsorbate_cal_Vext_s0_device, index_frame_cal_Vext_s0_device,

                a_cal_Vext_s0_device, b_cal_Vext_s0_device, c_cal_Vext_s0_device,
                alpha_rad_cal_Vext_s0_device, beta_rad_cal_Vext_s0_device, gamma_rad_cal_Vext_s0_device,
                loc_x_cal_Vext_s0_device, loc_y_cal_Vext_s0_device, loc_z_cal_Vext_s0_device,
                vector_adsorbate_x_rot_cal_Vext_s0_device, vector_adsorbate_y_rot_cal_Vext_s0_device, vector_adsorbate_z_rot_cal_Vext_s0_device,
                adsorbate_cart_x_rot_cal_Vext_s0_device, adsorbate_cart_y_rot_cal_Vext_s0_device, adsorbate_cart_z_rot_cal_Vext_s0_device, 
                modify_frame_a_cal_Vext_s0_device, modify_frame_b_cal_Vext_s0_device, modify_frame_c_cal_Vext_s0_device,
                minimum_distance_cal_Vext_s0_device,
                V_s0_temp);

    h_offset = (int *) malloc(sizeof(int)*(N_string[0]+1));
    h_offset[0] = 0;
    for (i=1; i<=N_string[0]; i++)
    {
        h_offset[i] = i*N_atom_adsorbate[0]*N_atom_frame[0]*times;
    }
    cudaMalloc((void**)&d_offset, (N_string[0]+1)*sizeof(int));
    cudaMemcpy(d_offset, h_offset, (N_string[0]+1)*sizeof(int), cudaMemcpyHostToDevice);
    free(h_offset);

    d_temp_storage = NULL;
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s0, 
        N_string[0], d_offset, d_offset+1);
    cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s0, 
        N_string[0], d_offset, d_offset+1);
    cudaFree(d_temp_storage);
    // cudaFree(d_offset);

    // check_double<<<1,32>>>(N_string[0], V_s0);
    // check_double_custom4<<<1,32>>>(401, s0_a_device, s0_b_device, s0_c_device, s0_alpha_device, s0_beta_device, s0_gamma_device, V_s0);
    // cudaDeviceSynchronize();
    // return 0;


    // double *temp = (double *) malloc(sizeof(double)*N_string[0]);
    // cudaMemcpy(temp, V_s0, (N_string[0])*sizeof(double), cudaMemcpyDeviceToHost);
    // for (i=0; i<N_string[0]; i++)
    // {
    //     printf("%lf\n", temp[i]);
    // }

    // cudaFree(index_s0_cal_Vext_s0_device);
    // cudaFree(index_adsorbate_cal_Vext_s0_device);
    // cudaFree(index_frame_cal_Vext_s0_device);
    // cudaFree(a_cal_Vext_s0_device);
    // cudaFree(b_cal_Vext_s0_device);
    // cudaFree(c_cal_Vext_s0_device);
    // cudaFree(alpha_rad_cal_Vext_s0_device);
    // cudaFree(beta_rad_cal_Vext_s0_device);
    // cudaFree(gamma_rad_cal_Vext_s0_device);
    // cudaFree(loc_x_cal_Vext_s0_device);
    // cudaFree(loc_y_cal_Vext_s0_device);
    // cudaFree(loc_z_cal_Vext_s0_device);
    // cudaFree(vector_adsorbate_x_rot_cal_Vext_s0_device);
    // cudaFree(vector_adsorbate_y_rot_cal_Vext_s0_device);
    // cudaFree(vector_adsorbate_z_rot_cal_Vext_s0_device);
    // cudaFree(adsorbate_cart_x_rot_cal_Vext_s0_device);
    // cudaFree(adsorbate_cart_y_rot_cal_Vext_s0_device);
    // cudaFree(adsorbate_cart_z_rot_cal_Vext_s0_device);
    // cudaFree(modify_frame_a_cal_Vext_s0_device);
    // cudaFree(modify_frame_b_cal_Vext_s0_device);
    // cudaFree(modify_frame_c_cal_Vext_s0_device);
    // cudaFree(minimum_distance_cal_Vext_s0_device);
    // cudaFree(V_s0_temp);
    // cudaFree(V_s0);
    double s0_cart_x[1], s0_cart_y[1], s0_cart_z[1];
    double *s0_a_ini, *s0_b_ini, *s0_c_ini;
    double *s0_alpha_ini, *s0_beta_ini, *s0_gamma_ini;
    double *s0_a_final, *s0_b_final, *s0_c_final;
    double *s0_alpha_final, *s0_beta_final, *s0_gamma_final;
    double *s0_x, *s0_y, *s0_z;
    s0_a_ini = (double *) malloc(sizeof(double)*N_string[0]);
    s0_b_ini = (double *) malloc(sizeof(double)*N_string[0]);
    s0_c_ini = (double *) malloc(sizeof(double)*N_string[0]);
    s0_alpha_ini = (double *) malloc(sizeof(double)*N_string[0]);
    s0_beta_ini = (double *) malloc(sizeof(double)*N_string[0]);
    s0_gamma_ini = (double *) malloc(sizeof(double)*N_string[0]);
    s0_a_final = (double *) malloc(sizeof(double)*N_string[0]);
    s0_b_final = (double *) malloc(sizeof(double)*N_string[0]);
    s0_c_final = (double *) malloc(sizeof(double)*N_string[0]);
    s0_alpha_final = (double *) malloc(sizeof(double)*N_string[0]);
    s0_beta_final = (double *) malloc(sizeof(double)*N_string[0]);
    s0_gamma_final = (double *) malloc(sizeof(double)*N_string[0]);
    s0_x = (double *) malloc(sizeof(double)*N_string[0]);
    s0_y = (double *) malloc(sizeof(double)*N_string[0]);
    s0_z = (double *) malloc(sizeof(double)*N_string[0]);

    cudaMemcpy(s0_a_ini, s0_a_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_b_ini, s0_b_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_c_ini, s0_c_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_alpha_ini, s0_alpha_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_beta_ini, s0_beta_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_gamma_ini, s0_gamma_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);

    double kb = 1.38e-23, T = 300;
    double *V_s0_1, *V_s0_2;
    double *V_s0_treated, *s0;
    double D_1, D_2;
    V_s0_1 = (double *) malloc(sizeof(double)*N_string[0]);
    V_s0_2 = (double *) malloc(sizeof(double)*N_string[0]);
    V_s0_treated = (double *) malloc(sizeof(double)*N_string[0]);
    s0 = (double *) malloc(sizeof(double)*N_string[0]);
    cudaMemcpy(V_s0_1, V_s0, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (i=0; i<N_string[0]; i++)
    {
        frac2car(s0_a_ini[i], s0_b_ini[i], s0_c_ini[i], frac2car_a, frac2car_b, frac2car_c, s0_cart_x, s0_cart_y, s0_cart_z);
        s0_x[i] = s0_cart_x[0]*1e-10;
        s0_y[i] = s0_cart_y[0]*1e-10;
        s0_z[i] = s0_cart_z[0]*1e-10;
        // printf("%.5e %.5e %.5e\n", s0_a_ini[i], s0_b_ini[i], s0_c_ini[i]);
        // printf("%.5e %.5e %.5e\n", s0_x[i], s0_y[i], s0_z[i]);
        // printf("%lf %lf %lf %lf %lf %lf\n", s0_a_ini[i], s0_b_ini[i], s0_c_ini[i], s0_alpha_ini[i], s0_beta_ini[i], s0_gamma_ini[i]);
    }
    for (i=0; i<N_string[0]; i++)
    {
        if (i==0)
        {
            s0[i] = 0;
        }
        else
        {
            s0[i] = s0[i-1] + sqrt( pow((s0_x[i]-s0_x[i-1]), 2) + pow((s0_y[i]-s0_y[i-1]), 2) + pow((s0_z[i]-s0_z[i-1]), 2) );
        }
         // = 1.0*i/(N_string[0]-1)*1e-10;
        if ((V_s0_1[i]/T)>6e2)
        {
            V_s0_treated[i] = exp(-6e2);
        }
        else
        {
            V_s0_treated[i] = exp(-V_s0_1[i]/T);
        }
        // printf("%.5e\n", V_s0_1[i]);
        
    }
    // printf("length: %.5e\n", s0[N_string[0]-1]);
    switch (direction[0])
    {
        case 1:
            D_1 = 0.5 * pow((La*1e-10), 2) * sqrt((kb*T)/(2*PI*(total_mass_adsorbate/1e3/6.02214076e23))) *  
            (  exp( -max(V_s0_1, N_string[0])/T ) / trapz(s0, V_s0_treated, N_string[0]) );
            break;
        case 2:
            D_1 = 0.5 * pow((Lb*1e-10), 2) * sqrt((kb*T)/(2*PI*(total_mass_adsorbate/1e3/6.02214076e23))) *  
            (  exp( -max(V_s0_1, N_string[0])/T ) / trapz(s0, V_s0_treated, N_string[0]) );
            break;
        case 3:
            D_1 = 0.5 * pow((Lc*1e-10), 2) * sqrt((kb*T)/(2*PI*(total_mass_adsorbate/1e3/6.02214076e23))) *  
            (  exp( -max(V_s0_1, N_string[0])/T ) / trapz(s0, V_s0_treated, N_string[0]) );
            break;
    }





    // remap the string to the set that can calculate the partial derivative
    double *s0_deri_a_device, *s0_deri_b_device, *s0_deri_c_device;
    double *s0_deri_alpha_device, *s0_deri_beta_device, *s0_deri_gamma_device;
    int *s0_deri_index_string_device, *s0_deri_index_var_device;
    int *s0_deri_index_adsorbate_device, *s0_deri_index_frame_device;

    double *s0_deri_loc_x_device, *s0_deri_loc_y_device, *s0_deri_loc_z_device;
    double *s0_deri_vector_adsorbate_x_rot_device, *s0_deri_vector_adsorbate_y_rot_device, *s0_deri_vector_adsorbate_z_rot_device;
    double *s0_deri_adsorbate_cart_x_rot_device, *s0_deri_adsorbate_cart_y_rot_device, *s0_deri_adsorbate_cart_z_rot_device;
    double *s0_deri_modify_frame_a_device, *s0_deri_modify_frame_b_device, *s0_deri_modify_frame_c_device;
    double *s0_deri_minimum_distance_device;
    double *s0_deri_total_Vext_device, *s0_deri_Vext_device;

    double *s0_gradient_device, *s0_gradient_square_device, *s0_gradient_length_device;

    



    double *diff_s_coordinate_all_device, *diff_s_orientation_all_device;
    double *diff_s_coordinate_device, *diff_s_orientation_device;
    double *total_diff_s_coordinate_device, *total_diff_s_orientation_device;


    // double

    // double *a_s0_cal_device, *b_s0_cal_device, *c_s0_cal_device;
    // double *alpha_rad_s0_device, *beta_rad_s0_device, *gamma_rad_s0_device;




    cudaMalloc((void **)&s0_deri_a_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_b_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_c_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_alpha_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_beta_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_gamma_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_index_string_device, sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_index_var_device, sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_index_adsorbate_device, sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_index_frame_device, sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_loc_x_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_loc_y_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_loc_z_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_vector_adsorbate_x_rot_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_vector_adsorbate_y_rot_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_vector_adsorbate_z_rot_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_adsorbate_cart_x_rot_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_adsorbate_cart_y_rot_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_adsorbate_cart_z_rot_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_modify_frame_a_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_modify_frame_b_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_modify_frame_c_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_minimum_distance_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_total_Vext_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_Vext_device, sizeof(double)*N_string[0]*7);
    cudaMalloc((void **)&s0_gradient_device, sizeof(double)*N_string[0]*6);
    cudaMalloc((void **)&s0_gradient_square_device, sizeof(double)*N_string[0]*6);
    cudaMalloc((void **)&s0_gradient_length_device, sizeof(double)*N_string[0]*2);



    cudaMalloc((void **)&diff_s_coordinate_all_device, sizeof(double)*N_string[0]*3);
    cudaMalloc((void **)&diff_s_orientation_all_device, sizeof(double)*N_string[0]*3);

    cudaMalloc((void **)&diff_s_coordinate_device, sizeof(double)*N_string[0]);
    cudaMalloc((void **)&diff_s_orientation_device, sizeof(double)*N_string[0]);
    cudaMalloc((void **)&total_diff_s_coordinate_device, sizeof(double));
    cudaMalloc((void **)&total_diff_s_orientation_device, sizeof(double));



    // parameter used for string method
    double rounding_coeff[1];
    double smooth_coeff[1];
    double *move_angle_rad_device, *move_frac_device;
    double *rounding_coeff_device;
    double *smooth_coeff_device;
    cudaMalloc((void **)&move_angle_rad_device, sizeof(double));
    cudaMalloc((void **)&move_frac_device, sizeof(double));
    cudaMalloc((void **)&rounding_coeff_device, sizeof(double));
    cudaMalloc((void **)&smooth_coeff_device, sizeof(double));
    // move_angle_degree[0] = 1.0;
    // move_frac[0] = 1e-4;
    move_angle_rad[0] = 1.0*move_angle_degree[0]/180*PI;
    rounding_coeff[0] = 1e-15;
    smooth_coeff[0] = 1e-4;

    cudaMemcpy(move_angle_rad_device, move_angle_rad, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(move_frac_device, move_frac, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rounding_coeff_device, rounding_coeff, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(smooth_coeff_device, smooth_coeff, sizeof(double), cudaMemcpyHostToDevice);
    // check_double<<<1,32>>>(1, move_angle_rad_device);
    // check_double<<<1,32>>>(1, move_frac_device);
    // check_double_sci<<<1,32>>>(1, rounding_coeff_device);
    // check_double_sci<<<1,32>>>(1, smooth_coeff_device);
    // cudaDeviceSynchronize();
    // return 0;



    int *V_deri_offset = (int *) malloc(sizeof(int)*(N_string[0]*7+1));
    V_deri_offset[0] = 0;
    for (i=1; i<=(N_string[0]*7); i++)
    {
        V_deri_offset[i] = i*N_atom_adsorbate[0]*N_atom_frame[0]*times;
    }
    int *V_deri_offset_device;
    cudaMalloc((void**)&V_deri_offset_device, (N_string[0]*7+1)*sizeof(int));
    cudaMemcpy(V_deri_offset_device, V_deri_offset, (N_string[0]*7+1)*sizeof(int), cudaMemcpyHostToDevice);
    free(V_deri_offset);

    int *s0_gradient_offset = (int *) malloc(sizeof(int)*(N_string[0]*2+1));
    s0_gradient_offset[0] = 0;
    for (i=1; i<=(N_string[0]*2); i++)
    {
        s0_gradient_offset[i] = i*3;
    }
    int *s0_gradient_offset_device;
    cudaMalloc((void**)&s0_gradient_offset_device, sizeof(int)*(N_string[0]*2+1));
    cudaMemcpy(s0_gradient_offset_device, s0_gradient_offset, sizeof(int)*(N_string[0]*2+1), cudaMemcpyHostToDevice);
    free(s0_gradient_offset);


    int *s1_l_sum_1_offset = (int *) malloc(sizeof(int)*(N_string[0]+1));
    s1_l_sum_1_offset[0] = 0;
    for (i=1; i<=(N_string[0]); i++)
    {
        s1_l_sum_1_offset[i] = i*6;
    }
    int *s1_l_sum_1_offset_device;
    cudaMalloc((void**)&s1_l_sum_1_offset_device, sizeof(int)*(N_string[0]+1));
    cudaMemcpy(s1_l_sum_1_offset_device, s1_l_sum_1_offset, sizeof(int)*(N_string[0]+1), cudaMemcpyHostToDevice);
    free(s1_l_sum_1_offset);

    int *s1_l_sum_2_offset = (int *) malloc(sizeof(int)*2);
    s1_l_sum_2_offset[0] = 0;
    s1_l_sum_2_offset[1] = N_string[0];
    int *s1_l_sum_2_offset_device;
    cudaMalloc((void**)&s1_l_sum_2_offset_device, sizeof(int)*2);
    cudaMemcpy(s1_l_sum_2_offset_device, s1_l_sum_2_offset, sizeof(int)*2, cudaMemcpyHostToDevice);
    free(s1_l_sum_2_offset);

    // int *s1_l_cumulation_offset = (int *) malloc(sizeof(int)*(N_string[0]+1));
    // s1_l_cumulation_offset[0] = 0;
    // for (i=1; i<=(N_string[0]); i++)
    // {
    //     s1_l_cumulation_offset[i] = ((int) (i*(i+1)*0.5));
    // }
    // int *s1_l_cumulation_offset_device;
    // cudaMalloc((void**)&s1_l_cumulation_offset_device, sizeof(int)*(N_string[0]+1));
    // cudaMemcpy(s1_l_cumulation_offset_device, s1_l_cumulation_offset, sizeof(int)*(N_string[0]+1), cudaMemcpyHostToDevice);
    // free(s1_l_cumulation_offset);








    int *s1_l_sum_separate_offset = (int *) malloc(sizeof(int)*(N_string[0]+1));
    s1_l_sum_separate_offset[0] = 0;
    for (i=1; i<=(N_string[0]); i++)
    {
        s1_l_sum_separate_offset[i] = i*3;
    }
    int *s1_l_sum_separate_offset_device;
    cudaMalloc((void**)&s1_l_sum_separate_offset_device, sizeof(int)*(N_string[0]+1));
    cudaMemcpy(s1_l_sum_separate_offset_device, s1_l_sum_separate_offset, sizeof(int)*(N_string[0]+1), cudaMemcpyHostToDevice);
    free(s1_l_sum_separate_offset);



    int *s1_l_sum_total_offset = (int *) malloc(sizeof(int)*2);
    s1_l_sum_total_offset[0] = 0;
    s1_l_sum_total_offset[1] = N_string[0];
    int *s1_l_sum_total_offset_device;
    cudaMalloc((void**)&s1_l_sum_total_offset_device, sizeof(int)*2);
    cudaMemcpy(s1_l_sum_total_offset_device, s1_l_sum_total_offset, sizeof(int)*2, cudaMemcpyHostToDevice);
    free(s1_l_sum_total_offset);



    int *s1_l_cumulation_offset = (int *) malloc(sizeof(int)*(N_string[0]+1));
    s1_l_cumulation_offset[0] = 0;
    for (i=1; i<=(N_string[0]); i++)
    {
        s1_l_cumulation_offset[i] = ((int) (i*(i+1)*0.5));
    }
    int *s1_l_cumulation_offset_device;
    cudaMalloc((void**)&s1_l_cumulation_offset_device, sizeof(int)*(N_string[0]+1));
    cudaMemcpy(s1_l_cumulation_offset_device, s1_l_cumulation_offset, sizeof(int)*(N_string[0]+1), cudaMemcpyHostToDevice);
    free(s1_l_cumulation_offset);






    double convergence_coorindate[1];
    double convergence_orientation[1];
    int signal_coordinate[1];
    int signal_orientation[1];
    // printf("check: %lf %lf\n", 1.0*set_conv_trans_percent/100, 1.0*set_conv_rot_percent/100);
    convergence_coorindate[0] = 1.0*set_conv_trans_percent/100*move_frac[0]*N_string[0]*sqrt(3);
    convergence_orientation[0] = 1.0*set_conv_rot_percent/100*move_angle_rad[0]*N_string[0]*sqrt(3);
    signal_coordinate[0] = 0;
    signal_orientation[0] = 0;

    double *convergence_coorindate_device;
    double *convergence_orientation_device;
    int *signal_coordinate_device;
    int *signal_orientation_device;
    cudaMalloc((void **)&convergence_coorindate_device, sizeof(double));
    cudaMalloc((void **)&convergence_orientation_device, sizeof(double));
    cudaMalloc((void **)&signal_coordinate_device, sizeof(int));
    cudaMalloc((void **)&signal_orientation_device, sizeof(int));
    cudaMemcpy(convergence_coorindate_device, convergence_coorindate, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(convergence_orientation_device, convergence_orientation, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(signal_coordinate_device, signal_coordinate, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(signal_orientation_device, signal_orientation, sizeof(int), cudaMemcpyHostToDevice);
    // check_double<<<1,32>>>(1, convergence_coorindate_device);
    // check_double<<<1,32>>>(1, convergence_orientation_device);
    // check_int<<<1,32>>>(1, signal_coordinate_device);
    // check_int<<<1,32>>>(1, signal_orientation_device);
    // cudaDeviceSynchronize();
    // return 0;










    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    cudaDeviceSynchronize();
    // printf("start\n");
    t = clock();

    int time_set = set_running_step;
    int i_time;
    for (i_time=0; i_time<time_set; i_time++)
    {
        remap_string_var<<<(int)((N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7-1)/running_block_size+1),running_block_size>>>
        (N_atom_adsorbate_device, temp_add_frame_device,

                    N_string_device,

                    s0_a_device, s0_b_device, s0_c_device,
                    s0_alpha_device, s0_beta_device, s0_gamma_device,



                    s0_deri_a_device, s0_deri_b_device, s0_deri_c_device, 
                    s0_deri_alpha_device, s0_deri_beta_device, s0_deri_gamma_device,

                    s0_deri_index_string_device, s0_deri_index_var_device,
                    s0_deri_index_adsorbate_device, s0_deri_index_frame_device,


                    move_angle_rad_device, move_frac_device);







        Vext_s0_deri_cal<<<(int)((N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7-1)/running_block_size+1),running_block_size>>>
        (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device, 
                    charge_adsorbate_device,
        vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
                    N_atom_frame_device, epsilon_frame_device, sigma_frame_device,  
                    charge_frame_device, 
                    frac_a_frame_device, frac_b_frame_device, frac_c_frame_device,
                    times_x_device, times_y_device, times_z_device,
                    cart_x_extended_device, cart_y_extended_device, cart_z_extended_device,
                    frac2car_a_device, frac2car_b_device, frac2car_c_device, 
                    cutoff_device, damping_a_device, 
                    temp_add_frame_device,

                    N_string_device,

                    s0_deri_a_device, s0_deri_b_device, s0_deri_c_device,
                    s0_deri_alpha_device, s0_deri_beta_device, s0_deri_gamma_device,

                    s0_deri_index_adsorbate_device, s0_deri_index_frame_device,

                    s0_deri_loc_x_device, s0_deri_loc_y_device, s0_deri_loc_z_device,
                    s0_deri_vector_adsorbate_x_rot_device, s0_deri_vector_adsorbate_y_rot_device, s0_deri_vector_adsorbate_z_rot_device,
                    s0_deri_adsorbate_cart_x_rot_device, s0_deri_adsorbate_cart_y_rot_device, s0_deri_adsorbate_cart_z_rot_device,
                    s0_deri_modify_frame_a_device, s0_deri_modify_frame_b_device, s0_deri_modify_frame_c_device,
                    s0_deri_minimum_distance_device,

                    s0_deri_total_Vext_device);



        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s0_deri_total_Vext_device, s0_deri_Vext_device, 
            N_string[0]*7, V_deri_offset_device, V_deri_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s0_deri_total_Vext_device, s0_deri_Vext_device, 
            N_string[0]*7, V_deri_offset_device, V_deri_offset_device+1);
        cudaFree(d_temp_storage);







        s0_grad_cal<<<(int)((N_string[0]*6-1)/running_block_size+1),running_block_size>>>
        (move_frac_device, move_angle_rad_device, rounding_coeff_device,
        N_string_device, s0_deri_Vext_device, s0_gradient_device, s0_gradient_square_device);

        // double *temp;
        // temp = (double *) malloc(sizeof(double)*N_string[0]*6);
        // cudaMemcpy(temp, s0_gradient_device, (N_string[0]*6)*sizeof(double), cudaMemcpyDeviceToHost);
        // for (i=0; i<N_string[0]; i++)
        // {
        //     printf("%lf %lf %lf %lf %lf %lf\n", temp[i*6+0], temp[i*6+1], temp[i*6+2], temp[i*6+3], temp[i*6+4], temp[i*6+5]);
        // }
        // return 0;


        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s0_gradient_square_device, s0_gradient_length_device, 
            N_string[0]*2, s0_gradient_offset_device, s0_gradient_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s0_gradient_square_device, s0_gradient_length_device, 
            N_string[0]*2, s0_gradient_offset_device, s0_gradient_offset_device+1);
        cudaFree(d_temp_storage);



        s0_grad_length_sqrt_cal<<<(int)((N_string[0]*2-1)/running_block_size+1),running_block_size>>>
        (rounding_coeff_device, N_string_device, s0_gradient_length_device);

        // double *temp;
        // temp = (double *) malloc(sizeof(double)*N_string[0]*2);
        // cudaMemcpy(temp, s0_gradient_length_device, (N_string[0]*2)*sizeof(double), cudaMemcpyDeviceToHost);
        // for (i=0; i<N_string[0]; i++)
        // {
        //     printf("%lf %lf\n", temp[i*2+0], temp[i*2+1]);
        // }
        // return 0;





        s0_new_cal<<<(int)((N_string[0]*6-1)/running_block_size+1),running_block_size>>>
        (rounding_coeff_device, N_string_device, 
        move_frac_device, move_angle_rad_device,




        s0_gradient_length_device, s0_gradient_device,
        s0_a_device, s0_b_device, s0_c_device, 
        s0_alpha_device, s0_beta_device, s0_gamma_device,
        s1_a_device, s1_b_device, s1_c_device,
        s1_alpha_device, s1_beta_device, s1_gamma_device);








        // check_double_custom2<<<1,32>>>(301, s1_a_device, s1_b_device, s1_c_device, s1_alpha_device, s1_beta_device, s1_gamma_device);
        // cudaDeviceSynchronize();
        // return 0;


        s1_fix_modify_upgrade<<<(int)((N_string[0]-1)/running_block_size+1),running_block_size>>>
        (N_string_device, 

        direction_device,

        s0_gradient_length_device, s0_gradient_device,
        s0_a_device, s0_b_device, s0_c_device, 
        s0_alpha_device, s0_beta_device, s0_gamma_device,
        s1_a_device, s1_b_device, s1_c_device,
        s1_alpha_device, s1_beta_device, s1_gamma_device);


        s1_frac2cart<<<(int)((N_string[0]-1)/running_block_size+1),running_block_size>>>
        (N_string_device, 

        frac2car_a_device, frac2car_b_device, frac2car_c_device,

        s1_a_device, s1_b_device, s1_c_device,
        s1_cart_x_device, s1_cart_y_device, s1_cart_z_device);



        // check_double_custom2<<<1,32>>>(301, s1_cart_x_device, s1_cart_y_device, s1_cart_z_device, s1_alpha_device, s1_beta_device, s1_gamma_device);
        // check_double_custom2<<<1,32>>>(401, s1_a_device, s1_b_device, s1_c_device, s1_alpha_device, s1_beta_device, s1_gamma_device);
        // cudaDeviceSynchronize();
        // return 0;


        
        s1_length_prep<<<(int)((N_string[0]*6-1)/running_block_size+1),running_block_size>>>
        (N_string_device, 

        s1_cart_x_device, s1_cart_y_device, s1_cart_z_device,
        s1_alpha_device, s1_beta_device, s1_gamma_device,
        s1_length_coordinate_all_device, s1_length_orientation_all_device);



        // double *temp1, *temp2;
        // temp1 = (double *) malloc(sizeof(double)*N_string[0]*3);
        // temp2 = (double *) malloc(sizeof(double)*N_string[0]*3);
        // cudaMemcpy(temp1, s1_length_coordinate_all_device, (N_string[0]*3)*sizeof(double), cudaMemcpyDeviceToHost);
        // cudaMemcpy(temp2, s1_length_orientation_all_device, (N_string[0]*3)*sizeof(double), cudaMemcpyDeviceToHost);
        // for (i=0; i<N_string[0]; i++)
        // {
        //     printf("%.3e %.3e %.3e %.3e %.3e %.3e\n", temp1[i*3+0], temp1[i*3+1], temp1[i*3+2], temp2[i*3+0], temp2[i*3+1], temp2[i*3+2]);
        // }
        // return 0;



        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_coordinate_all_device, s1_length_coordinate_device, 
            N_string[0], s1_l_sum_separate_offset_device, s1_l_sum_separate_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_coordinate_all_device, s1_length_coordinate_device, 
            N_string[0], s1_l_sum_separate_offset_device, s1_l_sum_separate_offset_device+1);
        cudaFree(d_temp_storage);



        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_orientation_all_device, s1_length_orientation_device, 
            N_string[0], s1_l_sum_separate_offset_device, s1_l_sum_separate_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_orientation_all_device, s1_length_orientation_device, 
            N_string[0], s1_l_sum_separate_offset_device, s1_l_sum_separate_offset_device+1);
        cudaFree(d_temp_storage);



        // check_double_custom<<<1,32>>>(301, s1_length_coordinate_device, s1_length_orientation_device);
        // cudaDeviceSynchronize();
        // return 0;



        s1_length_sqrt_cal<<<(int)((N_string[0]*2-1)/running_block_size+1),running_block_size>>>
        (rounding_coeff_device, N_string_device, s1_length_coordinate_device, s1_length_orientation_device);



        // check_double_custom<<<1,32>>>(301, s1_length_coordinate_device, s1_length_orientation_device);
        // cudaDeviceSynchronize();
        // return 0;


        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_coordinate_device, s1_l_abs_coordinate_device, 
            1, s1_l_sum_total_offset_device, s1_l_sum_total_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_coordinate_device, s1_l_abs_coordinate_device, 
            1, s1_l_sum_total_offset_device, s1_l_sum_total_offset_device+1);
        cudaFree(d_temp_storage);



        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_orientation_device, s1_l_abs_orientation_device, 
            1, s1_l_sum_total_offset_device, s1_l_sum_total_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_orientation_device, s1_l_abs_orientation_device, 
            1, s1_l_sum_total_offset_device, s1_l_sum_total_offset_device+1);
        cudaFree(d_temp_storage);



        // check_double_custom<<<1,32>>>(1, s1_l_abs_coordinate_device, s1_l_abs_orientation_device);
        // cudaDeviceSynchronize();
        // return 0;



        remap_s1_length_for_cumulation<<<(int)(((N_string[0]*(1+N_string[0])*0.5*2)-1)/running_block_size+1),running_block_size>>>
        (N_string_device, s1_length_coordinate_device, s1_length_orientation_device, 

        s1_length_coordinate_remap_device, s1_length_orientation_remap_device);



        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_coordinate_remap_device, s1_length_coordinate_cumulation_device, 
            N_string[0], s1_l_cumulation_offset_device, s1_l_cumulation_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_coordinate_remap_device, s1_length_coordinate_cumulation_device, 
            N_string[0], s1_l_cumulation_offset_device, s1_l_cumulation_offset_device+1);
        cudaFree(d_temp_storage);



        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_orientation_remap_device, s1_length_orientation_cumulation_device, 
            N_string[0], s1_l_cumulation_offset_device, s1_l_cumulation_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_orientation_remap_device, s1_length_orientation_cumulation_device, 
            N_string[0], s1_l_cumulation_offset_device, s1_l_cumulation_offset_device+1);
        cudaFree(d_temp_storage);



        // check_double_custom<<<1,32>>>(301, s1_length_coordinate_cumulation_device, s1_length_orientation_cumulation_device);
        // cudaDeviceSynchronize();
        // return 0;











        s1_2_s2<<<(int)((N_string[0]*2-1)/running_block_size+1),running_block_size>>>
        (N_string_device, s1_l_abs_coordinate_device, s1_l_abs_orientation_device,
        s1_length_coordinate_cumulation_device, s1_length_orientation_cumulation_device, 

        s1_a_device, s1_b_device, s1_c_device, 
        s1_alpha_device, s1_beta_device, s1_gamma_device,

        s2_a_device, s2_b_device, s2_c_device,
        s2_alpha_device, s2_beta_device, s2_gamma_device);

        // check_double_temp<<<1,32>>>(401, s2_a_device, s2_b_device, s2_c_device);
        // check_double_temp<<<1,32>>>(401, s2_alpha_device, s2_beta_device, s2_gamma_device);
        // check_double_custom2<<<1,32>>>(401, s1_a_device, s1_b_device, s1_c_device, s1_alpha_device, s1_beta_device, s1_gamma_device);
        // cudaDeviceSynchronize();
        // return 0;

        // check_double_custom2<<<1,32>>>(401, s0_a_device, s0_b_device, s0_c_device, s0_alpha_device, s0_beta_device, s0_gamma_device);
        // cudaDeviceSynchronize();
        // return 0;

        // calculate the potential of initial string
        Vext_cal_s0<<<(int)((N_string[0]*times*N_atom_adsorbate[0]*N_atom_frame[0]-1)/running_block_size+1),running_block_size>>>
        (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device, 
                    charge_adsorbate_device, 
        vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
                    N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
                    charge_frame_device, 
                    frac_a_frame_device, frac_b_frame_device, frac_c_frame_device, 
                    times_x_device, times_y_device, times_z_device, 
                    cart_x_extended_device, cart_y_extended_device, cart_z_extended_device, 
                    frac2car_a_device, frac2car_b_device, frac2car_c_device, 
                    cutoff_device, damping_a_device, 
                    temp_add_frame_device, 


                    // int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                    // double *delta_grid_device, double *delta_angle_device,
                    N_string_device,

                    s0_a_device, s0_b_device, s0_c_device, 
                    s0_alpha_device, s0_beta_device, s0_gamma_device,


                    index_s0_cal_Vext_s0_device,
                    // int *index_a_cal_Vext_s0_device, int *index_b_cal_Vext_s0_device, int *index_c_cal_Vext_s0_device,
                    // int *index_alpha_cal_Vext_s0_device, int *index_beta_cal_Vext_s0_device, int *index_gamma_cal_Vext_s0_device,
                    index_adsorbate_cal_Vext_s0_device, index_frame_cal_Vext_s0_device,

                    a_cal_Vext_s0_device, b_cal_Vext_s0_device, c_cal_Vext_s0_device,
                    alpha_rad_cal_Vext_s0_device, beta_rad_cal_Vext_s0_device, gamma_rad_cal_Vext_s0_device,
                    loc_x_cal_Vext_s0_device, loc_y_cal_Vext_s0_device, loc_z_cal_Vext_s0_device,
                    vector_adsorbate_x_rot_cal_Vext_s0_device, vector_adsorbate_y_rot_cal_Vext_s0_device, vector_adsorbate_z_rot_cal_Vext_s0_device,
                    adsorbate_cart_x_rot_cal_Vext_s0_device, adsorbate_cart_y_rot_cal_Vext_s0_device, adsorbate_cart_z_rot_cal_Vext_s0_device, 
                    modify_frame_a_cal_Vext_s0_device, modify_frame_b_cal_Vext_s0_device, modify_frame_c_cal_Vext_s0_device,
                    minimum_distance_cal_Vext_s0_device,
                    V_s0_temp);
        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s0, 
            N_string[0], d_offset, d_offset+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s0, 
            N_string[0], d_offset, d_offset+1);
        cudaFree(d_temp_storage);

        // calculate the potential of moved string
        Vext_cal_s0<<<(int)((N_string[0]*times*N_atom_adsorbate[0]*N_atom_frame[0]-1)/running_block_size+1),running_block_size>>>
        (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device, 
                    charge_adsorbate_device, 
        vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
                    N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
                    charge_frame_device, 
                    frac_a_frame_device, frac_b_frame_device, frac_c_frame_device, 
                    times_x_device, times_y_device, times_z_device, 
                    cart_x_extended_device, cart_y_extended_device, cart_z_extended_device, 
                    frac2car_a_device, frac2car_b_device, frac2car_c_device, 
                    cutoff_device, damping_a_device, 
                    temp_add_frame_device, 


                    // int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                    // double *delta_grid_device, double *delta_angle_device,
                    N_string_device,

                    s2_a_device, s2_b_device, s2_c_device, 
                    s2_alpha_device, s2_beta_device, s2_gamma_device,


                    index_s0_cal_Vext_s0_device,
                    // int *index_a_cal_Vext_s0_device, int *index_b_cal_Vext_s0_device, int *index_c_cal_Vext_s0_device,
                    // int *index_alpha_cal_Vext_s0_device, int *index_beta_cal_Vext_s0_device, int *index_gamma_cal_Vext_s0_device,
                    index_adsorbate_cal_Vext_s0_device, index_frame_cal_Vext_s0_device,

                    a_cal_Vext_s0_device, b_cal_Vext_s0_device, c_cal_Vext_s0_device,
                    alpha_rad_cal_Vext_s0_device, beta_rad_cal_Vext_s0_device, gamma_rad_cal_Vext_s0_device,
                    loc_x_cal_Vext_s0_device, loc_y_cal_Vext_s0_device, loc_z_cal_Vext_s0_device,
                    vector_adsorbate_x_rot_cal_Vext_s0_device, vector_adsorbate_y_rot_cal_Vext_s0_device, vector_adsorbate_z_rot_cal_Vext_s0_device,
                    adsorbate_cart_x_rot_cal_Vext_s0_device, adsorbate_cart_y_rot_cal_Vext_s0_device, adsorbate_cart_z_rot_cal_Vext_s0_device, 
                    modify_frame_a_cal_Vext_s0_device, modify_frame_b_cal_Vext_s0_device, modify_frame_c_cal_Vext_s0_device,
                    minimum_distance_cal_Vext_s0_device,
                    V_s0_temp);
        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s2, 
            N_string[0], d_offset, d_offset+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s2, 
            N_string[0], d_offset, d_offset+1);
        cudaFree(d_temp_storage);



        

        check_s2<<<(int)((N_string[0]*3-1)/running_block_size+1),running_block_size>>>
        (N_string_device, V_s0, V_s2,
        s0_alpha_device, s0_beta_device, s0_gamma_device, 
        s2_alpha_device, s2_beta_device, s2_gamma_device);










        smooth_angle<<<(int)((N_string[0]*3-1)/running_block_size+1),running_block_size>>>
        (N_string_device, smooth_coeff_device, 
        s2_alpha_device, s2_beta_device, s2_gamma_device, 
        s2_alpha_smooth_device, s2_beta_smooth_device, s2_gamma_smooth_device);

        cudaDeviceSynchronize();
        // return 0;









        // calculate the difference in coordinate and orientation after one iteration
        diff_s_prep<<<(int)((N_string[0]*6-1)/running_block_size+1),running_block_size>>>
        (N_string_device, 

        s0_a_device, s0_b_device, s0_c_device,
        s0_alpha_device, s0_beta_device, s0_gamma_device,

        s2_a_device, s2_b_device, s2_c_device,
        s2_alpha_device, s2_beta_device, s2_gamma_device,

        diff_s_coordinate_all_device, diff_s_orientation_all_device);



        
        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, diff_s_coordinate_all_device, diff_s_coordinate_device, 
            N_string[0], s1_l_sum_separate_offset_device, s1_l_sum_separate_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, diff_s_coordinate_all_device, diff_s_coordinate_device, 
            N_string[0], s1_l_sum_separate_offset_device, s1_l_sum_separate_offset_device+1);
        cudaFree(d_temp_storage);



        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, diff_s_orientation_all_device, diff_s_orientation_device, 
            N_string[0], s1_l_sum_separate_offset_device, s1_l_sum_separate_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, diff_s_orientation_all_device, diff_s_orientation_device, 
            N_string[0], s1_l_sum_separate_offset_device, s1_l_sum_separate_offset_device+1);
        cudaFree(d_temp_storage);

        cudaDeviceSynchronize();
        // check_double_custom<<<1,32>>>
        // (401, diff_s_coordinate_device, diff_s_orientation_device);


        s1_length_sqrt_cal<<<(int)((N_string[0]*2-1)/running_block_size+1),running_block_size>>>
        (rounding_coeff_device, N_string_device, diff_s_coordinate_device, diff_s_orientation_device);


        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, diff_s_coordinate_device, total_diff_s_coordinate_device, 
            1, s1_l_sum_total_offset_device, s1_l_sum_total_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, diff_s_coordinate_device, total_diff_s_coordinate_device, 
            1, s1_l_sum_total_offset_device, s1_l_sum_total_offset_device+1);
        cudaFree(d_temp_storage);



        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, diff_s_orientation_device, total_diff_s_orientation_device, 
            1, s1_l_sum_total_offset_device, s1_l_sum_total_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, diff_s_orientation_device, total_diff_s_orientation_device, 
            1, s1_l_sum_total_offset_device, s1_l_sum_total_offset_device+1);
        cudaFree(d_temp_storage);

        if ((i_time%200)==0)
        {   
            check_signal<<<(int)((2-1)/running_block_size+1),running_block_size>>>
            (N_string_device, 
            total_diff_s_coordinate_device, total_diff_s_orientation_device,
            convergence_coorindate_device, convergence_orientation_device,
            signal_coordinate_device, signal_orientation_device);

            cudaMemcpy(signal_coordinate, signal_coordinate_device, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(signal_orientation, signal_orientation_device, sizeof(int), cudaMemcpyDeviceToHost);

            if ((signal_coordinate[0]==1)&&(signal_orientation[0]==1))
            {
                break;
            }
        }





        copy2s0<<<(int)((N_string[0]*6-1)/running_block_size+1),running_block_size>>>
        (N_string_device, 
        signal_coordinate_device, signal_orientation_device,

        s2_a_device, s2_b_device, s2_c_device, 
        s2_alpha_smooth_device, s2_beta_smooth_device, s2_gamma_smooth_device, 
        s0_a_device, s0_b_device, s0_c_device, 
        s0_alpha_device, s0_beta_device, s0_gamma_device);

        cudaDeviceSynchronize();
        
    }

    if ((signal_coordinate[0]==1)&&(signal_orientation[0]==1))
    {
        // printf("converged!\n");
        printf("info:\t1\t%d\t%d\t", i_time, N_atom_frame[0]*times);
    }
    else
    {
        // printf("timed out\n");
        printf("info:\t0\t%d\t%d\t", i_time, N_atom_frame[0]*times);
    }

    // return 0;


    Vext_cal_s0<<<(int)((N_string[0]*times*N_atom_adsorbate[0]*N_atom_frame[0]-1)/running_block_size+1),running_block_size>>>
    (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device, 
                charge_adsorbate_device, 
    vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
                N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
                charge_frame_device, 
                frac_a_frame_device, frac_b_frame_device, frac_c_frame_device, 
                times_x_device, times_y_device, times_z_device, 
                cart_x_extended_device, cart_y_extended_device, cart_z_extended_device, 
                frac2car_a_device, frac2car_b_device, frac2car_c_device, 
                cutoff_device, damping_a_device, 
                temp_add_frame_device, 


                // int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                // double *delta_grid_device, double *delta_angle_device,
                N_string_device,

                s0_a_device, s0_b_device, s0_c_device, 
                s0_alpha_device, s0_beta_device, s0_gamma_device,


                index_s0_cal_Vext_s0_device,
                // int *index_a_cal_Vext_s0_device, int *index_b_cal_Vext_s0_device, int *index_c_cal_Vext_s0_device,
                // int *index_alpha_cal_Vext_s0_device, int *index_beta_cal_Vext_s0_device, int *index_gamma_cal_Vext_s0_device,
                index_adsorbate_cal_Vext_s0_device, index_frame_cal_Vext_s0_device,

                a_cal_Vext_s0_device, b_cal_Vext_s0_device, c_cal_Vext_s0_device,
                alpha_rad_cal_Vext_s0_device, beta_rad_cal_Vext_s0_device, gamma_rad_cal_Vext_s0_device,
                loc_x_cal_Vext_s0_device, loc_y_cal_Vext_s0_device, loc_z_cal_Vext_s0_device,
                vector_adsorbate_x_rot_cal_Vext_s0_device, vector_adsorbate_y_rot_cal_Vext_s0_device, vector_adsorbate_z_rot_cal_Vext_s0_device,
                adsorbate_cart_x_rot_cal_Vext_s0_device, adsorbate_cart_y_rot_cal_Vext_s0_device, adsorbate_cart_z_rot_cal_Vext_s0_device, 
                modify_frame_a_cal_Vext_s0_device, modify_frame_b_cal_Vext_s0_device, modify_frame_c_cal_Vext_s0_device,
                minimum_distance_cal_Vext_s0_device,
                V_s0_temp);

    // h_offset = (int *) malloc(sizeof(int)*(N_string[0]+1));
    // h_offset[0] = 0;
    // for (i=1; i<=N_string[0]; i++)
    // {
    //     h_offset[i] = i*N_atom_adsorbate[0]*N_atom_frame[0]*times;
    // }
    // cudaMalloc((void**)&d_offset, (N_string[0]+1)*sizeof(int));
    // cudaMemcpy(d_offset, h_offset, (N_string[0]+1)*sizeof(int), cudaMemcpyHostToDevice);
    // free(h_offset);

    d_temp_storage = NULL;
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s0, 
        N_string[0], d_offset, d_offset+1);
    cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s0, 
        N_string[0], d_offset, d_offset+1);
    cudaFree(d_temp_storage);
    // cudaFree(d_offset);







    cudaDeviceSynchronize();
    // // print xyz
    // check_double_special<<<1,32>>>
    // (401, N_atom_adsorbate_device,

    // frac2car_a_device, frac2car_b_device, frac2car_c_device,

    // vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,

    // s0_a_device, s0_b_device, s0_c_device, 
    // s0_alpha_device, s0_beta_device, s0_gamma_device);
    // cudaDeviceSynchronize();

    t = clock() - t;



    cudaMemcpy(s0_a_final, s0_a_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_b_final, s0_b_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_c_final, s0_c_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_alpha_final, s0_alpha_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_beta_final, s0_beta_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_gamma_final, s0_gamma_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(V_s0_2, V_s0, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    for (i=0; i<N_string[0]; i++)
    {
        frac2car(s0_a_final[i], s0_b_final[i], s0_c_final[i], frac2car_a, frac2car_b, frac2car_c, s0_cart_x, s0_cart_y, s0_cart_z);
        s0_x[i] = s0_cart_x[0]*1e-10;
        s0_y[i] = s0_cart_y[0]*1e-10;
        s0_z[i] = s0_cart_z[0]*1e-10;
    }
    for (i=0; i<N_string[0]; i++)
    {
        if (i==0)
        {
            s0[i] = 0;
        }
        else
        {
            s0[i] = s0[i-1] + sqrt( pow((s0_x[i]-s0_x[i-1]), 2) + pow((s0_y[i]-s0_y[i-1]), 2) + pow((s0_z[i]-s0_z[i-1]), 2) );
        }

        if ((V_s0_2[i]/T)>6e2)
        {
            V_s0_treated[i] = exp(-6e2);
        }
        else
        {
            V_s0_treated[i] = exp(-V_s0_2[i]/T);
        }   
    }
    // printf("length: %.5e\n", s0[N_string[0]-1]);
    switch (direction[0])
    {
        case 1:
            D_2 = 0.5 * pow((La*1e-10), 2) * sqrt((kb*T)/(2*PI*(total_mass_adsorbate/1e3/6.02214076e23))) *  
            (  exp( -max(V_s0_2, N_string[0])/T ) / trapz(s0, V_s0_treated, N_string[0]) );
            break;
        case 2:
            D_2 = 0.5 * pow((Lb*1e-10), 2) * sqrt((kb*T)/(2*PI*(total_mass_adsorbate/1e3/6.02214076e23))) *  
            (  exp( -max(V_s0_2, N_string[0])/T ) / trapz(s0, V_s0_treated, N_string[0]) );
            break;
        case 3:
            D_2 = 0.5 * pow((Lc*1e-10), 2) * sqrt((kb*T)/(2*PI*(total_mass_adsorbate/1e3/6.02214076e23))) *  
            (  exp( -max(V_s0_2, N_string[0])/T ) / trapz(s0, V_s0_treated, N_string[0]) );
            break;
    }



    if (argc==3)
    {
        fp1 =fopen(argv[2], "w+");
        if (D_1>D_2)
        {
            for (i=0; i<N_string[0]; i++)
            {
                fprintf(fp1, "%lf %lf %lf %lf %lf %lf %lf\n", s0_a_ini[i], s0_b_ini[i], s0_c_ini[i], s0_alpha_ini[i], s0_beta_ini[i], s0_gamma_ini[i], V_s0_1[i]);
            }
        }
        else
        {
            for (i=0; i<N_string[0]; i++)
            {
                fprintf(fp1, "%lf %lf %lf %lf %lf %lf %lf\n", s0_a_final[i], s0_b_final[i], s0_c_final[i], s0_alpha_final[i], s0_beta_final[i], s0_gamma_final[i], V_s0_2[i]);
            }
        }
        
        fclose(fp1);
    }
    else if (argc==4)
    {
        fp1 =fopen(argv[3], "w+");
        if (D_1>D_2)
        {
            for (i=0; i<N_string[0]; i++)
            {
                fprintf(fp1, "%lf %lf %lf %lf %lf %lf %lf\n", s0_a_ini[i], s0_b_ini[i], s0_c_ini[i], s0_alpha_ini[i], s0_beta_ini[i], s0_gamma_ini[i], V_s0_1[i]);
            }
        }
        else
        {
            for (i=0; i<N_string[0]; i++)
            {
                fprintf(fp1, "%lf %lf %lf %lf %lf %lf %lf\n", s0_a_final[i], s0_b_final[i], s0_c_final[i], s0_alpha_final[i], s0_beta_final[i], s0_gamma_final[i], V_s0_2[i]);
            }
        }
        fclose(fp1);
    }

    printf("%lf\n", ((double)t)/CLOCKS_PER_SEC);

}
