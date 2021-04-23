#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cub/cub.cuh>

#define PI 3.141592653589793
#define running_block_size 32
#define coul2Klevin 1.6710095663e+05




//convert fractional coordinate to cartesian coordinate
void frac2car(double frac_a, double frac_b, double frac_c, double *frac2car_a, double *frac2car_b, double *frac2car_c,
                double *cart_x, double *cart_y, double *cart_z)
{
    cart_x[0] = frac_a*frac2car_a[0] + frac_b*frac2car_a[1] + frac_c*frac2car_a[2];
    cart_y[0] = frac_a*frac2car_b[0] + frac_b*frac2car_b[1] + frac_c*frac2car_b[2];
    cart_z[0] = frac_a*frac2car_c[0] + frac_b*frac2car_c[1] + frac_c*frac2car_c[2];
}



//expand the lattice to a larger size
void pbc_expand(int *N_atom_frame, int *times_x, int *times_y, int *times_z, double *frac_a_frame, double *frac_b_frame, double *frac_c_frame,
                double *epsilon_frame, double *sigma_frame, double *charge_frame, double *mass_frame)
{
    int i, ii, iii, iiii;
    int j;
    iiii = 0;
    for (j=0; j<N_atom_frame[0]; j++)
    {
        for (i=0; i<times_x[0]; i++)
        {
            for (ii=0; ii<times_y[0]; ii++)
            {
                for (iii=0; iii<times_z[0]; iii++)
                {
                    if ((i!=0)||(ii!=0)||(iii!=0))
                    {
                        frac_a_frame[N_atom_frame[0]+iiii] = frac_a_frame[j] + i;
                        frac_b_frame[N_atom_frame[0]+iiii] = frac_b_frame[j] + ii;
                        frac_c_frame[N_atom_frame[0]+iiii] = frac_c_frame[j] + iii;
                        epsilon_frame[N_atom_frame[0]+iiii] = epsilon_frame[j];
                        sigma_frame[N_atom_frame[0]+iiii] = sigma_frame[j];
                        charge_frame[N_atom_frame[0]+iiii] = charge_frame[j];
                        mass_frame[N_atom_frame[0]+iiii] = mass_frame[j];
                        iiii++;
                    }
                }
            }
        }
    }
}







void rotate_moleucle(int *N_atom_adsorbate, double vector_adsorbate_x[], double vector_adsorbate_y[], double vector_adsorbate_z[], 
                        double rot_alpha_angle, double rot_beta_angle, double rot_gamma_angle, double vector_adsorbate_x_rot[], 
                        double vector_adsorbate_y_rot[], double vector_adsorbate_z_rot[])
{
    double rot_alpha_rad, rot_beta_rad, rot_gamma_rad;
    int i, ii, index;

    //convert angles to rad
    rot_alpha_rad = rot_alpha_angle/180*PI;
    rot_beta_rad = rot_beta_angle/180*PI;
    rot_gamma_rad = rot_gamma_angle/180*PI;

    for (i=0; i<N_atom_adsorbate[0]; i++)
    {
        vector_adsorbate_x_rot[i] = vector_adsorbate_x[i]*cos(rot_gamma_rad)*cos(rot_beta_rad) 
            - vector_adsorbate_y[i]*sin(rot_gamma_rad)*cos(rot_alpha_rad) + vector_adsorbate_y[i]*sin(rot_alpha_rad)*cos(rot_gamma_rad)*sin(rot_beta_rad) 
            + vector_adsorbate_z[i]*sin(rot_gamma_rad)*sin(rot_alpha_rad) + vector_adsorbate_z[i]*cos(rot_alpha_rad)*cos(rot_gamma_rad)*sin(rot_beta_rad);

        vector_adsorbate_y_rot[i] = vector_adsorbate_x[i]*sin(rot_gamma_rad)*cos(rot_beta_rad) 
            + vector_adsorbate_y[i]*cos(rot_gamma_rad)*cos(rot_alpha_rad) + vector_adsorbate_y[i]*sin(rot_gamma_rad)*sin(rot_beta_rad)*sin(rot_alpha_rad)
            - vector_adsorbate_z[i]*cos(rot_gamma_rad)*sin(rot_alpha_rad) + vector_adsorbate_z[i]*sin(rot_gamma_rad)*sin(rot_beta_rad)*cos(rot_alpha_rad);

        vector_adsorbate_z_rot[i] = -vector_adsorbate_x[i]*sin(rot_beta_rad) + vector_adsorbate_y[i]*cos(rot_beta_rad)*sin(rot_alpha_rad) 
            + vector_adsorbate_z[i]*cos(rot_alpha_rad)*cos(rot_beta_rad);
    }
}

